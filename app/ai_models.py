"""Modelos de IA na GPU: separação de stems (Demucs), etiquetas de áudio (AST/AudioSet)
e transcrição de letra (Whisper). Todos carregam sob demanda e compartilham um lock de GPU
para que as análises em paralelo não estourem a VRAM."""
from __future__ import annotations

import logging
import os
import re
import threading

import numpy as np

log = logging.getLogger("ai")

GPU_LOCK = threading.Lock()
_models: dict = {}
_load_lock = threading.Lock()

ENABLE_STEMS = os.environ.get("MUSICA_STEMS", "1") != "0"
ENABLE_TAGS = os.environ.get("MUSICA_TAGS", "1") != "0"
ENABLE_LYRICS = os.environ.get("MUSICA_LYRICS", "1") != "0"
WHISPER_MODEL = os.environ.get("MUSICA_WHISPER", "openai/whisper-large-v3-turbo")
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"


_capped = False


def _device():
    global _capped
    import torch
    if torch.cuda.is_available():
        if not _capped:
            # sem limite, o driver do Windows "transborda" a VRAM para a RAM e trava o PC
            torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MUSICA_VRAM_FRACTION", "0.85")))
            _capped = True
        return "cuda"
    return "cpu"


def free_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get(name, loader):
    if name not in _models:
        with _load_lock:
            if name not in _models:
                try:
                    _models[name] = loader()
                    log.info("Modelo carregado: %s", name)
                except Exception as e:
                    log.exception("Falha ao carregar %s", name)
                    _models[name] = None
    return _models[name]


# --------------------------------------------------------------------------- #
# Demucs
# --------------------------------------------------------------------------- #
def _load_demucs():
    from demucs.pretrained import get_model
    m = get_model("htdemucs")
    m.to(_device()).eval()
    return m


def separate(stereo: np.ndarray, sr: int) -> dict[str, np.ndarray] | None:
    """stereo: (2, T) a 44.1 kHz. Retorna {drums, bass, other, vocals} em mono (T,)."""
    if not ENABLE_STEMS:
        return None
    model = _get("demucs", _load_demucs)
    if model is None:
        return None
    import torch
    from demucs.apply import apply_model
    assert sr == model.samplerate
    wav = torch.from_numpy(np.ascontiguousarray(stereo, dtype=np.float32))
    ref = wav.mean(0)
    mu, sd = ref.mean(), ref.std() + 1e-8
    with GPU_LOCK, torch.inference_mode():
        out = apply_model(model, ((wav - mu) / sd)[None], device=_device(), split=True,
                          overlap=0.25, progress=False)[0]
        if _device() == "cuda":
            torch.cuda.empty_cache()
    out = out * sd + mu
    return {name: out[i].mean(0).cpu().numpy() for i, name in enumerate(model.sources)}


# --------------------------------------------------------------------------- #
# AST (AudioSet): instrumentos, gêneros, clima
# --------------------------------------------------------------------------- #
INSTRUMENTS = {
    "Guitar": "guitarra", "Electric guitar": "guitarra elétrica", "Acoustic guitar": "violão",
    "Bass guitar": "baixo elétrico", "Steel guitar, slide guitar": "guitarra slide",
    "Tapping (guitar technique)": "tapping", "Strum": "violão/guitarra dedilhado-batido",
    "Piano": "piano", "Electric piano": "piano elétrico", "Organ": "órgão", "Hammond organ": "órgão Hammond",
    "Synthesizer": "sintetizador", "Sampler": "sampler", "Harpsichord": "cravo",
    "Keyboard (musical)": "teclado", "Drum kit": "bateria", "Drum machine": "bateria eletrônica",
    "Snare drum": "caixa", "Bass drum": "bumbo", "Hi-hat": "chimbal", "Cymbal": "pratos",
    "Tambourine": "pandeiro/tamborim", "Timpani": "tímpanos", "Tabla": "tabla", "Drum roll": "rufo",
    "Rimshot": "aro", "Cowbell": "cowbell", "Shaker": "chocalho", "Maraca": "maraca", "Gong": "gongo",
    "Percussion": "percussão", "Marimba, xylophone": "marimba/xilofone", "Glockenspiel": "glockenspiel",
    "Vibraphone": "vibrafone", "Steelpan": "steel drum", "Violin, fiddle": "violino",
    "Cello": "violoncelo", "Double bass": "contrabaixo acústico", "String section": "naipe de cordas",
    "Pizzicato": "pizzicato", "Bowed string instrument": "cordas friccionadas", "Orchestra": "orquestra",
    "Brass instrument": "metais", "Trumpet": "trompete", "Trombone": "trombone", "French horn": "trompa",
    "Saxophone": "saxofone", "Flute": "flauta", "Clarinet": "clarinete", "Harmonica": "gaita",
    "Accordion": "acordeom/sanfona", "Bagpipes": "gaita de foles", "Banjo": "banjo", "Mandolin": "bandolim",
    "Ukulele": "ukulele/cavaquinho", "Sitar": "cítara", "Harp": "harpa", "Zither": "cítara",
    "Wind instrument, woodwind instrument": "sopros", "Bell": "sinos", "Choir": "coral",
    "Scratching (performance technique)": "scratch de DJ", "Beatboxing": "beatbox",
    "Singing": "canto", "Male singing": "voz masculina", "Female singing": "voz feminina",
    "Child singing": "voz infantil", "Rapping": "rap (vocal)", "Humming": "cantarolado",
    "Whistling": "assobio", "Yodeling": "yodel", "Chant": "canto coral/mantra", "Speech": "fala",
}
GENRES = {
    "Pop music": "pop", "Hip hop music": "hip hop", "Rock music": "rock", "Heavy metal": "heavy metal",
    "Punk rock": "punk", "Grunge": "grunge", "Progressive rock": "rock progressivo",
    "Rock and roll": "rock and roll", "Psychedelic rock": "rock psicodélico", "Rhythm and blues": "R&B",
    "Soul music": "soul", "Reggae": "reggae", "Country": "country", "Swing music": "swing",
    "Bluegrass": "bluegrass", "Funk": "funk (americano)", "Folk music": "folk",
    "Middle Eastern music": "música do Oriente Médio", "Jazz": "jazz", "Disco": "disco",
    "Classical music": "clássica/erudita", "Opera": "ópera", "Electronic music": "eletrônica",
    "House music": "house", "Techno": "techno", "Dubstep": "dubstep", "Drum and bass": "drum and bass",
    "Electronica": "electronica", "Electronic dance music": "EDM", "Ambient music": "ambient",
    "Trance music": "trance", "Music of Latin America": "música latina", "Salsa music": "salsa",
    "Flamenco": "flamenco", "Blues": "blues", "Music for children": "infantil", "New-age music": "new age",
    "Vocal music": "música vocal", "A capella": "a capella", "Music of Africa": "música africana",
    "Afrobeat": "afrobeat", "Christian music": "música cristã", "Gospel music": "gospel",
    "Music of Asia": "música asiática", "Carnatic music": "música carnática",
    "Music of Bollywood": "Bollywood", "Ska": "ska", "Traditional music": "tradicional/regional",
    "Independent music": "independente/indie", "Video game music": "trilha de videogame",
    "Soundtrack music": "trilha sonora", "Theme music": "tema/vinheta", "Lullaby": "canção de ninar",
    "Background music": "música de fundo", "Dance music": "música dançante",
    "Wedding music": "música de casamento", "Christmas music": "música natalina",
}
MOODS = {
    "Happy music": "alegre", "Funny music": "divertida", "Sad music": "triste", "Tender music": "terna/suave",
    "Exciting music": "empolgante", "Angry music": "raivosa/agressiva", "Scary music": "sombria/assustadora",
}


def _load_ast():
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    fe = ASTFeatureExtractor.from_pretrained(AST_MODEL)
    model = ASTForAudioClassification.from_pretrained(AST_MODEL).to(_device()).eval()
    if _device() == "cuda":
        model = model.half()
    return fe, model


def tag_windows(y16: np.ndarray, win_s: float = 10.0) -> tuple[np.ndarray, list[str], np.ndarray] | None:
    """Classifica janelas de 10 s. Retorna (probs[n_win, n_labels], labels, inicios_s)."""
    if not ENABLE_TAGS:
        return None
    loaded = _get("ast", _load_ast)
    if loaded is None:
        return None
    import torch
    fe, model = loaded
    sr, w = 16000, int(win_s * 16000)
    starts = list(range(0, max(1, len(y16) - w // 2), w))
    chunks = [y16[s:s + w] for s in starts]
    probs = []
    with GPU_LOCK, torch.inference_mode():
        for i in range(0, len(chunks), 16):
            inp = fe(chunks[i:i + 16], sampling_rate=sr, return_tensors="pt")
            x = inp["input_values"].to(_device(), dtype=next(model.parameters()).dtype)
            probs.append(torch.sigmoid(model(input_values=x).logits).float().cpu().numpy())
        free_gpu()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return np.concatenate(probs), labels, np.array(starts) / sr


def summarize_tags(probs: np.ndarray, labels: list[str], table: dict, thr: float, top: int):
    p = probs.mean(axis=0) if probs.ndim == 2 else probs
    pmax = probs.max(axis=0) if probs.ndim == 2 else probs
    out = []
    for i, lab in enumerate(labels):
        if lab in table and (p[i] >= thr or pmax[i] >= thr * 2.5):
            out.append({"label": table[lab], "en": lab, "score": round(float(p[i]), 3),
                        "peak": round(float(pmax[i]), 3)})
    out.sort(key=lambda d: -d["score"])
    return out[:top]


# --------------------------------------------------------------------------- #
# Whisper
# --------------------------------------------------------------------------- #
def _load_whisper():
    import torch
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model=WHISPER_MODEL,
                    dtype=torch.float16 if _device() == "cuda" else torch.float32,
                    device=0 if _device() == "cuda" else -1)


# frases que o Whisper costuma inventar em trechos sem voz
_HALLU = re.compile(
    r"(amara\.org|legendas? pela comunidade|obrigad[oa] por assistir|inscreva-se|subtitles? by|"
    r"thanks? for watching|please subscribe|♪+$|^\W*$|transcri[cç][aã]o|www\.|\.com)", re.I)


def vocal_regions(voc: np.ndarray, sr: int, thr_db: float = -32) -> list[tuple[float, float]]:
    import librosa
    hop = 512
    rms = librosa.feature.rms(y=voc, hop_length=hop)[0]
    db = 20 * np.log10(rms / (rms.max() + 1e-9) + 1e-9)
    act = db > thr_db
    t = librosa.frames_to_time(np.arange(len(act)), sr=sr, hop_length=hop)
    regs, start = [], None
    for ti, a in zip(t, act):
        if a and start is None:
            start = ti
        elif not a and start is not None:
            regs.append([start, ti]); start = None
    if start is not None:
        regs.append([start, t[-1]])
    merged = []
    for r in regs:  # junta pausas curtas
        if merged and r[0] - merged[-1][1] < 1.5:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(max(0, a - 0.3), b + 0.3) for a, b in merged if b - a >= 0.8]


def _norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


def _repeat_text(word: str, n: int) -> str:
    w = re.sub(r"[^\w']", "", word) or word
    return f"{w.capitalize()}, {w.lower()}, {w.lower()}… (vocalização repetida, ~{n}x)"


def compress_loop_text(text: str) -> tuple[str, int]:
    """Linha em que uma palavra domina (loop do Whisper ou vocalização) -> texto curto."""
    toks = [t for t in (_norm(x) for x in text.split()) if t]
    if len(toks) >= 8:
        from collections import Counter
        w, n = Counter(toks).most_common(1)[0]
        if n / len(toks) >= 0.6:
            return _repeat_text(w, n), n
    return text, 0


MIN_LOOP = 8          # repetições seguidas da mesma palavra para virar "vocalização repetida"
MAX_WORDS = 10        # palavras por linha de legenda
MAX_LINE_S = 5.5      # duração máxima de uma linha


def _group_words(blocks, res) -> dict | None:
    """Agrupa palavras em linhas de legenda.

    Quebra nas pausas da voz (≥ 0,6 s sempre; ≥ 0,3 s se a linha já tem 4+ palavras), em
    pontuação final, ou ao passar de 10 palavras / 5,5 s. Sequências longas da mesma palavra
    (loop do Whisper ou "oh, oh, oh…") viram uma única linha resumida.
    """
    langs, words = [], []
    for bi, ((a, b), r) in enumerate(zip(blocks, res)):
        for ch in r.get("chunks") or []:
            if ch.get("language"):
                langs.append(ch["language"])
            w = (ch.get("text") or "").strip()
            ts = ch.get("timestamp") or (None, None)
            if not w or ts[0] is None:
                continue
            s = round(a + ts[0], 2)
            e = round(a + (ts[1] if ts[1] is not None else ts[0] + 0.3), 2)
            words.append({"w": w, "s": s, "e": max(e, s + 0.05), "block": bi})

    lines, cur = [], []

    def flush():
        if cur:
            text = " ".join(w["w"] for w in cur).strip()
            if text and not _HALLU.search(text) and (not lines or lines[-1]["text"] != text):
                lines.append({"start": cur[0]["s"], "end": cur[-1]["e"], "text": text,
                              "words": [{"w": w["w"], "s": w["s"], "e": w["e"]} for w in cur]})
            cur.clear()

    i = 0
    while i < len(words):
        w = words[i]
        # loop: mesma palavra repetida muitas vezes seguidas
        j = i
        while j + 1 < len(words) and _norm(words[j + 1]["w"]) == _norm(w["w"]) != "":
            j += 1
        if j - i + 1 >= MIN_LOOP:
            flush()
            run = words[i:j + 1]
            lines.append({"start": run[0]["s"], "end": run[-1]["e"], "text": _repeat_text(w["w"], len(run)),
                          "repeat": len(run),
                          "words": [{"w": x["w"], "s": x["s"], "e": x["e"]} for x in run[:3]]})
            i = j + 1
            continue
        if cur:
            gap = w["s"] - cur[-1]["e"]
            if (w["block"] != cur[-1]["block"] or gap >= 0.6 or (len(cur) >= 4 and gap >= 0.3)
                    or len(cur) >= MAX_WORDS or w["e"] - cur[0]["s"] > MAX_LINE_S):
                flush()
        cur.append(w)
        if w["w"][-1] in ".!?…" and len(cur) >= 2:
            flush()
        i += 1
    flush()
    if not lines:
        return None
    lang = max(set(langs), key=langs.count) if langs else None
    return {"language": lang, "lines": lines, "word_level": True, "model": WHISPER_MODEL.split("/")[-1]}


def transcribe(voc16: np.ndarray, regions: list[tuple[float, float]]) -> dict | None:
    if not ENABLE_LYRICS or not regions:
        return None
    pipe = _get("whisper", _load_whisper)
    if pipe is None:
        return None
    import torch
    sr = 16000
    # agrupa regiões em blocos de até ~28 s para o Whisper
    blocks, cur = [], None
    for a, b in regions:
        if cur and b - cur[0] <= 28:
            cur[1] = b
        else:
            if cur:
                blocks.append(cur)
            cur = [a, b]
    if cur:
        blocks.append(cur)
    final = []
    for a, b in blocks:  # blocos longos (> 28 s) são fatiados
        while b - a > 28:
            final.append((a, a + 28)); a += 28
        final.append((a, b))
    def mk_inputs():
        return [{"raw": voc16[int(a * sr):int(b * sr)].astype(np.float32), "sampling_rate": sr} for a, b in final]

    def run(mode, idx):
        """Um bloco por vez: no modo palavra o Whisper guarda as atenções e gasta muita VRAM."""
        inputs, out = mk_inputs(), {}
        with GPU_LOCK, torch.inference_mode():
            for i in idx:
                try:
                    out[i] = pipe(inputs[i], return_timestamps=mode, return_language=True,
                                  generate_kwargs={"task": "transcribe"})
                except Exception as e:
                    log.warning("Whisper (%s) falhou no trecho %.1fs: %s", mode, final[i][0], str(e)[:200])
                    free_gpu()
            free_gpu()
        return out

    # 1º: palavra a palavra (legendas karaokê); trechos que falharem são refeitos por frase
    word_res = run("word", range(len(final)))
    ok = sorted(word_res)
    grouped = _group_words([final[i] for i in ok], [word_res[i] for i in ok]) if ok else None
    failed = [i for i in range(len(final)) if i not in word_res]
    if not failed:
        return grouped
    seg = _segment_lines([final[i] for i in failed], [run(True, failed).get(i, {}) for i in failed])
    if not grouped:
        return seg
    if seg:
        grouped["lines"] = sorted(grouped["lines"] + seg["lines"], key=lambda l: l["start"])
    return grouped


def _segment_lines(blocks, res) -> dict | None:
    lines, langs, prev = [], [], None
    for (a, b), r in zip(blocks, res):
        for ch in r.get("chunks") or [{"text": r.get("text", ""), "timestamp": (0, b - a)}]:
            text = (ch.get("text") or "").strip()
            if ch.get("language"):
                langs.append(ch["language"])
            if not text or _HALLU.search(text) or text == prev:
                continue
            ts = ch.get("timestamp") or (0, b - a)
            t0 = a + (ts[0] or 0)
            t1 = a + (ts[1] if ts[1] is not None else b - a)
            text, rep = compress_loop_text(text)
            lines.append({"start": round(t0, 2), "end": round(t1, 2), "text": text, **({"repeat": rep} if rep else {})})
            prev = text
    if not lines:
        return None
    lang = max(set(langs), key=langs.count) if langs else None
    return {"language": lang, "lines": lines, "model": WHISPER_MODEL.split("/")[-1]}
