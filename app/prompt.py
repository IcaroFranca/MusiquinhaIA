"""Gera prompts em texto para o Claude compreender uma música a partir da análise."""
from __future__ import annotations

import json


def _t(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def _s(x: float) -> str:
    return f"{x:.2f}s"


# --------------------------------------------------------------------------- #
# Briefing de vídeo (o que o usuário escolhe no formulário)
# --------------------------------------------------------------------------- #
BRIEF_OPTIONS = {
    "tipo": {
        "historia": "Videoclipe com história e personagens",
        "danca": "Videoclipe de dança/coreografia (personagens dançando no ritmo)",
        "abstrato": "Visualizer abstrato (formas, luz, partículas reagindo ao som)",
        "lyric": "Lyric video (a letra animada é o foco)",
        "livre": "Livre: escolha o que combina melhor com a música",
    },
    "tecnica": {
        "livre": "Livre (escolha a ferramenta)",
        "web": "HTML: Canvas para 2D, three.js para 3D e SVG onde fizer sentido",
        "hyperframes": "HyperFrames (HTML + GSAP renderizado em vídeo)",
        "remotion": "Remotion (React)",
        "ae": "After Effects (expressões/JSX)",
    },
    "formato": {
        "16:9": (1920, 1080),
        "9:16": (1080, 1920),
        "1:1": (1080, 1080),
        "4:5": (1080, 1350),
    },
}

DEFAULT_BRIEF = {"tipo": "livre", "tecnica": "livre", "formato": "16:9", "fps": 30,
                 "so_motion": True, "letra_na_tela": True}


def _video_task(brief: dict, has_lyrics: bool, files: bool) -> str:
    fps = int(brief.get("fps") or 30)
    show_lyrics = has_lyrics and brief.get("letra_na_tela", True)
    steps = [
        "Proponha um conceito visual criativo e único, coerente com o gênero, o clima, a energia"
        + (" e a letra." if has_lyrics else "."),
        "Escreva um storyboard por seção, com os tempos exatos de início e fim de cada cena.",
        "Corte as cenas nos inícios de compasso (lista \"Compassos\"): cortes rápidos nas seções de energia "
        "alta, planos mais longos nas de energia baixa.",
        "Coloque transições, flashes, zooms ou mudanças de cor nos momentos de impacto (subidas, drops, quebras, "
        "clímax) e nos acentos fortes. Use os build-ups para aumentar a tensão visual.",
    ]
    if show_lyrics:
        steps.append("Mostre a letra sincronizada usando os tempos de cada linha"
                     + (" e de cada palavra (timing.json → lyrics[].words) para efeito karaokê." if files else ".")
                     + " Linhas marcadas como \"vocalização repetida\" são uma sílaba cantada muitas vezes: "
                       "anime a sílaba, não escreva a contagem.")
    elif has_lyrics:
        steps.append("Não escreva a letra na tela; use-a só como inspiração para as imagens.")
    if files:
        steps.append("Para movimento que reage ao som (passos de dança, pulsos, boca cantando), use do timing.json: "
                     "drums.kick / drums.snare / drums.hihat (tempo de cada batida), vocal_regions (trechos com voz) "
                     "e envelopes (intensidade 0–1 de cada stem, 30 valores por segundo).")
    steps.append("Faça a intensidade visual (movimento, brilho, saturação, velocidade) acompanhar a curva de energia.")
    return "\n".join(
        ["Quero que você crie um videoclipe completo sincronizado com esta música. O arquivo de áudio original "
         "é a trilha do vídeo, mas você não consegue ouvi-lo: confie nos tempos acima, medidos no próprio áudio "
         "(precisão de ~20 ms)."]
        + [f"{i}. {s}" for i, s in enumerate(steps, 1)]
        + [f"Se gerar código, use os tempos do JSON diretamente e converta para quadros com "
           f"quadro = round(segundos × {fps}). Não estime tempos \"de ouvido\" e não altere a duração do áudio."])


GOALS = {
    "video": {"label": "Videoclipe sincronizado", "task": None},
    "entender": {
        "label": "Entender a música",
        "task": (
            "Com base nesses dados, descreva como essa música provavelmente soa: gênero e subgênero "
            "prováveis, clima/emoção, tipo de arranjo e instrumentação plausível, e o arco emocional ao "
            "longo das seções. Se houver letra, interprete os temas e como ela se encaixa na harmonia e na "
            "energia de cada seção. Explique o que cada métrica sugere e aponte o que é incerto. Termine com "
            "um parágrafo curto que descreva a faixa para alguém que nunca a ouviu."
        ),
    },
    "producao": {
        "label": "Feedback de mix/master",
        "task": (
            "Atue como engenheiro de mixagem e masterização. Avalie loudness (LUFS, LRA, true peak "
            "aproximado), equilíbrio espectral, dinâmica e imagem estéreo frente às referências de "
            "streaming (Spotify/YouTube ~-14 LUFS, Apple Music ~-16 LUFS) e ao gênero provável. "
            "Liste problemas prováveis e ajustes concretos (EQ, compressão, limiter, estéreo), em ordem de prioridade."
        ),
    },
    "dj": {
        "label": "DJ / mixagem ao vivo",
        "task": (
            "Atue como DJ. Sugira pontos de cue, onde entrar e sair numa transição (em compassos), "
            "faixas de BPM e tons compatíveis (roda de Camelot) para mixar antes/depois, e em que "
            "momento do set essa faixa funciona melhor, usando a curva de energia."
        ),
    },
    "composicao": {
        "label": "Composição / remix",
        "task": (
            "Atue como produtor e compositor. Com base na estrutura, tom e energia, proponha ideias "
            "de remix, variações de arranjo, progressões de acordes compatíveis com o tom, e como "
            "reorganizar as seções para aumentar o impacto. Seja concreto (compassos, instrumentos, BPM)."
        ),
    },
}


# --------------------------------------------------------------------------- #
# JSON de tempos
# --------------------------------------------------------------------------- #
def build_timing(a: dict, filename: str | None = None, full: bool = True) -> dict:
    """Mapa de tempo em JSON. full=True é o timing.json (com batidas, palavras, bateria e envelopes);
    full=False é a versão compacta que vai dentro do prompt."""
    lyr = a.get("lyrics") or {}
    t = {
        "audio_file": filename,
        "duration": a["duration"],
        "bpm": a["tempo"]["bpm"],
        "time_signature": a["tempo"]["time_signature"],
        "beat_seconds": round(60 / a["tempo"]["bpm"], 4) if a["tempo"]["bpm"] else None,
        "downbeats": a["tempo"]["downbeats"],
        "sections": [{"start": s["start"], "end": s["end"], "name": s["name"], "group": s["label"],
                      "energy": round(s["energy"], 2), "energy_level": s["energy_level"]} for s in a["sections"]],
        "events": a.get("events", []),
        "accents": a.get("accents", []),
        "lyrics": [{"start": l["start"], "end": l["end"], "text": l["text"],
                    **({"repeat": l["repeat"]} if l.get("repeat") else {}),
                    **({"part": l["part"]} if l.get("part") else {}),
                    **({"words": l["words"]} if full and l.get("words") else {})}
                   for l in lyr.get("lines", [])],
        "energy_per_second": [round(v, 2) for v in a["energy_curve"]],
    }
    if full:
        t["beats"] = a["tempo"]["beats"]
        if a.get("drums"):
            t["drums"] = a["drums"]
        if a.get("vocal_regions") is not None:
            t["vocal_regions"] = a["vocal_regions"]
        if a.get("envelopes"):
            t["envelopes"] = a["envelopes"]
    return t


def _video_blocks(a: dict, lyrics: bool, filename: str | None, files: bool) -> list[str]:
    tp = a["tempo"]
    db = tp["downbeats"]
    out = [
        "",
        "## Mapa de tempo para edição (segundos)",
        f"- Duração exata do áudio: {a['duration']:.2f}s",
        f"- BPM {tp['bpm']}, compasso {tp['time_signature']}: 1 batida ≈ {60 / tp['bpm']:.3f}s, "
        f"1 compasso ≈ {60 / tp['bpm'] * int(tp['time_signature'].split('/')[0]):.3f}s"
        + (f"; primeira batida em {tp['beats'][0]:.2f}s" if tp.get("beats") else ""),
        "- Compassos (nº:início): " + " ".join(f"{i + 1}:{t:.2f}" for i, t in enumerate(db)),
        "",
        "### Seções (tempo exato)",
    ]
    for s in a["sections"]:
        out.append(f"- {_s(s['start'])} → {_s(s['end'])}: {s['name']} (grupo {s['label']}), "
                   f"energia {s['energy_level']} ({s['energy']:.2f})")
    ev = a.get("events") or []
    if ev:
        out += ["", "### Momentos de impacto"]
        for e in ev:
            span = f"{_s(e['t'])} → {_s(e['end'])}" if e.get("end") else _s(e["t"])
            out.append(f"- {span}: {e['type']}, {e['desc']}")
    acc = a.get("accents") or []
    if acc:
        out += ["", "### Acentos percussivos fortes, distribuídos pela música (bons para flash/corte seco)",
                " ".join(_s(x["t"]) for x in acc)]
    dr = a.get("drums") or {}
    if dr.get("kick") or dr.get("snare"):
        out += ["", "### Bateria (do stem isolado; aproximado)",
                f"- Bumbo: {len(dr.get('kick', []))} batidas | caixa/palma: {len(dr.get('snare', []))} | "
                f"chimbal: {len(dr.get('hihat', []))}"
                + (" — tempos de cada uma no timing.json" if files else "")]
    lyr = a.get("lyrics") or {}
    if lyrics and lyr.get("lines"):
        src = "letra oficial fornecida pelo usuário, sincronizada pelo áudio" if lyr.get("source") == "colada" else \
              "corrigida pelo usuário" if lyr.get("edited") else \
              f"transcrição automática Whisper, idioma {lyr.get('language') or '?'}; pode conter erros de audição"
        out += ["", f"### Letra sincronizada ({src})"]
        part = None
        for l in lyr["lines"]:
            if l.get("part") and l["part"] != part:
                part = l["part"]
                out.append(f"- [{part}]")
            out.append(f"- {_s(l['start'])} → {_s(l['end'])}: {l['text']}")
    t = build_timing(a, filename, full=False)
    if not lyrics:
        t.pop("lyrics", None)
    out += ["", "### Mesmos dados em JSON (para usar em código)", "```json",
            json.dumps(t, ensure_ascii=False, separators=(",", ":")), "```"]
    return out


def _brief_block(brief: dict, filename: str | None, files: bool) -> list[str]:
    tipo = BRIEF_OPTIONS["tipo"].get(brief.get("tipo"), BRIEF_OPTIONS["tipo"]["livre"])
    tec = BRIEF_OPTIONS["tecnica"].get(brief.get("tecnica"), BRIEF_OPTIONS["tecnica"]["livre"])
    fmt = brief.get("formato", "16:9")
    w, h = BRIEF_OPTIONS["formato"].get(fmt, (1920, 1080))
    out = ["", "## Briefing do vídeo",
           f"- Tipo: {tipo}",
           f"- Técnica: {tec}",
           f"- Formato: {w}×{h} ({fmt}) a {int(brief.get('fps') or 30)} fps, duração igual à da música"]
    if brief.get("so_motion", True):
        out.append("- Feito só com motion graphics (2D, 3D ou ambos): sem filmagens reais nem fotos")
    if files:
        out.append(f"- Arquivos nesta pasta: {filename or 'o áudio'} (trilha) e timing.json "
                   "(todos os tempos, letra palavra a palavra, bateria e envelopes por stem)")
    return out


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
STEM_PT = {"vocals": "voz", "drums": "bateria", "bass": "baixo", "other": "outros (harmonia/melodia)"}


def build_prompt(a: dict, goal: str = "video", extra: str | None = None, lyrics: bool = True,
                 filename: str | None = None, brief: dict | None = None, files: bool = False) -> str:
    """files=True quando o prompt vai num pacote exportado (áudio + timing.json na mesma pasta)."""
    g = GOALS.get(goal, GOALS["video"])
    video = goal == "video"
    brief = {**DEFAULT_BRIEF, **(brief or {})}
    m, tp, k, ld, tb, d = a["meta"], a["tempo"], a["key"], a["loudness"], a["timbre"], a["descriptors"]

    lines = [
        "Vou te passar uma análise técnica automática de uma música (você não tem acesso ao áudio, "
        "apenas a estes dados extraídos por algoritmos de análise de áudio). Os valores são "
        "estimativas: tonalidade e nomes de seções podem estar errados.",
        "",
        "## Identificação",
        f"- Título: {m.get('title') or 'desconhecido'}",
    ]
    for lbl, key_ in (("Artista", "artist"), ("Álbum", "album"), ("Ano", "year"), ("Gênero (tag do arquivo)", "genre_tag")):
        if m.get(key_):
            lines.append(f"- {lbl}: {m[key_]}")
    lines.append(f"- Duração: {_t(a['duration'])}")

    lines += [
        "",
        "## Ritmo e harmonia",
        f"- BPM: {tp['bpm']} (variação ±{tp['bpm_std']} BPM → {d['estabilidade_tempo']})",
        f"- Compasso estimado: {tp['time_signature']}",
        f"- Tonalidade estimada: {k['key']} / {k['key_pt']} (Camelot {k['camelot']}, "
        f"confiança {int(k['confidence'] * 100)}%, alternativa: {k['alternative']})",
        f"- Densidade rítmica: {tb['onset_density']} ataques/s ({d['densidade_ritmica']})",
        f"- Percussividade: {int(tb['percussive_ratio'] * 100)}% da energia é percussiva ({d['percussividade']})",
        "",
        "## Som e mixagem",
        f"- Loudness integrado: {ld['lufs']} LUFS | faixa de loudness (LRA): {ld['lra']} LU ({d['dinamica']})",
        f"- Pico: {ld['peak_dbfs']} dBFS | fator de crista: {ld['crest_db']} dB | amostras clipadas: {ld['clipped_samples']}",
        f"- Brilho (centroide espectral): {tb['centroid_hz']} Hz ({d['brilho']})",
        f"- Peso dos graves: {d['peso_graves']} | imagem estéreo: {d['abertura_estereo']} (side/mid {ld['stereo_width']})",
        "- Distribuição de energia por banda: "
        + ", ".join(f"{b} {int(v * 100)}%" for b, v in tb["bands"].items()),
    ]

    ins = a.get("instrumentation")
    if ins:
        lines += ["", "## Instrumentação (detectada por IA)"]
        lines.append(f"- Voz: {'sim' if ins['has_vocals'] else 'não detectada (provavelmente instrumental)'}")
        if ins.get("stems_global"):
            lines.append("- Participação na energia (separação Demucs): " + ", ".join(
                f"{STEM_PT.get(k, k)} {int(v * 100)}%" for k, v in ins["stems_global"].items()))
        if ins.get("instruments"):
            lines.append("- Sons/instrumentos reconhecidos (modelo AudioSet, confiança média): " + ", ".join(
                f"{d['label']} ({int(d['score'] * 100)}%)" for d in ins["instruments"]))
        if ins.get("genres"):
            lines.append("- Gêneros sugeridos pelo modelo: " + ", ".join(
                f"{d['label']} ({int(d['score'] * 100)}%)" for d in ins["genres"])
                + " (o modelo não conhece gêneros brasileiros como sertanejo, pagode, funk carioca, forró ou MPB)")
        if ins.get("moods"):
            lines.append("- Clima sugerido: " + ", ".join(d["label"] for d in ins["moods"]))

    h = a.get("harmony")
    if h:
        lines += ["", "## Harmonia (acordes detectados automaticamente; só tríades maiores/menores)"]
        if h.get("progression"):
            p = h["progression"]
            rle = p.get("rle") or [[c, 1] for c in p["chords"]]
            seq = " → ".join(f"{c} ({n} compassos)" if n > 1 else c for c, n in rle)
            lines.append(f"- Progressão principal (ciclo de {len(p['chords'])} compassos, repete {p['repeats']}x): {seq}")
        if h.get("most_used"):
            lines.append("- Acordes mais usados: " + ", ".join(h["most_used"]))

    lines += [
        "",
        "## Estrutura (seções detectadas automaticamente)",
        "Letras iguais = seções com material musical parecido. "
        + ("Nomes de seção vêm dos cabeçalhos da letra oficial (onde há canto); nas partes instrumentais são palpites. "
           if any("name_auto" in s for s in a["sections"]) else "Nomes de seção são palpites heurísticos. ")
        + "Energia em escala 0–1 (1 = momento mais forte da música), a mesma usada na curva e nos eventos.",
    ]
    for s in a["sections"]:
        lines += ["", f"### {_t(s['start'])}–{_t(s['end'])} · {s['name']} (grupo {s['label']}) · {s['bars']} compassos"]
        lines.append(f"- Energia {s['energy_level']} ({s['energy']:.2f}), brilho {s['brightness_hz']} Hz, "
                     f"{s['onset_density']} ataques/s, tom local {s['local_key']}")
        if s.get("stems"):
            lines.append("- Stems: " + ", ".join(f"{STEM_PT.get(k, k)} {v}" for k, v in s["stems"].items()))
        if s.get("instruments"):
            lines.append("- Sons: " + ", ".join(s["instruments"]))
        if s.get("chords"):
            lines.append(f"- Acordes por compasso: {s['chords']}")
        # no modo vídeo a letra vai uma vez só, sincronizada, mais abaixo
        if lyrics and not video and s.get("lyrics"):
            lines.append("- Letra: " + " / ".join(f"\"{l}\"" for l in s["lyrics"]))

    curve = a["energy_curve"]
    step = 5 if a["duration"] < 420 else 10
    pts = [f"{_t(i)}={max(curve[i:i + step]):.2f}" for i in range(0, len(curve), step)]
    lines += ["", f"## Curva de energia (pico 0–1, a cada {step} s)", " ".join(pts)]

    if video:
        lines += _brief_block(brief, filename, files)
        lines += _video_blocks(a, lyrics, filename, files)
        has_lyr = bool(lyrics and (a.get("lyrics") or {}).get("lines"))
        task = _video_task(brief, has_lyr, files)
    else:
        task = g["task"]
    lines += ["", "## Tarefa", task]
    if extra:
        lines += ["", "Pedidos específicos: " + extra.strip()]
    return "\n".join(lines)
