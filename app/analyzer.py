"""Motor de análise musical.

Extrai de um arquivo de áudio: metadados, BPM/batidas/compasso, tonalidade,
loudness (LUFS), energia, timbre e estrutura (seções). As batidas usam o modelo
neural Beat This! na GPU quando disponível; o resto usa librosa (CPU).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter

import librosa
import numpy as np
import pyloudnorm as pyln
from scipy.cluster.hierarchy import fcluster, linkage

from . import ai_models

ANALYSIS_VERSION = 8
log = logging.getLogger("analyzer")

SR = 22050          # taxa para features
SR_LOUD = 44100     # taxa para loudness
HOP = 512

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCHES_PT = ["Dó", "Dó#", "Ré", "Ré#", "Mi", "Fá", "Fá#", "Sol", "Sol#", "Lá", "Lá#", "Si"]
# Perfis de Krumhansl-Schmuckler
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

AUDIO_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus", ".aac", ".aiff", ".aif"}

# --------------------------------------------------------------------------- #
# Batidas (GPU)
# --------------------------------------------------------------------------- #
_beat_model = None
_beat_lock = threading.Lock()
_beat_backend = "librosa"


def gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"cuda": True, "device": torch.cuda.get_device_name(0), "torch": torch.__version__}
        return {"cuda": False, "device": "CPU", "torch": torch.__version__}
    except Exception as e:  # torch ausente
        return {"cuda": False, "device": "CPU", "error": str(e)}


def _get_beat_model():
    global _beat_model, _beat_backend
    if _beat_model is None:
        try:
            import torch
            from beat_this.inference import Audio2Beats
            device = ai_models._device()
            _beat_model = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
            _beat_backend = f"beat_this ({device})"
            log.info("Modelo de batidas carregado: %s", _beat_backend)
        except Exception as e:
            log.warning("Beat This! indisponível (%s); usando librosa.", e)
            _beat_model = False
    return _beat_model


def detect_beats(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, str]:
    model = _get_beat_model()
    if model:
        try:
            with ai_models.GPU_LOCK:
                beats, downbeats = model(y.astype(np.float32), sr)
                ai_models.free_gpu()
            beats, downbeats = np.asarray(beats, float), np.asarray(downbeats, float)
            if len(beats) >= 8:
                return beats, downbeats, _beat_backend
        except Exception as e:
            log.warning("Falha no Beat This!, caindo para librosa: %s", e)
    _, frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    beats = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    return beats, beats[::4], "librosa"


def clean_downbeats(db: np.ndarray) -> np.ndarray:
    """Remove inícios de compasso espúrios (muito próximos do anterior)."""
    if len(db) < 4:
        return db
    bar = float(np.median(np.diff(db)))
    keep = [db[0]]
    for t in db[1:]:
        if t - keep[-1] >= 0.6 * bar:
            keep.append(t)
    # preenche compassos que o modelo pulou (buraco de 1,5+ compasso)
    filled = [keep[0]]
    for t in keep[1:]:
        gap = t - filled[-1]
        n = int(round(gap / bar))
        if n >= 2:
            filled += [filled[-1] + gap * k / n for k in range(1, n)]
        filled.append(t)
    return np.array(filled)


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def camelot(pitch: int, mode: str) -> str:
    major_root = pitch if mode == "major" else (pitch + 3) % 12
    num = ((major_root * 7) % 12 + 7) % 12 + 1
    return f"{num}{'B' if mode == 'major' else 'A'}"


def estimate_key(chroma_vec: np.ndarray) -> dict:
    c = chroma_vec / (chroma_vec.sum() + 1e-9)
    scores = []
    for p in range(12):
        scores.append((np.corrcoef(c, np.roll(KS_MAJOR, p))[0, 1], p, "major"))
        scores.append((np.corrcoef(c, np.roll(KS_MINOR, p))[0, 1], p, "minor"))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    r, p, mode = best
    return {
        "key": f"{PITCHES[p]} {'major' if mode == 'major' else 'minor'}",
        "key_short": PITCHES[p] + ("" if mode == "major" else "m"),
        "key_pt": f"{PITCHES_PT[p]} {'maior' if mode == 'major' else 'menor'}",
        "pitch": p,
        "mode": mode,
        "camelot": camelot(p, mode),
        "confidence": round(float(max(0.0, min(1.0, (r - second[0]) * 5 + r * 0.5))), 2),
        "alternative": f"{PITCHES[second[1]]}{'' if second[2] == 'major' else 'm'}",
    }


# --------------------------------------------------------------------------- #
# Acordes
# --------------------------------------------------------------------------- #
CHORD_NAMES = PITCHES + [p + "m" for p in PITCHES] + ["N"]


def _chord_templates() -> np.ndarray:
    T = []
    for third in (4, 3):
        for p in range(12):
            t = np.zeros(12)
            t[[p, (p + third) % 12, (p + 7) % 12]] = [1.0, 0.8, 0.8]
            T.append(t)
    T = np.array(T)
    return T / np.linalg.norm(T, axis=1, keepdims=True)


def recognize_chords(C: np.ndarray, energy: np.ndarray, grid_times: np.ndarray,
                     duration: float, stay: float = 0.7) -> list[dict]:
    """C: chroma sincronizado na batida (12 x N). Viterbi sobre 24 tríades + N (sem acorde)."""
    n = C.shape[1]
    Cn = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)
    sim = _chord_templates() @ Cn                                   # 24 x N
    quiet = energy < (np.percentile(energy, 95) * 0.08)
    sim_n = np.where(quiet, 1.0, 0.5)[None, :]
    S = np.vstack([sim, sim_n]) * 12
    logp = S - np.log(np.exp(S).sum(axis=0, keepdims=True))           # log-softmax
    k = S.shape[0]
    trans = np.full((k, k), np.log((1 - stay) / (k - 1)))
    np.fill_diagonal(trans, np.log(stay))
    dp, bp = logp[:, 0].copy(), np.zeros((k, n), int)
    for t in range(1, n):
        cand = dp[:, None] + trans
        bp[:, t] = cand.argmax(axis=0)
        dp = cand.max(axis=0) + logp[:, t]
    path = np.zeros(n, int)
    path[-1] = dp.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = bp[path[t], t]
    out = []
    for i, s in enumerate(path):
        t0 = float(grid_times[i])
        t1 = float(grid_times[i + 1]) if i + 1 < len(grid_times) else duration
        if out and out[-1]["chord"] == CHORD_NAMES[s]:
            out[-1]["end"] = round(t1, 2)
        else:
            out.append({"start": round(t0, 2), "end": round(t1, 2), "chord": CHORD_NAMES[s]})
    return out


def chord_at(chords: list[dict], t0: float, t1: float) -> str:
    """Acorde que ocupa mais tempo no intervalo."""
    best, dur = "N", 0.0
    acc: dict[str, float] = {}
    for c in chords:
        ov = min(c["end"], t1) - max(c["start"], t0)
        if ov > 0:
            acc[c["chord"]] = acc.get(c["chord"], 0) + ov
    for ch, d in acc.items():
        if d > dur:
            best, dur = ch, d
    return best


def main_progression(bar_chords: list[str]) -> dict | None:
    seq = [c for c in bar_chords]
    best = None
    for n in (4, 8, 2):
        cnt = Counter(tuple(seq[i:i + n]) for i in range(0, len(seq) - n + 1))
        for gram, c in cnt.most_common(5):
            if "N" in gram or len(set(gram)) < 2 or c < 2:
                continue
            if best is None or c * n > best[1] * len(best[0]) * 1.3:
                best = (gram, c)
            break
    if not best:
        return None
    gram, reps = list(best[0]), best[1]
    while len(gram) % 2 == 0 and len(gram) > 2 and gram[: len(gram) // 2] == gram[len(gram) // 2:]:
        gram = gram[: len(gram) // 2]   # "Am F C G Am F C G" -> "Am F C G"
        reps *= 2
    rle = []
    for c in gram:                       # "D#m D#m D#m A#m" -> [["D#m", 3], ["A#m", 1]]
        if rle and rle[-1][0] == c:
            rle[-1][1] += 1
        else:
            rle.append([c, 1])
    return {"chords": gram, "repeats": reps, "rle": rle}


def collapse(seq: list[str], limit: int = 16) -> str:
    out = []
    for c in seq:
        if not out or out[-1] != c:
            out.append(c)
    s = " ".join(out[:limit])
    return s + (" …" if len(out) > limit else "")


def detect_accents(onset_env: np.ndarray, duration: float) -> list[dict]:
    """Ataques percussivos mais fortes (bons para cortes/flashes no vídeo)."""
    from scipy.signal import find_peaks
    dist = max(1, int(0.25 * SR / HOP))
    pk, _ = find_peaks(onset_env, distance=dist, prominence=np.percentile(onset_env, 90) * 0.5)
    if not len(pk):
        return []
    # distribui pela música inteira: os 2 mais fortes de cada janela de 8 s
    # (antes, os mais fortes da faixa toda se concentravam num trecho só)
    pt = librosa.frames_to_time(pk, sr=SR, hop_length=HOP)
    floor = float(np.percentile(onset_env[pk], 60))
    chosen = []
    for w0 in np.arange(0, duration, 8.0):
        inw = pk[(pt >= w0) & (pt < w0 + 8.0)]
        inw = inw[onset_env[inw] >= floor]
        chosen += list(inw[np.argsort(-onset_env[inw])[:2]])
    if not chosen:
        return []
    top = np.array(sorted(chosen))
    mx = float(onset_env[top].max() + 1e-9)
    t = librosa.frames_to_time(top, sr=SR, hop_length=HOP)
    return [{"t": round(float(a), 3), "strength": round(float(onset_env[f] / mx), 2)} for a, f in zip(t, top)]


def curve_mean(curve: list[float], a: float, b: float) -> float:
    """Média da curva de energia (1 valor/s) no intervalo [a, b). É a ÚNICA escala de energia
    usada em seções, eventos e no prompt, para não haver números que pareçam contraditórios."""
    if b <= a:
        return 0.0
    ts = np.arange(a, b, 0.1)
    return float(np.interp(ts, np.arange(len(curve)) + 0.5, curve).mean())


def detect_events(curve: list[float], downbeats: np.ndarray, duration: float) -> list[dict]:
    """Subidas, quedas, pausas, build-ups e clímax, alinhados a compassos."""
    edges = list(downbeats) + [duration]
    if len(edges) < 3:
        return []
    e = np.array([curve_mean(curve, a, b) for a, b in zip(edges[:-1], edges[1:])])
    ev = []
    for i in range(1, len(e)):
        d = e[i] - e[i - 1]
        if d > 0.18:
            j = i - 1
            while j > 0 and e[j] > e[j - 1] + 0.02:
                j -= 1
            if i - j >= 3:
                ev.append({"t": round(float(edges[j]), 3), "end": round(float(edges[i]), 3), "type": "build-up",
                           "desc": f"energia crescendo por {i - j} compassos até a subida"})
            ev.append({"t": round(float(edges[i]), 3), "type": "subida" if d < 0.35 else "drop/explosão",
                       "desc": f"energia salta de {e[i - 1]:.2f} para {e[i]:.2f}", "bar": i + 1})
        elif d < -0.18:
            ev.append({"t": round(float(edges[i]), 3), "type": "queda/quebra",
                       "desc": f"energia cai de {e[i - 1]:.2f} para {e[i]:.2f}", "bar": i + 1})
    quiet = e < 0.12
    i = 0
    while i < len(e):
        if quiet[i]:
            j = i
            while j + 1 < len(e) and quiet[j + 1]:
                j += 1
            if 0 < i or j < len(e) - 1:
                ev.append({"t": round(float(edges[i]), 3), "end": round(float(edges[j + 1]), 3),
                           "type": "pausa/silêncio", "desc": f"{j - i + 1} compasso(s) quase sem som", "bar": i + 1})
            i = j + 1
        else:
            i += 1
    k = int(np.argmax(e))
    ev.append({"t": round(float(edges[k]), 3), "end": round(float(edges[k + 1]), 3), "type": "clímax",
               "desc": "compasso de maior energia da música", "bar": k + 1})
    return sorted(ev, key=lambda x: x["t"])


def detect_drums(drums: np.ndarray) -> dict:
    """Bumbo, caixa e chimbal a partir do stem de bateria (por banda de frequência; aproximado)."""
    from scipy.signal import find_peaks
    S = np.abs(librosa.stft(drums, hop_length=HOP))
    f = librosa.fft_frequencies(sr=SR)
    out = {}
    for name, lo, hi, gap in (("kick", 20, 150, 0.2), ("snare", 150, 4000, 0.15), ("hihat", 6000, 11000, 0.08)):
        band = S[(f >= lo) & (f < hi)]
        env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(band + 1e-9, ref=np.max), sr=SR, hop_length=HOP)
        if not len(env) or env.max() <= 0:
            out[name] = []
            continue
        pk, _ = find_peaks(env, distance=max(1, int(gap * SR / HOP)), height=env.max() * 0.3,
                           prominence=np.percentile(env, 95) * 0.4)
        out[name] = [round(float(t), 3) for t in librosa.frames_to_time(pk, sr=SR, hop_length=HOP)]
    # o bumbo vaza para a banda da caixa: descarta "caixas" coladas num bumbo
    kick = np.array(out.get("kick") or [-9])
    out["snare"] = [t for t in out.get("snare", []) if np.min(np.abs(kick - t)) > 0.04]
    return out


def stem_envelopes(signals: dict[str, np.ndarray], fps: int = 30) -> dict:
    """Intensidade de cada stem quadro a quadro (0–1), para animação reagir ao som."""
    hop = SR // fps
    out = {"fps": fps}
    for k, v in signals.items():
        r = librosa.feature.rms(y=v, frame_length=2048, hop_length=hop, center=True)[0]
        out[k] = np.clip(r / (np.percentile(r, 99) + 1e-9), 0, 1).round(2).tolist()
    return out


def stem_level(db: float) -> str:
    return "em destaque" if db > -6 else ("presente" if db > -14 else ("ao fundo" if db > -24 else "ausente"))


def read_metadata(path: str) -> dict:
    meta = {"title": None, "artist": None, "album": None, "year": None, "genre_tag": None,
            "bitrate_kbps": None, "sample_rate": None, "channels": None}
    try:
        import mutagen
        f = mutagen.File(path, easy=True)
        if f is not None:
            tags = f.tags or {}

            def first(k):
                v = tags.get(k)
                return str(v[0]).strip() if v else None
            meta.update(title=first("title"), artist=first("artist"), album=first("album"),
                        year=(first("date") or "")[:4] or None, genre_tag=first("genre"))
            info = getattr(f, "info", None)
            if info is not None:
                br = getattr(info, "bitrate", None)
                meta["bitrate_kbps"] = int(br / 1000) if br else None
                meta["sample_rate"] = getattr(info, "sample_rate", None)
                meta["channels"] = getattr(info, "channels", None)
    except Exception as e:
        log.debug("metadados: %s", e)
    if not meta["title"]:
        meta["title"] = os.path.splitext(os.path.basename(path))[0]
    return meta


def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-9))


def _level(v, lo, hi):
    return "baixa" if v < lo else ("alta" if v > hi else "média")


# --------------------------------------------------------------------------- #
# Estrutura
# --------------------------------------------------------------------------- #
def _novelty(X: np.ndarray, w: int) -> np.ndarray:
    """Curva de novidade (Foote) sobre a matriz de autossimilaridade."""
    Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-9)
    S = Xn.T @ Xn
    n = S.shape[0]
    g = np.exp(-0.5 * (np.linspace(-2, 2, 2 * w) ** 2))
    K = np.outer(g, g) * np.kron(np.array([[1, -1], [-1, 1]]), np.ones((w, w)))
    Sp = np.pad(S, w, mode="edge")
    nov = np.array([np.sum(Sp[i:i + 2 * w, i:i + 2 * w] * K) for i in range(n)])
    nov -= nov.min()
    return nov / (nov.max() + 1e-9)


def segment_structure(X: np.ndarray, grid_times: np.ndarray, duration: float,
                      downbeats: np.ndarray, beats_per_bar: int = 4) -> list[tuple[float, float, int]]:
    """Retorna [(início, fim, rótulo_numérico)] via novidade + agrupamento de seções."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    from scipy.spatial.distance import pdist, squareform

    n = X.shape[1]
    Xs = uniform_filter1d(X, size=beats_per_bar, axis=1)
    w = int(np.clip(beats_per_bar * 4, 8, max(8, n // 6)))
    nov = _novelty(Xs, w)
    peaks, _ = find_peaks(nov, distance=beats_per_bar * 4, prominence=0.08)
    peaks = peaks[(peaks > beats_per_bar * 2) & (peaks < n - beats_per_bar * 2)]
    max_b = int(np.clip(duration / 12, 2, 24))
    if len(peaks) > max_b:
        peaks = np.sort(peaks[np.argsort(-nov[peaks])[:max_b]])
    times = sorted(set([0.0] + [float(grid_times[i]) for i in peaks]))
    # alinhar ao downbeat mais próximo (até ~1 s)
    if len(downbeats):
        snapped = []
        for t in times:
            j = np.argmin(np.abs(downbeats - t))
            snapped.append(float(downbeats[j]) if abs(downbeats[j] - t) < 1.0 and t > 0 else t)
        times = sorted(set(snapped))
    times = [t for t in times if t < duration - 1] + [duration]
    if times[0] > 0:
        times = [0.0] + times
    # juntar seções muito curtas (< 6 s)
    segs = [[times[i], times[i + 1]] for i in range(len(times) - 1)]
    merged = []
    for s in segs:
        if merged and (s[1] - s[0] < 6 or merged[-1][1] - merged[-1][0] < 6):
            merged[-1][1] = s[1]
        else:
            merged.append(s)

    # rótulos por similaridade de timbre/harmonia
    vecs = []
    for s0, s1 in merged:
        m = (grid_times >= s0) & (grid_times < s1)
        seg = X[:, m] if m.any() else X
        vecs.append(np.concatenate([seg.mean(axis=1), 0.5 * seg.std(axis=1)]))
    V = np.array(vecs)
    if len(V) >= 3:
        D = squareform(pdist(V))
        np.fill_diagonal(D, np.inf)
        nn = float(np.median(D.min(axis=1)))   # distância típica até a seção "gêmea"
        Z = linkage(V, method="average")
        raw = fcluster(Z, t=2.0 * nn, criterion="distance")
        hi_k = max(2, int(np.ceil(len(V) * 0.6)))
        lo_k = max(2, int(round(len(V) / 3.5)))   # música real raramente tem 1 só tipo de seção
        if len(set(raw)) > hi_k:
            raw = fcluster(Z, t=hi_k, criterion="maxclust")
        elif len(set(raw)) < lo_k:
            raw = fcluster(Z, t=lo_k, criterion="maxclust")
    elif len(V) == 2:
        raw = np.array([1, 2])
    else:
        raw = np.array([1])
    remap, labels = {}, []
    for r in raw:
        remap.setdefault(r, len(remap))
        labels.append(remap[r])
    return [(float(a), float(b), int(l)) for (a, b), l in zip(merged, labels)]


def name_sections(segs, seg_energy):
    letters = [chr(ord("A") + l) if l < 26 else f"S{l}" for _, _, l in segs]
    counts = Counter(l for _, _, l in segs)
    lab_energy = {l: np.mean([e for (_, _, ll), e in zip(segs, seg_energy) if ll == l]) for l in counts}
    repeated = [l for l in counts if counts[l] >= 2]
    chorus = max(repeated, key=lambda l: lab_energy[l] + 0.05 * counts[l]) if repeated else None
    others = [l for l in repeated if l != chorus]
    verse = max(others, key=lambda l: counts[l]) if others else None
    med = float(np.median(seg_energy))
    # timbre uniforme (comum em EDM): um grupo domina; separa por energia dentro dele
    dominant = chorus is not None and len(segs) >= 5 and counts[chorus] >= 0.5 * len(segs)
    dom_med = float(np.median([e for (_, _, l), e in zip(segs, seg_energy) if l == chorus])) if dominant else 0
    names = []
    for i, ((s0, s1, l), e) in enumerate(zip(segs, seg_energy)):
        if i == 0 and l != chorus and e < 0.85 * med:
            nm = "Intro"
        elif i == len(segs) - 1 and l != chorus and e < 0.85 * med and len(segs) > 2:
            nm = "Outro"
        elif l == chorus and dominant:
            nm = "Refrão" if e >= dom_med else "Verso"
        elif l == chorus:
            nm = "Refrão"
        elif l == verse:
            nm = "Verso"
        elif counts[l] == 1 and 0 < i < len(segs) - 1:
            nm = "Ponte"
        else:
            nm = f"Seção {letters[i]}"
        names.append(nm)
    return letters, names


# --------------------------------------------------------------------------- #
# Análise principal
# --------------------------------------------------------------------------- #
def analyze(path: str) -> dict:
    t0 = time.time()
    meta = read_metadata(path)

    stereo, _ = librosa.load(path, sr=SR_LOUD, mono=False)
    if stereo.ndim == 1:
        stereo = np.vstack([stereo, stereo])
    duration = stereo.shape[1] / SR_LOUD
    if duration < 5:
        raise ValueError("Áudio muito curto (< 5 s)")

    # ---- Loudness -------------------------------------------------------- #
    meter = pyln.Meter(SR_LOUD)
    lufs = float(meter.integrated_loudness(stereo.T))
    win, hop_s = 3 * SR_LOUD, SR_LOUD
    st = []
    for s in range(0, stereo.shape[1] - win, hop_s):
        v = meter.integrated_loudness(stereo[:, s:s + win].T)
        if np.isfinite(v) and v > -70:
            st.append(v)
    st = np.array(st) if st else np.array([lufs])
    lra = float(np.percentile(st, 95) - np.percentile(st, 10))
    peak = float(np.max(np.abs(stereo)))
    peak_db = float(_db(peak))
    rms_all_db = float(_db(np.sqrt(np.mean(stereo ** 2))))
    clipped = int(np.sum(np.abs(stereo) >= 0.999))
    side = (stereo[0] - stereo[1]) / 2
    mid = (stereo[0] + stereo[1]) / 2
    stereo_width = float(np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) + 1e-9))

    # ---- Separação de stems (Demucs, GPU) -------------------------------- #
    stems = None
    try:
        stems = ai_models.separate(stereo, SR_LOUD)
    except Exception as e:
        log.warning("Demucs falhou: %s", e)

    y = librosa.resample(librosa.to_mono(stereo), orig_sr=SR_LOUD, target_sr=SR)
    del stereo, side, mid
    stems22 = {k: librosa.resample(v, orig_sr=SR_LOUD, target_sr=SR) for k, v in stems.items()} if stems else None
    del stems

    # ---- Batidas / tempo ------------------------------------------------- #
    beats, downbeats, beat_backend = detect_beats(y, SR)
    downbeats = clean_downbeats(downbeats)
    ibi = np.diff(beats)
    if len(ibi):
        # média dos intervalos "normais": a mediana sofre com a grade de 20 ms do modelo
        med = np.median(ibi)
        ok = ibi[(ibi > med * 0.85) & (ibi < med * 1.15)]
        bpm = float(60 / ok.mean())
    else:
        bpm = 0.0
    local_bpm = 60 / ibi if len(ibi) else np.array([bpm])
    tempo_std = float(np.std(local_bpm[(local_bpm > bpm * 0.8) & (local_bpm < bpm * 1.2)])) if len(ibi) else 0
    beats_per_bar = []
    for a, b in zip(downbeats[:-1], downbeats[1:]):
        beats_per_bar.append(int(np.sum((beats >= a - 0.03) & (beats < b - 0.03))))
    meter_n = Counter([b for b in beats_per_bar if 2 <= b <= 7]).most_common(1)
    time_sig = f"{meter_n[0][0]}/4" if meter_n else "4/4"

    # ---- Features espectrais -------------------------------------------- #
    S = np.abs(librosa.stft(y, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR)
    power = S ** 2
    # participação por banda em amplitude (potência faz os graves esmagarem o resto)
    tot = S.sum() + 1e-9
    bands = {
        "sub (<60 Hz)": S[freqs < 60].sum() / tot,
        "graves (60–250 Hz)": S[(freqs >= 60) & (freqs < 250)].sum() / tot,
        "médios-graves (250–2k Hz)": S[(freqs >= 250) & (freqs < 2000)].sum() / tot,
        "médios-agudos (2k–6k Hz)": S[(freqs >= 2000) & (freqs < 6000)].sum() / tot,
        "agudos (>6k Hz)": S[freqs >= 6000].sum() / tot,
    }
    y_h, y_p = librosa.effects.hpss(y)
    perc_ratio = float(np.sum(y_p ** 2) / (np.sum(y_h ** 2) + np.sum(y_p ** 2) + 1e-9))
    rms = librosa.feature.rms(S=S, hop_length=HOP)[0]
    centroid = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    onset_env = librosa.onset.onset_strength(y=y_p, sr=SR, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR, hop_length=HOP)
    onset_density = float(len(onsets) / duration)
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=SR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=power, sr=SR)),
                                n_mfcc=13)
    del S, power, y_h, y_p

    # chroma "harmônico": sem bateria e com o baixo reforçando a fundamental
    if stems22:
        ch_h = librosa.feature.chroma_cqt(y=stems22["other"] + 0.2 * stems22["vocals"], sr=SR, hop_length=HOP)
        ch_b = librosa.feature.chroma_cqt(y=stems22["bass"], sr=SR, hop_length=HOP)
        n_c = min(ch_h.shape[1], ch_b.shape[1])
        nz = lambda c: c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)
        chord_chroma = nz(ch_h[:, :n_c]) + 1.0 * nz(ch_b[:, :n_c])
    else:
        chord_chroma = chroma
    key = estimate_key(chord_chroma.mean(axis=1))

    # ---- Estrutura ------------------------------------------------------- #
    beat_frames = librosa.time_to_frames(beats, sr=SR, hop_length=HOP)
    beat_frames = beat_frames[beat_frames < chroma.shape[1]]
    n_fr = min(chroma.shape[1], mfcc.shape[1], len(rms), len(onset_env))
    loud = librosa.amplitude_to_db(rms[:n_fr] + 1e-6)[None, :]
    feat = np.vstack([chroma[:, :n_fr], mfcc[:, :n_fr], loud, loud, loud, onset_env[None, :n_fr]])
    feat = (feat - feat.mean(axis=1, keepdims=True)) / (feat.std(axis=1, keepdims=True) + 1e-9)
    Xs = librosa.util.sync(feat, beat_frames, aggregate=np.median)
    grid_times = np.concatenate([[0.0], librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP)])
    grid_times = grid_times[: Xs.shape[1]]
    bpb = int(time_sig.split("/")[0])
    segs = segment_structure(Xs, grid_times, duration, downbeats, bpb)

    frame_t = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=HOP)
    rms_max = float(rms.max() + 1e-9)

    # curva de energia (pico por segundo, 0–1): base de toda a escala de energia
    n_sec = int(np.ceil(duration))
    sec_idx = np.minimum((frame_t).astype(int), n_sec - 1)
    energy_curve = np.zeros(n_sec)
    np.maximum.at(energy_curve, sec_idx, rms)
    energy_curve = (energy_curve / (energy_curve.max() + 1e-9)).round(3).tolist()

    accents = detect_accents(onset_env, duration)
    events = detect_events(energy_curve, downbeats, duration)

    # ---- Acordes --------------------------------------------------------- #
    bf = beat_frames[beat_frames < chord_chroma.shape[1]]
    Cb = librosa.util.sync(chord_chroma, bf, aggregate=np.median)
    Eb = librosa.util.sync(rms[None, : chord_chroma.shape[1]], bf, aggregate=np.mean)[0]
    gt = np.concatenate([[0.0], librosa.frames_to_time(bf, sr=SR, hop_length=HOP)])[: Cb.shape[1]]
    chords = recognize_chords(Cb, Eb, gt, duration)
    bar_edges = list(downbeats) + [duration]
    bar_chords = [chord_at(chords, a, b) for a, b in zip(bar_edges[:-1], bar_edges[1:])]
    progression = main_progression(bar_chords)
    chord_counts = Counter()
    for c in chords:
        if c["chord"] != "N":
            chord_counts[c["chord"]] += c["end"] - c["start"]

    # ---- Stems por seção ------------------------------------------------- #
    stem_rms = {}
    if stems22:
        for k_, v in stems22.items():
            stem_rms[k_] = librosa.feature.rms(y=v, hop_length=HOP)[0][: len(rms)]
    voc_share = 0.0
    has_vocals = False
    if "vocals" in stem_rms:
        tot_e = sum(float(np.sum(v ** 2)) for v in stem_rms.values()) + 1e-9
        voc_share = float(np.sum(stem_rms["vocals"] ** 2) / tot_e)
        vdb = 20 * np.log10(stem_rms["vocals"] / (rms[: len(stem_rms["vocals"])] + 1e-9) + 1e-9)
        has_vocals = voc_share > 0.02 and float(np.mean(vdb > -14)) > 0.08

    # ---- Instrumentos / gênero / clima (AST, GPU) ------------------------ #
    tags = None
    try:
        y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
        tags = ai_models.tag_windows(y16)
    except Exception as e:
        log.warning("AST falhou: %s", e)
    instruments = genres = moods = []
    if tags:
        probs, labels, win_starts = tags
        instruments = ai_models.summarize_tags(probs, labels, ai_models.INSTRUMENTS, 0.03, 12)
        genres = ai_models.summarize_tags(probs, labels, ai_models.GENRES, 0.03, 6)
        moods = ai_models.summarize_tags(probs, labels, ai_models.MOODS, 0.02, 3)

    # ---- Letra (Whisper, GPU) -------------------------------------------- #
    lyrics = None
    regions = []
    if has_vocals:
        try:
            regions = ai_models.vocal_regions(stems22["vocals"], SR)
            voc16 = librosa.resample(stems22["vocals"], orig_sr=SR, target_sr=16000)
            lyrics = ai_models.transcribe(voc16, regions)
        except Exception as e:
            log.warning("Whisper falhou: %s", e)
    # ---- Dados para animação (dança/reação ao som) ---------------------- #
    drums_hits, envelopes = {}, None
    if stems22:
        try:
            drums_hits = detect_drums(stems22["drums"])
            envelopes = stem_envelopes({"mix": y, **{k_: v for k_, v in stems22.items()}})
        except Exception as e:
            log.warning("bateria/envelopes falharam: %s", e)

    seg_energy, sections = [], []
    for s0, s1, _ in segs:
        seg_energy.append(curve_mean(energy_curve, s0, s1))
    letters, names = name_sections(segs, seg_energy)
    e_lo, e_hi = np.percentile(seg_energy, 33), np.percentile(seg_energy, 66)
    for (s0, s1, _), e, letter, nm in zip(segs, seg_energy, letters, names):
        m = (frame_t >= s0) & (frame_t < s1)
        sb = beats[(beats >= s0) & (beats < s1)]
        mc = m[: chord_chroma.shape[1]]
        sk = estimate_key(chord_chroma[:, mc].mean(axis=1)) if mc.any() else key
        sec_stems = {}
        for k_, v in stem_rms.items():
            mm = m[: len(v)]
            if mm.any():
                sec_stems[k_] = stem_level(float(_db(v[mm].mean() / (rms[: len(v)][mm].mean() + 1e-9))))
        sec_instr = []
        if tags:
            wm = (win_starts + 10 > s0) & (win_starts < s1)
            if wm.any():
                sec_instr = [d["label"] for d in ai_models.summarize_tags(
                    probs[wm], labels, ai_models.INSTRUMENTS, 0.05, 6)]
        sec_bars = [c for (a, b), c in zip(zip(bar_edges[:-1], bar_edges[1:]), bar_chords) if a >= s0 - 0.05 and a < s1 - 0.05]
        sec_lyrics = [l["text"] for l in (lyrics or {}).get("lines", []) if s0 <= (l["start"] + l["end"]) / 2 < s1]
        sections.append({
            "stems": sec_stems, "instruments": sec_instr,
            "chords": collapse(sec_bars) if sec_bars else "",
            "lyrics": sec_lyrics,
            "start": round(s0, 2), "end": round(s1, 2), "label": letter, "name": nm,
            "bars": int(np.sum((downbeats >= s0) & (downbeats < s1))),
            "energy": round(e, 3),
            "energy_level": "baixa" if e <= e_lo else ("alta" if e >= e_hi else "média"),
            "loudness_db": round(float(_db(rms[m].mean())) if m.any() else -60, 1),
            "brightness_hz": int(centroid[m].mean()) if m.any() else 0,
            "onset_density": round(float(np.sum((frame_t[onsets] >= s0) & (frame_t[onsets] < s1)) / max(s1 - s0, 1e-3)), 2),
            "bpm": round(float(60 / np.median(np.diff(sb))), 1) if len(sb) > 4 else round(bpm, 1),
            "local_key": sk["key_short"],
        })

    # ---- Curvas para a interface ---------------------------------------- #
    n_peaks = 1600
    chunk = max(1, len(y) // n_peaks)
    peaks = np.abs(y[: chunk * n_peaks].reshape(-1, chunk)).max(axis=1)
    peaks = (peaks / (peaks.max() + 1e-9)).round(3).tolist()

    # ---- Descritores globais -------------------------------------------- #
    cent_mean = float(centroid.mean())
    energy_score = float(np.clip(
        0.4 * np.clip((lufs + 30) / 22, 0, 1) + 0.3 * np.clip(onset_density / 5, 0, 1)
        + 0.3 * np.clip(perc_ratio / 0.6, 0, 1), 0, 1))
    descriptors = {
        "brilho": "escuro/quente" if cent_mean < 1500 else ("brilhante" if cent_mean > 2800 else "equilibrado"),
        "peso_graves": _level(bands["sub (<60 Hz)"] + bands["graves (60–250 Hz)"], 0.2, 0.4),
        "percussividade": _level(perc_ratio, 0.25, 0.45),
        "densidade_ritmica": _level(onset_density, 1.5, 3.5),
        "estabilidade_tempo": "estável (provável click/eletrônico)" if tempo_std < 1.0
        else ("levemente variável" if tempo_std < 3 else "variável (provável tocado ao vivo)"),
        "dinamica": "muito comprimida" if lra < 4 else ("dinâmica moderada" if lra < 9 else "bem dinâmica"),
        "abertura_estereo": "quase mono" if stereo_width < 0.1 else ("aberta" if stereo_width > 0.45 else "moderada"),
    }

    # vetor para "músicas parecidas"
    vector = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1), chroma.mean(axis=1),
        [bpm / 200, perc_ratio, onset_density / 5, cent_mean / 4000, float(flatness.mean()) * 10,
         lufs / -30, lra / 15, stereo_width],
        list(bands.values()),
    ]).astype(float)

    return {
        "version": ANALYSIS_VERSION,
        "meta": meta,
        "harmony": {
            "chords": chords,
            "bar_chords": bar_chords,
            "progression": progression,
            "most_used": [c for c, _ in chord_counts.most_common(8)],
        },
        "instrumentation": {
            "has_vocals": has_vocals,
            "vocal_share": round(voc_share, 3),
            "stems_global": {k_: round(float(np.sum(v ** 2) / (sum(float(np.sum(x ** 2)) for x in stem_rms.values()) + 1e-9)), 3)
                             for k_, v in stem_rms.items()},
            "instruments": instruments,
            "genres": genres,
            "moods": moods,
        },
        "lyrics": lyrics,
        "accents": accents,
        "events": events,
        "drums": drums_hits,
        "vocal_regions": [[round(a, 2), round(b, 2)] for a, b in regions],
        "envelopes": envelopes,
        "duration": round(duration, 2),
        "tempo": {"bpm": round(bpm, 1), "bpm_std": round(tempo_std, 2), "time_signature": time_sig,
                  "beats": [round(float(b), 3) for b in beats],
                  "downbeats": [round(float(b), 3) for b in downbeats],
                  "backend": beat_backend},
        "key": key,
        "loudness": {"lufs": round(lufs, 1), "lra": round(lra, 1), "peak_dbfs": round(peak_db, 2),
                     "rms_dbfs": round(rms_all_db, 1), "crest_db": round(peak_db - rms_all_db, 1),
                     "clipped_samples": clipped, "stereo_width": round(stereo_width, 3)},
        "timbre": {"centroid_hz": int(cent_mean), "percussive_ratio": round(perc_ratio, 3),
                   "onset_density": round(onset_density, 2), "flatness": round(float(flatness.mean()), 4),
                   "bands": {k: round(float(v), 3) for k, v in bands.items()}},
        "energy_score": round(energy_score, 3),
        "descriptors": descriptors,
        "sections": sections,
        "energy_curve": energy_curve,
        "peaks": peaks,
        "vector": [round(float(v), 5) for v in vector],
        "analysis_seconds": round(time.time() - t0, 1),
    }
