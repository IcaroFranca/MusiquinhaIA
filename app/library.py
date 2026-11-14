"""Biblioteca (SQLite) e fila de análise em segundo plano."""
from __future__ import annotations

import gc
import json
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import analyzer

log = logging.getLogger("library")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DATA_DIR, "library.db")
os.makedirs(DATA_DIR, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    filename TEXT, size INTEGER, mtime REAL,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | processing | done | error
    error TEXT,
    title TEXT, artist TEXT, album TEXT, year TEXT, genre_tag TEXT,
    duration REAL, bpm REAL, key TEXT, camelot TEXT, mode TEXT, key_confidence REAL,
    lufs REAL, lra REAL, energy REAL, brightness REAL, sections INTEGER,
    analysis TEXT, vector TEXT,
    added_at REAL, analyzed_at REAL
);
CREATE INDEX IF NOT EXISTS ix_status ON tracks(status);
CREATE INDEX IF NOT EXISTS ix_bpm ON tracks(bpm);
CREATE INDEX IF NOT EXISTS ix_camelot ON tracks(camelot);
"""

_local = threading.local()


def conn() -> sqlite3.Connection:
    c = getattr(_local, "c", None)
    if c is None:
        c = sqlite3.connect(DB_PATH, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        _local.c = c
    return c


def init():
    c = conn()
    c.executescript(SCHEMA)
    # análises interrompidas voltam para a fila
    c.execute("UPDATE tracks SET status='pending' WHERE status='processing'")
    c.commit()


# --------------------------------------------------------------------------- #
# Inclusão de arquivos
# --------------------------------------------------------------------------- #
def add_file(path: str) -> bool:
    path = os.path.abspath(path)
    st = os.stat(path)
    c = conn()
    row = c.execute("SELECT id, mtime, size FROM tracks WHERE path=?", (path,)).fetchone()
    if row and row["mtime"] == st.st_mtime and row["size"] == st.st_size:
        return False
    if row:
        c.execute("UPDATE tracks SET status='pending', error=NULL, mtime=?, size=? WHERE id=?",
                  (st.st_mtime, st.st_size, row["id"]))
    else:
        c.execute("INSERT INTO tracks(path, filename, size, mtime, title, added_at) VALUES(?,?,?,?,?,?)",
                  (path, os.path.basename(path), st.st_size, st.st_mtime,
                   os.path.splitext(os.path.basename(path))[0], time.time()))
    c.commit()
    return True


def scan_folder(folder: str) -> dict:
    if not os.path.isdir(folder):
        raise FileNotFoundError(folder)
    found = added = 0
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in analyzer.AUDIO_EXTS:
                found += 1
                try:
                    added += add_file(os.path.join(root, f))
                except OSError as e:
                    log.warning("ignorando %s: %s", f, e)
    worker.wake()
    return {"found": found, "queued": added}


# --------------------------------------------------------------------------- #
# Consulta
# --------------------------------------------------------------------------- #
LIST_COLS = ("id, filename, status, error, title, artist, album, year, genre_tag, duration, bpm, key, "
             "camelot, mode, key_confidence, lufs, lra, energy, brightness, sections")
SORTS = {"title": "title COLLATE NOCASE", "artist": "artist COLLATE NOCASE", "bpm": "bpm",
         "key": "camelot", "lufs": "lufs", "energy": "energy", "duration": "duration", "added": "added_at"}


def list_tracks(q=None, bpm_min=None, bpm_max=None, camelot=None, status=None,
                sort="added", desc=True, half_double=False, limit=500, offset=0):
    where, args = [], []
    if q:
        where.append("(title LIKE ? OR artist LIKE ? OR album LIKE ? OR filename LIKE ? OR genre_tag LIKE ?)")
        args += [f"%{q}%"] * 5
    if bpm_min is not None or bpm_max is not None:
        lo, hi = bpm_min or 0, bpm_max or 999
        cond = "(bpm BETWEEN ? AND ?)"
        a = [lo, hi]
        if half_double:  # aceita metade/dobro do BPM
            cond = "(bpm BETWEEN ? AND ? OR bpm BETWEEN ? AND ? OR bpm BETWEEN ? AND ?)"
            a = [lo, hi, lo / 2, hi / 2, lo * 2, hi * 2]
        where.append(cond)
        args += a
    if camelot:
        keys = [k.strip().upper() for k in camelot.split(",") if k.strip()]
        where.append(f"camelot IN ({','.join('?' * len(keys))})")
        args += keys
    if status:
        where.append("status=?")
        args.append(status)
    sql = f"SELECT {LIST_COLS} FROM tracks"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY {SORTS.get(sort, 'added_at')} {'DESC' if desc else 'ASC'} LIMIT ? OFFSET ?"
    rows = conn().execute(sql, args + [limit, offset]).fetchall()
    total = conn().execute("SELECT COUNT(*) FROM tracks" + (" WHERE " + " AND ".join(where) if where else ""),
                           args).fetchone()[0]
    return {"total": total, "items": [dict(r) for r in rows]}


def get_track(tid: int) -> dict | None:
    r = conn().execute("SELECT * FROM tracks WHERE id=?", (tid,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["analysis"] = json.loads(d["analysis"]) if d["analysis"] else None
    d.pop("vector", None)
    return d


def stats() -> dict:
    rows = conn().execute("SELECT status, COUNT(*) n FROM tracks GROUP BY status").fetchall()
    counts = {r["status"]: r["n"] for r in rows}
    return {"counts": counts, "total": sum(counts.values()), "current": list(worker.current.values()),
            "workers": worker.n_workers, "gpu": analyzer.gpu_info(),
            "beat_backend": analyzer._beat_backend}


def camelot_neighbors(code: str) -> list[str]:
    n, l = int(code[:-1]), code[-1]
    up, down = n % 12 + 1, (n - 2) % 12 + 1
    return [code, f"{up}{l}", f"{down}{l}", f"{n}{'A' if l == 'B' else 'B'}"]


def similar(tid: int, n: int = 12) -> list[dict]:
    rows = conn().execute("SELECT id, vector, bpm, camelot FROM tracks WHERE status='done' AND vector IS NOT NULL").fetchall()
    ids = [r["id"] for r in rows]
    if tid not in ids or len(ids) < 2:
        return []
    V = np.array([json.loads(r["vector"]) for r in rows])
    if len(V) >= 5:  # com poucas faixas a padronização distorce tudo
        V = (V - V.mean(0)) / (V.std(0) + 1e-9)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    i = ids.index(tid)
    sims = V @ V[i]
    me = rows[i]
    compat = set(camelot_neighbors(me["camelot"])) if me["camelot"] else set()
    out = []
    for j in np.argsort(-sims):
        if j == i:
            continue
        r = rows[j]
        out.append({"id": r["id"], "similarity": round(float(sims[j]), 3),
                    "key_compatible": r["camelot"] in compat,
                    "bpm_diff": round(abs((r["bpm"] or 0) - (me["bpm"] or 0)), 1)})
        if len(out) >= n:
            break
    info = {r["id"]: dict(r) for r in conn().execute(
        f"SELECT {LIST_COLS} FROM tracks WHERE id IN ({','.join(str(o['id']) for o in out)})")}
    for o in out:
        o.update({k: info[o["id"]][k] for k in ("title", "artist", "bpm", "key", "camelot")})
    return out


# --------------------------------------------------------------------------- #
# Letra editada pelo usuário
# --------------------------------------------------------------------------- #
def _retime(text: str, start: float, end: float, old_words: list | None) -> list[dict]:
    toks = text.split()
    if not toks:
        return []
    if old_words and len(old_words) == len(toks):          # só correção de grafia: mantém os tempos
        return [{"w": t, "s": w["s"], "e": w["e"]} for t, w in zip(toks, old_words)]
    weights = np.array([len(t) + 1 for t in toks], float)   # senão distribui pelo tamanho das palavras
    edges = start + (end - start) * np.concatenate([[0], np.cumsum(weights) / weights.sum()])
    return [{"w": t, "s": round(float(a), 2), "e": round(float(b), 2)} for t, a, b in zip(toks, edges[:-1], edges[1:])]


def apply_lyrics(a: dict, new_lines: list[dict]) -> dict:
    """new_lines: [{i (índice original ou None), start, end, text}]. Atualiza letra e seções."""
    lyr = a.get("lyrics") or {"language": None, "lines": [], "model": "manual"}
    old = lyr.get("lines", [])
    lyr.setdefault("original_lines", old)
    out = []
    for nl in new_lines:
        text = (nl.get("text") or "").strip()
        if not text:
            continue                                         # linha apagada
        i = nl.get("i")
        src = old[i] if isinstance(i, int) and 0 <= i < len(old) else {}
        start, end = float(nl.get("start", src.get("start", 0))), float(nl.get("end", src.get("end", 0)))
        if end <= start:
            end = start + 1.0
        same = src and src.get("text") == text and src.get("start") == start and src.get("end") == end
        words = src.get("words") if same else _retime(text, start, end, src.get("words"))
        line = {"start": round(start, 2), "end": round(end, 2), "text": text, "words": words or []}
        if src.get("part"):
            line["part"] = src["part"]
        out.append(line)
    out.sort(key=lambda l: l["start"])
    lyr.update(lines=out, edited=True, word_level=True)
    a["lyrics"] = lyr
    sync_sections(a)
    return a


def sync_sections(a: dict):
    """Põe a letra em cada seção e, se a letra colada tinha cabeçalhos ([Chorus]…), usa-os como nome da seção."""
    lines = (a.get("lyrics") or {}).get("lines", [])
    for s in a.get("sections", []):
        inside = [l for l in lines if s["start"] <= (l["start"] + l["end"]) / 2 < s["end"]]
        s["lyrics"] = [l["text"] for l in inside]
        votes: dict[str, float] = {}
        for l in inside:
            if l.get("part"):
                votes[l["part"]] = votes.get(l["part"], 0) + l["end"] - l["start"]
        if votes:
            s.setdefault("name_auto", s["name"])
            s["name"] = max(votes, key=votes.get)
        elif any(l.get("part") for l in lines):     # letra com partes, mas aqui ninguém canta
            s.setdefault("name_auto", s["name"])
            s["name"] = ("Intro" if s["end"] <= lines[0]["start"] + 0.5 else
                         "Final instrumental" if s["start"] >= lines[-1]["end"] - 0.5 else "Instrumental")
        elif "name_auto" in s:
            s["name"] = s.pop("name_auto")


def paste_lyrics(tid: int, text: str) -> dict:
    """Substitui a letra pela versão colada, mantendo a sincronia que o Whisper ouviu."""
    from .lyrics_align import align, parse_pasted
    t = get_track(tid)
    if not t or not t["analysis"]:
        raise KeyError(tid)
    a = t["analysis"]
    pasted = parse_pasted(text)
    if not pasted:
        raise ValueError("Nenhuma linha de letra encontrada no texto colado.")
    lyr = a.get("lyrics") or {"language": None, "lines": [], "model": "manual"}
    ref_lines = lyr.get("original_lines", lyr.get("lines", []))
    ref_words = [w for l in ref_lines for w in (l.get("words") or [])]
    lines = align(pasted, ref_words, a.get("vocal_regions") or [], a.get("duration") or 0)
    lyr.setdefault("original_lines", lyr.get("lines", []))
    lyr.update(lines=lines, edited=True, word_level=True, source="colada")
    a["lyrics"] = lyr
    sync_sections(a)
    conn().execute("UPDATE tracks SET analysis=? WHERE id=?", (json.dumps(a), tid))
    conn().commit()
    return {"lyrics": lyr, "sections": a["sections"]}


def save_lyrics(tid: int, new_lines: list[dict] | None, restore: bool = False) -> dict:
    t = get_track(tid)
    if not t or not t["analysis"]:
        raise KeyError(tid)
    a = t["analysis"]
    if restore:
        orig = (a.get("lyrics") or {}).get("original_lines")
        if orig is not None:
            a["lyrics"]["lines"] = orig
            for k in ("original_lines", "edited", "source"):
                a["lyrics"].pop(k, None)
            sync_sections(a)
    else:
        a = apply_lyrics(a, new_lines or [])
    conn().execute("UPDATE tracks SET analysis=? WHERE id=?", (json.dumps(a), tid))
    conn().commit()
    return a["lyrics"]


def set_pending(tid: int):
    conn().execute("UPDATE tracks SET status='pending', error=NULL WHERE id=?", (tid,))
    conn().commit()
    worker.wake()


def delete(tid: int):
    conn().execute("DELETE FROM tracks WHERE id=?", (tid,))
    conn().commit()


# --------------------------------------------------------------------------- #
# Fila de análise
# --------------------------------------------------------------------------- #
class Worker:
    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self.current: dict[int, dict] = {}
        self._ev = threading.Event()
        self._claim = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="analise")

    def start(self):
        for _ in range(self.n_workers):
            self._pool.submit(self._loop)

    def wake(self):
        self._ev.set()

    def _next(self):
        with self._claim:
            c = conn()
            r = c.execute("SELECT id, path FROM tracks WHERE status='pending' ORDER BY id LIMIT 1").fetchone()
            if r:
                c.execute("UPDATE tracks SET status='processing' WHERE id=?", (r["id"],))
                c.commit()
            return r

    def _loop(self):
        while True:
            r = self._next()
            if not r:
                self._ev.wait(5)
                self._ev.clear()
                continue
            tid = r["id"]
            self.current[tid] = {"id": tid, "file": os.path.basename(r["path"]), "since": time.time()}
            try:
                a = analyzer.analyze(r["path"])
                # letra corrigida à mão sobrevive à reanálise
                prev = conn().execute("SELECT analysis FROM tracks WHERE id=?", (tid,)).fetchone()
                prev_lyr = (json.loads(prev["analysis"]) or {}).get("lyrics") if prev and prev["analysis"] else None
                if prev_lyr and prev_lyr.get("edited"):
                    a["lyrics"] = prev_lyr                     # mantém os tempos por palavra já corrigidos
                    sync_sections(a)
                m = a["meta"]
                conn().execute(
                    """UPDATE tracks SET status='done', error=NULL, title=?, artist=?, album=?, year=?,
                       genre_tag=?, duration=?, bpm=?, key=?, camelot=?, mode=?, key_confidence=?,
                       lufs=?, lra=?, energy=?, brightness=?, sections=?, analysis=?, vector=?,
                       analyzed_at=? WHERE id=?""",
                    (m["title"], m["artist"], m["album"], m["year"], m["genre_tag"], a["duration"],
                     a["tempo"]["bpm"], a["key"]["key_short"], a["key"]["camelot"], a["key"]["mode"],
                     a["key"]["confidence"], a["loudness"]["lufs"], a["loudness"]["lra"], a["energy_score"],
                     a["timbre"]["centroid_hz"], len(a["sections"]),
                     json.dumps({k: v for k, v in a.items() if k != "vector"}), json.dumps(a["vector"]),
                     time.time(), tid))
                log.info("ok %s (%.1fs)", r["path"], a["analysis_seconds"])
            except Exception as e:
                log.exception("erro em %s", r["path"])
                conn().execute("UPDATE tracks SET status='error', error=? WHERE id=?", (str(e)[:500], tid))
            conn().commit()
            self.current.pop(tid, None)
            gc.collect()


# A GPU é serializada por um lock; 2 análises em paralelo já mantêm CPU e GPU ocupadas.
worker = Worker(int(os.environ.get("MUSICA_WORKERS", 2)))
