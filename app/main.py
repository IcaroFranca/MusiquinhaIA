"""API FastAPI + interface web do Analisador Musical."""
from __future__ import annotations

import logging
import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from urllib.parse import quote

from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analyzer, generator, library
from .prompt import BRIEF_OPTIONS, DEFAULT_BRIEF, GOALS, build_prompt, build_timing

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
UPLOADS = os.path.join(library.DATA_DIR, "uploads")
os.makedirs(UPLOADS, exist_ok=True)


@asynccontextmanager
async def lifespan(_app):
    library.init()
    library.worker.start()
    generator.load_saved(EXPORTS)
    yield


app = FastAPI(title="Analisador Musical", lifespan=lifespan)


@app.middleware("http")
async def no_cache_ui(request, call_next):
    """A interface muda a cada atualização do app: o navegador sempre revalida (ETag), sem usar cópia velha."""
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


class ScanReq(BaseModel):
    folder: str


@app.get("/api/status")
def status():
    return library.stats()


@app.post("/api/scan")
def scan(req: ScanReq):
    try:
        return library.scan_folder(req.folder.strip().strip('"'))
    except FileNotFoundError:
        raise HTTPException(404, "Pasta não encontrada")


@app.post("/api/pick-folder")
def pick_folder():
    """Abre o seletor de pastas nativo do Windows (o servidor roda na sua máquina)."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Escolha a pasta de músicas")
    root.destroy()
    return {"folder": os.path.normpath(folder) if folder else None}


@app.post("/api/upload")
async def upload(files: list[UploadFile]):
    n = 0
    for f in files:
        name = os.path.basename(f.filename or "")
        if os.path.splitext(name)[1].lower() not in analyzer.AUDIO_EXTS:
            continue
        dest = os.path.join(UPLOADS, name)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        n += library.add_file(dest)
    library.worker.wake()
    return {"queued": n}


@app.get("/api/tracks")
def tracks(q: str | None = None, bpm_min: float | None = None, bpm_max: float | None = None,
           camelot: str | None = None, status: str | None = None, sort: str = "added",
           desc: bool = True, half_double: bool = False, limit: int = 500, offset: int = 0):
    return library.list_tracks(q, bpm_min, bpm_max, camelot, status, sort, desc, half_double, limit, offset)


def _track(tid: int) -> dict:
    t = library.get_track(tid)
    if not t:
        raise HTTPException(404, "Faixa não encontrada")
    return t


@app.get("/api/tracks/{tid}")
def track(tid: int):
    t = _track(tid)
    if t.get("analysis"):
        t["analysis"].pop("envelopes", None)   # ~150 KB que a interface não usa (vão no timing.json)
    return t


@app.get("/api/tracks/{tid}/audio")
def audio(tid: int):
    t = _track(tid)
    if not os.path.exists(t["path"]):
        raise HTTPException(410, "Arquivo não existe mais no disco")
    return FileResponse(t["path"])


@app.get("/api/goals")
def goals():
    return {k: v["label"] for k, v in GOALS.items()}


@app.get("/api/brief-options")
def brief_options():
    return {"tipo": BRIEF_OPTIONS["tipo"], "tecnica": BRIEF_OPTIONS["tecnica"],
            "formato": list(BRIEF_OPTIONS["formato"]), "fps": [24, 30, 60], "default": DEFAULT_BRIEF}


class PromptReq(BaseModel):
    goal: str = "video"
    extra: str | None = None
    lyrics: bool = True
    brief: dict = {}


def _analyzed(tid: int) -> dict:
    t = _track(tid)
    if not t["analysis"]:
        raise HTTPException(409, "Faixa ainda não analisada")
    return t


@app.post("/api/tracks/{tid}/prompt", response_class=PlainTextResponse)
def prompt(tid: int, req: PromptReq):
    t = _analyzed(tid)
    return build_prompt(t["analysis"], req.goal, req.extra, req.lyrics, t["filename"], req.brief)


@app.get("/api/tracks/{tid}/prompt", response_class=PlainTextResponse)
def prompt_get(tid: int, goal: str = "video", extra: str | None = None, lyrics: bool = True):
    t = _analyzed(tid)
    return build_prompt(t["analysis"], goal, extra, lyrics, t["filename"])


@app.get("/api/tracks/{tid}/timing.json")
def timing(tid: int):
    t = _analyzed(tid)
    name = os.path.splitext(t["filename"])[0] + ".timing.json"
    return JSONResponse(build_timing(t["analysis"], t["filename"]),
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})


class LyricsReq(BaseModel):
    lines: list[dict] = []
    restore: bool = False


@app.put("/api/tracks/{tid}/lyrics")
def save_lyrics(tid: int, req: LyricsReq):
    _analyzed(tid)
    return library.save_lyrics(tid, req.lines, req.restore)


class PasteReq(BaseModel):
    text: str


@app.post("/api/tracks/{tid}/lyrics/paste")
def paste_lyrics(tid: int, req: PasteReq):
    _analyzed(tid)
    try:
        return library.paste_lyrics(tid, req.text)
    except ValueError as e:
        raise HTTPException(400, str(e))


EXPORTS = os.path.join(os.path.dirname(library.DATA_DIR), "exports")


@app.post("/api/tracks/{tid}/export")
def export(tid: int, req: PromptReq, open_folder: bool = True):
    """Cria uma pasta pronta para abrir no Claude: áudio + PROMPT.md + timing.json."""
    out = make_package(_analyzed(tid), req)
    if open_folder and os.name == "nt":
        os.startfile(out["folder"])
    return out


# ---- geração automática pelo Claude Code ---------------------------------- #
class GenerateReq(PromptReq):
    model: str = "opus"
    effort: str = "low"


@app.get("/api/claude/status")
def claude_status():
    return {**generator.claude_status(), "models": generator.MODELS, "efforts": generator.EFFORTS}


@app.post("/api/tracks/{tid}/generate")
def generate(tid: int, req: GenerateReq):
    st = generator.claude_status()
    if not st.get("loggedIn"):
        raise HTTPException(412, "Claude Code não está logado")
    pkg = make_package(_analyzed(tid), req)
    return generator.start(tid, pkg["folder"], req.model, req.effort).public()


def _job(jid: str) -> generator.Job:
    j = generator.JOBS.get(jid)
    if not j:
        raise HTTPException(404, "Geração não encontrada")
    return j


@app.get("/api/jobs/{jid}")
def job(jid: str, since: int = 0):
    return _job(jid).public(since)


@app.get("/api/tracks/{tid}/job")
def track_job(tid: int, since: int = 0):
    j = generator.latest_for(tid)
    return j.public(since) if j else None


@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str):
    generator.cancel(_job(jid))
    return {"ok": True}


@app.get("/api/jobs/{jid}/video")
def job_video(jid: str):
    j = _job(jid)
    if not j.video or not os.path.exists(j.video):
        raise HTTPException(404, "Sem vídeo")
    return FileResponse(j.video, media_type="video/mp4")


@app.post("/api/jobs/{jid}/open")
def job_open(jid: str):
    j = _job(jid)
    if os.name == "nt":
        os.startfile(os.path.dirname(j.video) if j.video else j.folder)
    return {"ok": True}


def make_package(t: dict, req: PromptReq) -> dict:
    import json
    import re
    tid = t["id"]
    a = t["analysis"]
    name = re.sub(r'[<>:"/\\|?*]+', "", f"{a['meta'].get('artist') or ''} - {a['meta'].get('title') or t['filename']}")
    name = re.sub(r"\s+", " ", name).strip(" -.") or f"faixa-{tid}"
    folder = os.path.join(EXPORTS, name[:80])
    os.makedirs(folder, exist_ok=True)
    audio_name = t["filename"]
    dest = os.path.join(folder, audio_name)
    if not os.path.exists(dest):
        shutil.copy2(t["path"], dest)
    text = build_prompt(a, "video", req.extra, req.lyrics, audio_name, req.brief, files=True)
    with open(os.path.join(folder, "PROMPT.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(folder, "timing.json"), "w", encoding="utf-8") as f:
        json.dump(build_timing(a, audio_name), f, ensure_ascii=False, separators=(",", ":"))
    return {"folder": folder, "files": sorted(os.listdir(folder))}


@app.get("/api/tracks/{tid}/similar")
def similar(tid: int, n: int = 12):
    return library.similar(tid, n)


@app.post("/api/tracks/{tid}/reanalyze")
def reanalyze(tid: int):
    _track(tid)
    library.set_pending(tid)
    return {"ok": True}


@app.post("/api/retry-errors")
def retry_errors():
    c = library.conn()
    n = c.execute("UPDATE tracks SET status='pending', error=NULL WHERE status='error'").rowcount
    c.commit()
    library.worker.wake()
    return {"queued": n}


@app.delete("/api/tracks/{tid}")
def remove(tid: int):
    library.delete(tid)
    return {"ok": True}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(os.path.join(STATIC, "favicon.svg"), media_type="image/svg+xml")


@app.post("/api/reanalyze-all")
def reanalyze_all(outdated_only: bool = True):
    """Reenfileira faixas analisadas por versões antigas do analisador."""
    import json
    c = library.conn()
    n = 0
    for r in c.execute("SELECT id, analysis FROM tracks WHERE status='done'").fetchall():
        ver = (json.loads(r["analysis"]) or {}).get("version", 1) if r["analysis"] else 0
        if not outdated_only or ver < analyzer.ANALYSIS_VERSION:
            c.execute("UPDATE tracks SET status='pending' WHERE id=?", (r["id"],))
            n += 1
    c.commit()
    library.worker.wake()
    return {"queued": n}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")
