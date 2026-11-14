"""Geração automática de clipes: roda o Claude Code (modo não interativo) na pasta do pacote
exportado e acompanha o progresso pelo stream de eventos JSON.

O login é o do próprio Claude Code (assinatura do usuário), sem chave de API.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid

log = logging.getLogger("generator")


def claude_bin() -> str | None:
    cands = [os.environ.get("MUSICA_CLAUDE"), shutil.which("claude"),
             os.path.join(os.path.expanduser("~"), ".local", "bin", "claude.exe"),
             os.path.join(os.path.expanduser("~"), ".local", "bin", "claude")]
    return next((c for c in cands if c and os.path.exists(c)), None)


def claude_status() -> dict:
    b = claude_bin()
    if not b:
        return {"installed": False, "loggedIn": False}
    try:
        r = subprocess.run([b, "auth", "status"], capture_output=True, text=True, encoding="utf-8", timeout=30)
        info = json.loads(r.stdout or "{}")
    except Exception as e:
        return {"installed": True, "loggedIn": False, "bin": b, "error": str(e)}
    return {"installed": True, "bin": b, "loggedIn": bool(info.get("loggedIn")), "method": info.get("authMethod")}


MODELS = {"opus": "Opus (melhor qualidade)", "sonnet": "Sonnet (mais rápido)", "fable": "Fable (mais capaz, mais lento)"}
EFFORTS = ["low", "medium", "high", "xhigh"]

# O agente trabalha só dentro da pasta do pacote. Ferramentas de arquivo + os comandos que um
# projeto de vídeo precisa; qualquer outro comando é negado automaticamente.
# Só estas ferramentas existem na sessão (sem PowerShell, MCP de terceiros, agendamentos etc.).
TOOLS = "Bash,Read,Write,Edit,Glob,Grep,Skill"
ALLOWED_TOOLS = [
    "Read", "Write", "Edit", "Glob", "Grep", "Skill",
    "Bash(npx *)", "Bash(npm *)", "Bash(node *)", "Bash(python *)", "Bash(py *)",
    "Bash(ffmpeg *)", "Bash(ffprobe *)", "Bash(mkdir *)", "Bash(ls *)", "Bash(cp *)", "Bash(cd *)",
    "Bash(cat *)", "Bash(head *)", "Bash(tail *)", "Bash(wc *)", "Bash(echo *)",
    # só leitura, muito usados pelo agente para consultar arquivos e referências das skills
    "Bash(sed *)", "Bash(grep *)", "Bash(find *)", "Bash(sort *)", "Bash(uniq *)", "Bash(pwd)",
]

INSTRUCTION = (
    "Nesta pasta estão: o áudio da música, PROMPT.md (análise completa + briefing + tarefa) e timing.json "
    "(todos os tempos). Leia PROMPT.md e faça exatamente o que ele pede: crie o videoclipe completo, do "
    "conceito ao vídeo final. Use a skill /hyperframes para montar e renderizar, a menos que o briefing peça "
    "outra técnica. Trabalhe sozinho até o fim, sem pedir confirmação: não há ninguém para responder. "
    "Ao terminar, renderize o MP4 final com o áudio para a pasta renders/ dentro desta pasta, verifique a "
    "duração com ffprobe e escreva um resumo curto do conceito e das cenas em STORYBOARD.md."
)


class Job:
    def __init__(self, track_id: int, folder: str, model: str, effort: str):
        self.id = uuid.uuid4().hex[:10]
        self.track_id, self.folder, self.model, self.effort = track_id, folder, model, effort
        self.status = "queued"        # queued | running | done | error | cancelled
        self.log: list[dict] = []
        self.started = self.finished = None
        self.session_id = None
        self.cost_usd = None
        self.turns = None
        self.video = None
        self.error = None
        self.proc: subprocess.Popen | None = None

    def add(self, kind: str, text: str):
        self.log.append({"t": round(time.time() - (self.started or time.time()), 1), "kind": kind, "text": text[:600]})

    def public(self, since: int = 0) -> dict:
        return {"id": self.id, "track_id": self.track_id, "folder": self.folder, "status": self.status,
                "model": self.model, "effort": self.effort, "started": self.started, "finished": self.finished,
                "elapsed": round((self.finished or time.time()) - self.started) if self.started else 0,
                "cost_usd": self.cost_usd, "turns": self.turns, "session_id": self.session_id,
                "video": bool(self.video), "error": self.error,
                "log": self.log[since:], "log_total": len(self.log)}


JOBS: dict[str, Job] = {}
_lock = threading.Lock()   # um clipe por vez: renderização e modelos de GPU disputam a mesma máquina


def _describe_tool(name: str, inp: dict) -> str:
    if name == "Bash":
        return f"$ {inp.get('command', '')}"
    if name in ("Write", "Edit", "Read"):
        verb = {"Write": "criando", "Edit": "editando", "Read": "lendo"}[name]
        p = inp.get("file_path", "")
        return f"{verb} {os.path.basename(p) or p}"
    if name == "Skill":
        return f"usando a skill {inp.get('skill') or inp.get('name') or ''}"
    if name == "TodoWrite":
        todos = inp.get("todos") or []
        cur = next((t.get("content") for t in todos if t.get("status") == "in_progress"), None)
        return f"plano: {cur}" if cur else "atualizando o plano"
    return name


def _newest_mp4(folder: str) -> str | None:
    best = None
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in ("node_modules", ".hyperframes")]
        for f in files:
            if f.lower().endswith(".mp4"):
                p = os.path.join(root, f)
                if best is None or os.path.getmtime(p) > os.path.getmtime(best):
                    best = p
    return best


def _run(job: Job):
    with _lock:
        if job.status == "cancelled":
            return
        b = claude_bin()
        job.status, job.started = "running", time.time()
        job.add("info", f"Claude Code ({job.model}, effort {job.effort}) começou em {job.folder}")
        before = _newest_mp4(job.folder)
        before_mtime = os.path.getmtime(before) if before else 0
        cmd = [b, "-p", INSTRUCTION, "--model", job.model, "--effort", job.effort,
               "--output-format", "stream-json", "--verbose",
               "--permission-mode", "acceptEdits", "--permission-prompts", "none",
               "--tools", TOOLS, "--strict-mcp-config",
               "--allowedTools", *ALLOWED_TOOLS]
        try:
            job.proc = subprocess.Popen(
                cmd, cwd=job.folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            with open(os.path.join(job.folder, "claude_log.jsonl"), "a", encoding="utf-8") as raw:
                for line in job.proc.stdout:
                    raw.write(line)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        job.add("out", line)
                        continue
                    t = ev.get("type")
                    if t == "system" and ev.get("subtype") == "init":
                        job.session_id = ev.get("session_id")
                    elif t == "assistant":
                        for blk in (ev.get("message") or {}).get("content") or []:
                            if blk.get("type") == "text" and blk.get("text", "").strip():
                                job.add("text", blk["text"].strip())
                            elif blk.get("type") == "tool_use":
                                job.add("tool", _describe_tool(blk.get("name", ""), blk.get("input") or {}))
                    elif t == "user":
                        for blk in (ev.get("message") or {}).get("content") or []:
                            if isinstance(blk, dict) and blk.get("type") == "tool_result" and blk.get("is_error"):
                                c = blk.get("content")
                                msg = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
                                job.add("warn", msg[:300])
                    elif t == "result":
                        job.cost_usd = ev.get("total_cost_usd")
                        job.turns = ev.get("num_turns")
                        if ev.get("is_error") or ev.get("subtype") != "success":
                            job.error = ev.get("result") or ev.get("subtype")
                        elif ev.get("result") and not (job.log and job.log[-1]["text"] == ev["result"].strip()[:600]):
                            job.add("text", ev["result"].strip())
            job.proc.wait()
        except Exception as e:
            log.exception("geração falhou")
            job.error = str(e)
        job.finished = time.time()
        if job.status == "cancelled":
            job.add("info", "Cancelado.")
            return
        mp4 = _newest_mp4(job.folder)
        if mp4 and os.path.getmtime(mp4) > before_mtime:
            job.video = mp4
            job.status = "done"
            job.add("info", f"Vídeo pronto: {os.path.relpath(mp4, job.folder)}")
        else:
            job.status = "error"
            job.error = job.error or "O Claude terminou sem gerar um MP4 novo."
            job.add("warn", job.error)
        save(job)


# ---- persistência: cada geração deixa um job.json na pasta do pacote ------- #
def save(job: Job):
    try:
        data = {**job.public(), "video_path": job.video}
        with open(os.path.join(job.folder, "job.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except OSError as e:
        log.warning("não salvei job.json: %s", e)


def load_saved(exports_dir: str):
    """Recarrega gerações terminadas (sobrevivem a reinícios do servidor)."""
    if not os.path.isdir(exports_dir):
        return
    for name in os.listdir(exports_dir):
        p = os.path.join(exports_dir, name, "job.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            j = Job(d["track_id"], d["folder"], d["model"], d["effort"])
            j.id, j.status, j.log = d["id"], d["status"], d.get("log", [])
            j.started, j.finished = d.get("started"), d.get("finished")
            j.cost_usd, j.turns, j.session_id, j.error = d.get("cost_usd"), d.get("turns"), d.get("session_id"), d.get("error")
            j.video = d.get("video_path") if d.get("video_path") and os.path.exists(d["video_path"]) else None
            if j.status in ("queued", "running"):   # servidor caiu no meio
                j.status, j.error = "error", "Interrompido (o servidor foi reiniciado durante a geração)."
            JOBS[j.id] = j
        except Exception as e:
            log.warning("job.json inválido em %s: %s", name, e)


def start(track_id: int, folder: str, model: str = "opus", effort: str = "low") -> Job:
    if model not in MODELS:
        model = "opus"
    if effort not in EFFORTS:
        effort = "low"
    job = Job(track_id, folder, model, effort)
    JOBS[job.id] = job
    if _lock.locked():
        job.add("info", "Na fila: outro clipe está sendo gerado.")
    threading.Thread(target=_run, args=(job,), daemon=True, name=f"clipe-{job.id}").start()
    return job


def cancel(job: Job):
    job.status = "cancelled"
    if job.proc and job.proc.poll() is None:
        if os.name == "nt":   # mata a árvore toda (o agente pode ter aberto render/preview)
            subprocess.run(["taskkill", "/PID", str(job.proc.pid), "/T", "/F"], capture_output=True)
        else:
            job.proc.kill()


def latest_for(track_id: int) -> Job | None:
    js = [j for j in JOBS.values() if j.track_id == track_id]
    return max(js, key=lambda j: j.started or time.time()) if js else None
