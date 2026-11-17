"""Serveur local du Clipper : API + interface web + planificateur de publication.

Lancement : double-clic sur lancer.command (le serveur tourne en arrière-plan, tu peux fermer
le terminal et le navigateur). Arrêt : arreter.command.
Interface : http://127.0.0.1:7860
"""
from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import threading
import time
import traceback
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from clipper import pipeline as P
from clipper import publish as PUB
from clipper import social as SOC

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
UPLOADS = P.WORK / "uploads"
for d in (P.WORK, P.OUT, UPLOADS):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Clipper")
worker = ThreadPoolExecutor(max_workers=1)   # une génération à la fois (8 Go de RAM) ; les autres attendent
LOCK = threading.Lock()
JOBS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Confort : notifications macOS et anti-veille
# ---------------------------------------------------------------------------

def notify(title: str, message: str = "") -> None:
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    try:
        subprocess.Popen(["osascript", "-e", f'display notification "{esc(message)}" with title "Clipper" '
                                             f'subtitle "{esc(title)}" sound name "Glass"'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


@contextlib.contextmanager
def keep_awake():
    """Empêche la mise en veille du Mac pendant une génération ou un envoi."""
    proc = None
    try:
        proc = subprocess.Popen(["caffeinate", "-i", "-s"])
    except Exception:
        pass
    try:
        yield
    finally:
        if proc:
            proc.terminate()


# ---------------------------------------------------------------------------
# Persistance : chaque projet a un job.json dans son dossier de clips
# ---------------------------------------------------------------------------

def _save(job: dict) -> None:
    if job.get("out_dir"):
        data = {k: v for k, v in job.items() if not k.startswith("_")}
        tmp = Path(job["out_dir"], "job.json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        tmp.replace(Path(job["out_dir"], "job.json"))


def _file_url(path: str | None, version: float = 0) -> str | None:
    if not path:
        return None
    rel = Path(path).resolve().relative_to(P.OUT.resolve())
    return f"/files/{rel.as_posix()}?v={int(version)}"


def _import_legacy(folder: Path) -> None:
    """Projets créés avec la première version (Gradio) : on les reprend dans « Mes projets »."""
    import re
    data = json.loads((folder / "clips.json").read_text())
    src = data.get("source", "")
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", src)
    work = P.WORK / m.group(1) if m else None
    transcripts = sorted(work.glob("transcription_*.json"), key=lambda p: "small" not in p.name) if work and work.exists() else []
    videos = [v for v in work.glob("source*") if v.suffix in (".mp4", ".mkv", ".webm")] if work and work.exists() else []
    created = (folder / "clips.json").stat().st_mtime
    job = {"id": uuid.uuid4().hex[:10], "created": created, "finished": created, "status": "done", "step": "done",
           "progress": 1.0, "message": "Terminé", "title": data.get("titre") or folder.name,
           "source": {"type": "url", "url": src}, "language": data.get("langue"),
           "options": {"render": {"aspect": "9:16", "framing": "face", "captions": True, "title_mode": "none",
                                  "progress_bar": False, "quality": "high", "style": {}}},
           "out_dir": str(folder), "video": str(videos[0]) if videos else None,
           "transcript": str(transcripts[0]) if transcripts else None, "clips": []}
    full = ""
    if job["transcript"]:
        full = P.join_words([P.Word(**w) for w in json.loads(Path(job["transcript"]).read_text())["words"]])
    for i, c in enumerate(data.get("clips", [])):
        path = Path(c.get("path") or "")
        if not path.exists():
            continue
        thumb = path.with_suffix(".jpg")
        if not thumb.exists():
            P.thumbnail(path, 1.5)
        v = path.stat().st_mtime
        job["clips"].append({**c, "id": i, "status": "done", "render": {}, "url": _file_url(str(path), v),
                             "thumb": _file_url(str(thumb), v),
                             "social": SOC.generate(c.get("title", ""), c.get("text", ""), full, data.get("langue"), seed=i),
                             "download": f"/api/download?path={path.relative_to(P.OUT).as_posix()}"})
    JOBS[job["id"]] = job
    _save(job)


def _load_history() -> None:
    for d in P.OUT.iterdir():
        if d.is_dir() and (d / "clips.json").exists() and not (d / "job.json").exists():
            try:
                _import_legacy(d)
            except Exception:
                traceback.print_exc()
    for f in P.OUT.glob("*/job.json"):
        try:
            job = json.loads(f.read_text())
            if job.get("status") in ("queued", "running"):
                job["status"], job["message"] = "error", "Interrompu (le serveur a été arrêté)"
                for c in job.get("clips", []):
                    if c.get("status") in ("pending", "rendering"):
                        c["status"] = "error"
            for i, c in enumerate(job.get("clips", [])):  # anciens projets sans textes par réseau
                if c.get("status") == "done" and not c.get("social"):
                    c["social"] = SOC.generate(c.get("title", ""), c.get("text", ""), "", job.get("language"), seed=i)
            JOBS[job["id"]] = job
        except Exception:
            traceback.print_exc()


def _public(job: dict) -> dict:
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _update(obj: dict, **kw) -> None:
    with LOCK:
        obj.update(kw)


# ---------------------------------------------------------------------------
# Génération en arrière-plan
# ---------------------------------------------------------------------------

def _run_job(job: dict) -> None:
    o = job["options"]
    with keep_awake():
        try:
            def prog(step: str, lo: float, hi: float):
                def cb(p: float, msg: str):
                    _update(job, step=step, progress=round(lo + (hi - lo) * max(0, min(p, 1)), 4), message=msg)
                return cb

            _update(job, status="running", step="download", message="Récupération de la vidéo", started=time.time())
            if job["source"]["type"] == "file":
                video, title = P.import_local(job["source"]["path"])
            else:
                video, title = P.download(job["source"]["url"], prog("download", 0, 0.12),
                                          max_height=int(o.get("source_height", 1440)))
            job["title"] = job.get("title") or title
            job["video"] = str(video)

            lang = o.get("language") or None
            model, vocab = o.get("whisper_model", "small"), o.get("vocabulary", "")
            words, duration, detected = P.transcribe(video, model, lang, prog("transcribe", 0.12, 0.5), vocab)
            if not words:
                raise RuntimeError("Aucune parole détectée dans la vidéo.")
            job["transcript"] = str(P.transcript_cache(video, model, lang, vocab))
            job["duration"], job["language"] = duration, detected

            clips = P.find_highlights(words, int(o.get("n_clips", 5)), float(o.get("dmin", 25)),
                                      float(o.get("dmax", 60)), o.get("llm_model") or None,
                                      prog("analyze", 0.5, 0.56))
            if not clips:
                raise RuntimeError("Aucun passage exploitable. Essaie une durée minimale plus courte.")

            _update(job, message="Écriture des descriptions et hashtags")
            full = P.join_words(words)
            out_dir = P.OUT / f"{P.slug(job['title'])}-{job['id'][:6]}"
            out_dir.mkdir(parents=True, exist_ok=True)
            job["out_dir"] = str(out_dir)
            job["clips"] = []
            for i, c in enumerate(clips):
                c.text = P.join_words([w for w in words if c.start <= w.start < c.end])
                job["clips"].append({**asdict(c), "id": i, "status": "pending", "url": None, "thumb": None,
                                     "render": o.get("render", {}),
                                     "social": SOC.generate(c.title, c.text, full, detected, o.get("llm_model"), seed=i)})
                prog("analyze", 0.56, 0.6)((i + 1) / len(clips), "Écriture des descriptions et hashtags")
            _save(job)

            opts = P.RenderOptions.from_dict(o.get("render"))
            for i, c in enumerate(clips):
                lo, hi = 0.6 + 0.4 * i / len(clips), 0.6 + 0.4 * (i + 1) / len(clips)
                cb = prog("render", lo, hi)
                _render_one(job, i, c, words, opts, lambda p, m, i=i: cb(p, f"{m} {i + 1}/{len(clips)}"))

            _update(job, status="done", step="done", progress=1.0, message="Terminé", finished=time.time())
            notify("Clips prêts", f"{len(clips)} clips · {job['title'][:60]}")
            _auto_queue(job)
        except Exception as e:
            traceback.print_exc()
            _update(job, status="error", message=str(e) or e.__class__.__name__)
            for c in job.get("clips", []):
                if c.get("status") in ("pending", "rendering"):
                    c["status"] = "error"
            notify("Échec de la génération", str(e)[:120])
        finally:
            _save(job)


def _render_one(job: dict, i: int, clip: P.Clip, words: list[P.Word], opts: P.RenderOptions, cb) -> None:
    entry = job["clips"][i]
    _update(entry, status="rendering")
    path = Path(job["out_dir"]) / f"{i + 1:02d}_{P.slug(clip.title)}.mp4"
    old = entry.get("path")
    P.render_clip(Path(job["video"]), clip, words, path, opts, cb)
    if old and old != str(path):
        Path(old).unlink(missing_ok=True)
        Path(old).with_suffix(".jpg").unlink(missing_ok=True)
        Path(old).with_suffix(".txt").unlink(missing_ok=True)
    thumb = P.thumbnail(path, min(1.5, clip.duration / 2))
    social = entry.get("social") or {}
    lines = [clip.title, ""]
    for net in SOC.NETWORKS:
        if net in social:
            lines += [f"--- {net.upper()} ---", SOC.caption(social, net), ""]
    lines += [f"Extrait {P.fmt_time(clip.start)} → {P.fmt_time(clip.end)}", "", "Transcription :", clip.text]
    path.with_suffix(".txt").write_text("\n".join(lines))
    v = time.time()
    data = asdict(clip)
    data.update(path=str(path), status="done", error=None, url=_file_url(str(path), v),
                thumb=_file_url(str(thumb), v),
                download=f"/api/download?path={path.relative_to(P.OUT).as_posix()}")
    _update(entry, **data)
    _save(job)
    # le clip est peut-être déjà dans la file de publication : on met à jour le fichier
    for item in PUB.get_queue():
        if item["job_id"] == job["id"] and item["clip_id"] == i:
            PUB.update_item(item["id"], path=str(path), thumb=entry["thumb"], title=clip.title)


def _words_for(job: dict) -> list[P.Word]:
    data = json.loads(Path(job["transcript"]).read_text())
    return [P.Word(**w) for w in data["words"]]


def _rerender(job: dict, i: int, changes: dict) -> None:
    entry = job["clips"][i]
    with keep_awake():
        try:
            words = _words_for(job)
            start = float(changes.get("start", entry["start"]))
            end = float(changes.get("end", entry["end"]))
            if end - start < 3:
                raise ValueError("Le clip doit durer au moins 3 secondes.")
            text = P.join_words([w for w in words if start <= w.start < end])
            title = changes.get("title", entry["title"])
            clip = P.Clip(start=start, end=end, title=title, hook=entry.get("hook", ""),
                          hashtags=entry.get("hashtags", []), score=entry["score"], text=text)
            if "social" in changes:
                entry["social"] = changes["social"]
            elif title != entry["title"] or text != entry.get("text"):
                entry["social"] = SOC.generate(title, text, P.join_words(words), job.get("language"),
                                               job["options"].get("llm_model"), seed=i)
            render = changes.get("render") or entry.get("render") or job["options"].get("render", {})
            entry["render"] = render
            _render_one(job, i, clip, words, P.RenderOptions.from_dict(render),
                        lambda p, m: _update(entry, progress=round(p, 3)))
        except Exception as e:
            traceback.print_exc()
            _update(entry, status="error", error=str(e))
            _save(job)


def _queue_items(job: dict, clip_ids: list[int], networks: list[str]) -> list[dict]:
    items = []
    for i in clip_ids:
        c = job["clips"][i]
        if c.get("status") != "done" or not c.get("path"):
            continue
        items.append({"job_id": job["id"], "clip_id": i, "path": c["path"], "thumb": c.get("thumb"),
                      "title": c["title"], "copy": c.get("social", {}), "networks": networks})
    return items


def _auto_queue(job: dict) -> None:
    s = PUB.get_schedule()
    if not s.get("auto_queue"):
        return
    ids = [c["id"] for c in job["clips"] if c.get("status") == "done" and c.get("score", 0) >= s.get("auto_queue_min_score", 0)]
    nets = [n for n in s.get("auto_queue_networks", []) if n in PUB.PROVIDERS]
    if ids and nets:
        PUB.enqueue(_queue_items(job, ids, nets))


# ---------------------------------------------------------------------------
# API : projets et clips
# ---------------------------------------------------------------------------

class RenderIn(BaseModel):
    aspect: str = "9:16"
    framing: str = "face"
    captions: bool = True
    title_mode: str = "intro"
    progress_bar: bool = False
    quality: str = "high"
    style: dict = Field(default_factory=dict)


class JobIn(BaseModel):
    url: str | None = None
    upload_id: str | None = None
    n_clips: int = Field(5, ge=1, le=30)
    dmin: float = Field(25, ge=5, le=170)
    dmax: float = Field(60, ge=10, le=180)
    whisper_model: str = "small"
    language: str | None = None
    vocabulary: str = ""
    source_height: int = 1440
    llm_model: str | None = None
    render: RenderIn = Field(default_factory=RenderIn)


class ClipEditIn(BaseModel):
    start: float | None = None
    end: float | None = None
    title: str | None = None
    social: dict | None = None
    render: RenderIn | None = None


@app.get("/api/config")
def config():
    return {"fonts": P.available_fonts(), "llm_models": P.ollama_models(),
            "whisper_models": list(P.WHISPER_MODELS), "disk_free_gb": round(shutil.disk_usage(P.ROOT).free / 1e9, 1)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if ext not in {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}:
        raise HTTPException(400, "Format non supporté (mp4, mov, mkv, webm, m4v, avi).")
    uid = uuid.uuid4().hex[:12]
    dst = UPLOADS / f"{uid}{ext}"
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f, 1024 * 1024)
    return {"upload_id": uid, "name": file.filename}


@app.post("/api/jobs")
def create_job(body: JobIn):
    if body.dmin >= body.dmax:
        raise HTTPException(400, "La durée minimale doit être inférieure à la durée maximale.")
    if body.upload_id:
        files = list(UPLOADS.glob(f"{body.upload_id}.*"))
        if not files:
            raise HTTPException(404, "Fichier envoyé introuvable.")
        source = {"type": "file", "path": str(files[0])}
    elif body.url and body.url.strip().startswith(("http://", "https://")):
        source = {"type": "url", "url": body.url.strip()}
    else:
        raise HTTPException(400, "Colle un lien valide ou dépose une vidéo.")
    waiting = sum(1 for j in JOBS.values() if j["status"] in ("queued", "running"))
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "created": time.time(), "status": "queued", "step": "queued", "progress": 0.0,
           "message": "En attente" + (f" ({waiting} vidéo(s) avant)" if waiting else ""), "source": source,
           "options": body.model_dump(), "clips": [], "title": None}
    JOBS[jid] = job
    worker.submit(_run_job, job)
    return _public(job)


@app.get("/api/jobs")
def list_jobs():
    jobs = sorted(JOBS.values(), key=lambda j: -j.get("created", 0))
    return [{k: j.get(k) for k in ("id", "title", "status", "created", "message", "progress")}
            | {"thumb": next((c.get("thumb") for c in j.get("clips", []) if c.get("thumb")), None),
               "count": len(j.get("clips", [])),
               "source": j.get("source", {}).get("url") or "Vidéo importée"} for j in jobs]


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "Projet introuvable.")
    queued = {(i["job_id"], i["clip_id"]): {n: t["status"] for n, t in i["targets"].items()} for i in PUB.get_queue()}
    out = _public(job)
    out["clips"] = [{**c, "publish": queued.get((jid, c["id"]), {})} for c in job.get("clips", [])]
    return out


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "Projet introuvable.")
    if job["status"] in ("queued", "running"):
        raise HTTPException(409, "Ce projet est en cours de traitement.")
    if job.get("out_dir") and Path(job["out_dir"]).resolve().parent == P.OUT.resolve():
        shutil.rmtree(job["out_dir"], ignore_errors=True)
    for item in PUB.get_queue():
        if item["job_id"] == jid:
            PUB.remove(item["id"])
    JOBS.pop(jid, None)
    return {"ok": True}


@app.post("/api/jobs/{jid}/clips/{i}")
def edit_clip(jid: str, i: int, body: ClipEditIn):
    job = JOBS.get(jid)
    if not job or i < 0 or i >= len(job.get("clips", [])):
        raise HTTPException(404, "Clip introuvable.")
    if not job.get("transcript") or not Path(job["transcript"]).exists() or not job.get("video") \
            or not Path(job["video"]).exists():
        raise HTTPException(409, "La vidéo source de ce projet n'est plus disponible, impossible de le regénérer.")
    entry = job["clips"][i]
    if entry.get("status") == "rendering":
        raise HTTPException(409, "Ce clip est déjà en cours de rendu.")
    changes = body.model_dump(exclude_none=True)
    _update(entry, status="rendering", progress=0, error=None)
    worker.submit(_rerender, job, i, changes)
    return entry


@app.post("/api/jobs/{jid}/clips/{i}/social")
def save_social(jid: str, i: int, body: dict):
    """Enregistre les textes modifiés à la main (sans regénérer la vidéo)."""
    job = JOBS.get(jid)
    if not job or i < 0 or i >= len(job.get("clips", [])):
        raise HTTPException(404, "Clip introuvable.")
    entry = job["clips"][i]
    entry["social"] = {n: body[n] for n in SOC.NETWORKS if isinstance(body.get(n), dict)} or entry.get("social", {})
    _save(job)
    for item in PUB.get_queue():
        if item["job_id"] == jid and item["clip_id"] == i:
            PUB.update_item(item["id"], copy=entry["social"])
    return entry["social"]


@app.post("/api/jobs/{jid}/clips/{i}/social/regenerate")
def regenerate_social(jid: str, i: int):
    job = JOBS.get(jid)
    if not job or i < 0 or i >= len(job.get("clips", [])):
        raise HTTPException(404, "Clip introuvable.")
    c = job["clips"][i]
    full = P.join_words(_words_for(job)) if job.get("transcript") and Path(job["transcript"]).exists() else ""
    c["social"] = SOC.generate(c["title"], c.get("text", ""), full, job.get("language"),
                               job["options"].get("llm_model"), seed=int(time.time()) % 7)
    _save(job)
    return c["social"]


@app.get("/api/download")
def download(path: str):
    f = (P.OUT / path).resolve()
    if P.OUT.resolve() not in f.parents or not f.exists():
        raise HTTPException(404)
    return FileResponse(f, filename=f.name, media_type="video/mp4")


# ---------------------------------------------------------------------------
# API : comptes, file de publication, planning
# ---------------------------------------------------------------------------

class CredsIn(BaseModel):
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str | None = None


class QueueIn(BaseModel):
    job_id: str
    clip_ids: list[int]
    networks: list[str]


def _provider(p: str) -> str:
    if p not in PUB.PROVIDERS:
        raise HTTPException(404, "Réseau inconnu.")
    return p


@app.get("/api/automation")
def automation():
    return {"accounts": PUB.accounts_public(), "schedule": PUB.get_schedule(), "queue": PUB.get_queue(),
            "next": PUB.next_slots()}


@app.post("/api/automation/schedule")
def set_schedule(body: dict):
    return PUB.save_schedule(body)


@app.post("/api/accounts/{p}/credentials")
def set_credentials(p: str, body: CredsIn):
    PUB.save_credentials(_provider(p), body.client_id, body.client_secret, body.redirect_uri)
    return PUB.accounts_public()[p]


@app.get("/api/accounts/{p}/connect")
def connect(p: str):
    try:
        return RedirectResponse(PUB.auth_url(_provider(p)))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/accounts/{p}/disconnect")
def disconnect(p: str):
    PUB.disconnect(_provider(p))
    return PUB.accounts_public()[p]


@app.get("/oauth/{p}/callback")
def oauth_callback(p: str, code: str | None = None, state: str | None = None, error: str | None = None,
                   error_description: str | None = None):
    _provider(p)
    if error or not code:
        msg = error_description or error or "Connexion annulée."
        return RedirectResponse(f"/#/automation?err={_q(msg)}")
    try:
        PUB.finish_oauth(p, code, state or "")
    except Exception as e:
        traceback.print_exc()
        return RedirectResponse(f"/#/automation?err={_q(str(e))}")
    return RedirectResponse(f"/#/automation?ok={p}")


def _q(s: str) -> str:
    import urllib.parse
    return urllib.parse.quote(s[:200])


@app.post("/api/queue")
def add_to_queue(body: QueueIn):
    job = JOBS.get(body.job_id)
    if not job:
        raise HTTPException(404, "Projet introuvable.")
    nets = [n for n in body.networks if n in PUB.PROVIDERS]
    if not nets:
        raise HTTPException(400, "Choisis au moins un réseau.")
    items = _queue_items(job, body.clip_ids, nets)
    if not items:
        raise HTTPException(400, "Aucun clip prêt à publier.")
    PUB.enqueue(items)
    return {"queue": PUB.get_queue(), "added": len(items)}


@app.delete("/api/queue/{item_id}")
def queue_remove(item_id: str):
    PUB.remove(item_id)
    return PUB.get_queue()


@app.post("/api/queue/{item_id}/move")
def queue_move(item_id: str, body: dict):
    PUB.move(item_id, int(body.get("delta", 0)))
    return PUB.get_queue()


@app.post("/api/queue/{item_id}/publish/{network}")
def queue_publish_now(item_id: str, network: str):
    _provider(network)
    if not PUB.accounts_public()[network]["connected"]:
        raise HTTPException(409, f"Connecte d'abord ton compte {PUB.PROVIDERS[network]['label']}.")
    def run():
        with keep_awake():
            PUB.publish_item(item_id, network, notify)
    threading.Thread(target=run, daemon=True).start()
    return {"ok": True}


@app.post("/api/queue/{item_id}/retry/{network}")
def queue_retry(item_id: str, network: str):
    for i in PUB.get_queue():
        if i["id"] == item_id and network in i["targets"]:
            i["targets"][network] = {"status": "queued"}
            PUB.update_item(item_id, targets=i["targets"])
    return PUB.get_queue()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

app.mount("/files", StaticFiles(directory=P.OUT), name="files")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health():
    return {"ok": True, "jobs_running": sum(1 for j in JOBS.values() if j["status"] in ("queued", "running"))}


@app.exception_handler(Exception)
async def on_error(_req, exc):
    traceback.print_exc()
    return JSONResponse({"detail": str(exc)}, status_code=500)


if __name__ == "__main__":
    import sys
    _load_history()
    PUB.Scheduler(notify, keep_awake).start()
    port = PUB.PORT
    (HERE / "config" / "server.pid").write_text(str(__import__("os").getpid()))
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
