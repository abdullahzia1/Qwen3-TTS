"""HTTP API (v1). The UI and every consuming app use these same endpoints."""

import json
import random
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from voicekit.text import chunk, normalize

from .db import Store
from .engine import Engine
from .worker import Worker

MAX_TEXT_CHARS = 60_000
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
PRIORITY_INTERACTIVE = 10


class TTSRequest(BaseModel):
    voice: str = Field(description="'name' (latest version) or 'name/vN'")
    text: str
    seed: int = 0
    params: Dict[str, Any] = Field(default_factory=dict)
    lexicon: Optional[Dict[str, str]] = None
    max_chars: int = Field(400, ge=80, le=1200)
    batch_size: int = Field(1, ge=1, le=16)
    target_lufs: float = Field(-16.0, ge=-30, le=-8)


class PreviewRequest(BaseModel):
    text: str
    lexicon: Optional[Dict[str, str]] = None
    max_chars: int = Field(400, ge=80, le=1200)


class RegenRequest(BaseModel):
    seed: Optional[int] = None


def _voice_versions(voices_dir: Path) -> List[Dict[str, Any]]:
    out = []
    if not voices_dir.exists():
        return out
    for vdir in sorted(p for p in voices_dir.iterdir() if p.is_dir()):
        versions = []
        for v in sorted(vdir.glob("v*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0):
            meta_path = v / "voice.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                v_info = {k: meta.get(k) for k in (
                    "version", "created_at", "mode", "language", "ref_text", "ref_duration_s",
                    "chars_per_sec", "f0_median_hz", "params", "notes",
                )}
                v_info["warnings"] = meta.get("warnings") or (
                    [f"Reference is {meta['ref_duration_s']:.0f}s; 8-15s gives steadier pitch and faster generation."]
                    if meta.get("ref_duration_s", 0) > 20 else []
                )
                versions.append(v_info)
        if versions:
            out.append({"name": vdir.name, "latest": versions[-1]["version"], "versions": versions})
    return out


def _resolve_voice(voices_dir: Path, voice: str) -> str:
    """Pin 'name' to 'name/vN' at submit time, so a later version can't change a queued job."""
    name, _, ver = voice.partition("/v")
    for v in _voice_versions(voices_dir):
        if v["name"] == name:
            if not ver:
                return f"{name}/v{v['latest']}"
            if any(str(x["version"]) == ver for x in v["versions"]):
                return voice
    raise HTTPException(404, f"voice '{voice}' not found")


def create_app(store: Store, engine: Engine, worker: Worker, data_dir: Path,
               api_key: Optional[str] = None, web_dist: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title="voicekit", version="0.1.0")
    voices_dir = data_dir / "voices"

    def auth(authorization: Optional[str] = Header(None)) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "invalid or missing API key")

    deps = [Depends(auth)]

    # --- engine (Start / Stop Model) ---------------------------------------
    @app.get("/v1/engine", dependencies=deps)
    def engine_status():
        return {**engine.status(), "queue_depth": store.queue_depth()}

    @app.post("/v1/engine/start", dependencies=deps)
    def engine_start():
        engine.start()
        return engine_status()

    @app.post("/v1/engine/stop", dependencies=deps)
    def engine_stop():
        engine.stop()
        return engine_status()

    # --- voices -------------------------------------------------------------
    @app.get("/v1/voices", dependencies=deps)
    def list_voices():
        return _voice_versions(voices_dir)

    @app.get("/v1/voices/{name}/v/{version}/ref.wav", dependencies=deps)
    def voice_ref(name: str, version: int):
        p = voices_dir / name / f"v{version}" / "ref.wav"
        if not NAME_RE.match(name) or not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type="audio/wav")

    @app.post("/v1/voices", status_code=202, dependencies=deps)
    def create_voice(
        name: str = Form(...),
        ref_text: str = Form(""),
        language: str = Form("English"),
        x_vector_only: bool = Form(False),
        notes: str = Form(""),
        file: UploadFile = File(...),
    ):
        if not NAME_RE.match(name):
            raise HTTPException(422, "name must be lowercase letters, digits, '-' or '_' (max 48)")
        if not x_vector_only and not ref_text.strip():
            raise HTTPException(422, "ref_text is required unless x_vector_only is set")
        uploads = data_dir / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or "ref.wav").suffix.lower() or ".wav"
        ref_path = uploads / f"{uuid.uuid4().hex}{suffix}"
        with ref_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        request = {"name": name, "ref_text": ref_text, "language": language,
                   "x_vector_only": x_vector_only, "notes": notes}
        return store.create_job("voice", request, "voice.create", {"ref_path": str(ref_path)},
                                priority=PRIORITY_INTERACTIVE)

    # --- TTS ------------------------------------------------------------------
    @app.post("/v1/tts/preview-chunks", dependencies=deps)
    def preview_chunks(req: PreviewRequest):
        norm = normalize(req.text, req.lexicon)
        return {"text_normalized": norm,
                "chunks": [{"text": c.text, "boundary": c.boundary, "chars": len(c.text)}
                           for c in chunk(norm, max_chars=req.max_chars)]}

    @app.post("/v1/tts/jobs", status_code=202, dependencies=deps)
    def create_tts_job(req: TTSRequest, idempotency_key: Optional[str] = Header(None)):
        if not any(ch.isalnum() for ch in req.text):
            raise HTTPException(422, "text has nothing speakable (no letters or digits)")
        if len(req.text) > MAX_TEXT_CHARS:
            raise HTTPException(413, f"text exceeds {MAX_TEXT_CHARS} characters")
        body = req.model_dump()
        body["voice"] = _resolve_voice(voices_dir, req.voice)
        return store.create_job("tts", body, "tts.render", {}, idempotency_key=idempotency_key)

    @app.get("/v1/tts/jobs", dependencies=deps)
    def list_tts_jobs(limit: int = 50):
        return store.list_jobs("tts", limit=min(limit, 200))

    @app.get("/v1/jobs/{job_id}", dependencies=deps)
    @app.get("/v1/tts/jobs/{job_id}", dependencies=deps)
    def get_job(job_id: str):
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(404)
        return job

    @app.get("/v1/tts/jobs/{job_id}/audio", dependencies=deps)
    def job_audio(job_id: str):
        p = worker.job_dir(job_id) / "final.wav"
        if not store.get_job(job_id) or not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type="audio/wav", headers={"Cache-Control": "no-store"},
                            filename=f"{job_id}.wav")

    @app.post("/v1/tts/jobs/{job_id}/chunks/{idx}/regenerate", status_code=202, dependencies=deps)
    def regenerate_chunk(job_id: str, idx: int, req: RegenRequest):
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(404)
        if job["status"] not in ("succeeded", "partial") or store.has_pending(job_id):
            raise HTTPException(409, f"job is {job['status']}; wait for it to finish")
        if not 0 <= idx < len(job["result"]["chunks"]):
            raise HTTPException(404, "no such chunk")
        seed = req.seed if req.seed is not None else random.randrange(1, 2**31)
        store.add_task(job_id, "tts.regen_chunk", {"idx": idx, "seed": seed}, priority=PRIORITY_INTERACTIVE)
        store.update_job(job_id, status="queued")
        return store.get_job(job_id)

    if web_dist and (web_dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")

    return app
