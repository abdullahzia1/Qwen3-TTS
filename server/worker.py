"""Single worker thread: pulls tasks, runs them on the loaded model, stores audio."""

import json
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

from voicekit.synth import ChunkResult, assemble, render_chunk, summarize, synthesize
from voicekit.text import Chunk
from voicekit.voice import create_voice, load_voice, model_info

from .db import Store, now
from .engine import Engine


class Worker:
    def __init__(self, store: Store, engine: Engine, data_dir: Path, autostart: bool = True):
        self.store, self.engine, self.data_dir = store, engine, data_dir
        self.autostart = autostart
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="voicekit-worker")

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / "jobs" / job_id

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._shutdown.set()
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._shutdown.is_set():
            if self.store.queue_depth() == 0:
                self.engine.maybe_idle_stop()
                time.sleep(0.3)
                continue
            if self.engine.state in ("stopped", "error"):
                if self.autostart and self.engine.state == "stopped":
                    self.engine.start()
                time.sleep(0.3)
                continue
            if not self.engine.acquire():
                time.sleep(0.3)
                continue
            task = self.store.claim_task()
            if task is None:
                self.engine.release()
                continue
            try:
                self._run(task)
                self.store.finish_task(task["id"])
            except Exception as e:  # noqa: BLE001 - a bad task must not kill the worker
                msg = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                self.store.finish_task(task["id"], error=msg)
                self.store.update_job(task["job_id"], status="failed", error=msg, finished_at=now())
            finally:
                self.engine.release()

    def _run(self, task: Dict[str, Any]) -> None:
        op = task["op"]
        job = self.store.get_job(task["job_id"])
        self.store.update_job(job["id"], status="running", started_at=job["started_at"] or now(), error=None)
        if op == "tts.render":
            self._render(job)
        elif op == "tts.regen_chunk":
            self._regen_chunk(job, **task["args"])
        elif op == "voice.create":
            self._create_voice(job, **task["args"])
        else:
            raise ValueError(f"unknown op {op}")

    # --- ops ----------------------------------------------------------------
    def _render(self, job: Dict[str, Any]) -> None:
        req = job["request"]
        name, _, ver = req["voice"].partition("/v")
        voice = load_voice(str(self.voices_dir), name, int(ver) if ver else None)

        def progress(done: int, total: int) -> None:
            self.store.update_job(job["id"], progress=round(done / total, 3))

        res = synthesize(
            self.engine.tts, voice, req["text"], seed=req.get("seed", 0), params=req.get("params") or {},
            lexicon=req.get("lexicon"), max_chars=req.get("max_chars", 400),
            batch_size=req.get("batch_size", 1), target_lufs=req.get("target_lufs", -16.0), on_progress=progress,
        )
        d = self.job_dir(job["id"])
        (d / "chunks").mkdir(parents=True, exist_ok=True)
        for i, w in enumerate(res.chunk_wavs):
            sf.write(d / "chunks" / f"{i:03d}.wav", w, res.sr, subtype="FLOAT")
        self._save_final(job["id"], res.wav, res.sr)
        result = {**res.to_dict(), "sample_rate": res.sr, "audio_version": 1}
        self._finish(job["id"], result)

    def _regen_chunk(self, job: Dict[str, Any], idx: int, seed: int) -> None:
        req, result = job["request"], job["result"]
        name, _, ver = result["meta"]["voice"].partition("/v")
        voice = load_voice(str(self.voices_dir), name, int(ver))
        chunks = [ChunkResult(**c) for c in result["chunks"]]
        target = chunks[idx]
        params = result["meta"]["params"]

        others = [c.f0_median_hz for c in chunks if c.idx != idx and c.f0_median_hz]
        target_f0 = float(np.median(others)) if len(others) >= 2 else None
        wav, sr, new = render_chunk(self.engine.tts, voice, idx, Chunk(target.text, target.boundary), seed, params,
                                    target_f0=target_f0)
        d = self.job_dir(job["id"])
        sf.write(d / "chunks" / f"{idx:03d}.wav", wav, sr, subtype="FLOAT")
        chunks[idx] = new

        raw: List[np.ndarray] = [sf.read(d / "chunks" / f"{i:03d}.wav", dtype="float32")[0] for i in range(len(chunks))]
        out = assemble(raw, chunks, sr, result["meta"].get("target_lufs", req.get("target_lufs", -16.0)))
        version = result.get("audio_version", 1) + 1
        self._save_final(job["id"], out, sr)
        meta = {**result["meta"], **summarize(chunks, out, sr), "model": model_info(self.engine.tts)}
        regens = result.get("regenerations", []) + [{"idx": idx, "seed": new.seed, "at": now()}]
        self._finish(job["id"], {
            "meta": meta, "chunks": [asdict(c) for c in chunks], "sample_rate": sr,
            "audio_version": version, "regenerations": regens,
        })

    def _create_voice(self, job: Dict[str, Any], ref_path: str) -> None:
        req = job["request"]
        try:
            v = self._build_voice(req, ref_path)
        finally:
            # The cleaned reference is stored inside the voice version; the raw upload is no longer needed.
            Path(ref_path).unlink(missing_ok=True)
        self.store.update_job(job["id"], status="succeeded", progress=1.0, finished_at=now(),
                              result={"voice": v.id, "meta": v.meta})

    def _build_voice(self, req: Dict[str, Any], ref_path: str):
        return create_voice(
            self.engine.tts, str(self.voices_dir), req["name"], ref_path, req.get("ref_text", ""),
            language=req.get("language", "English"), x_vector_only=req.get("x_vector_only", False),
            params=req.get("params") or {}, notes=req.get("notes", ""),
        )

    # --- helpers ------------------------------------------------------------
    def _save_final(self, job_id: str, wav: np.ndarray, sr: int) -> None:
        d = self.job_dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "final.tmp.wav"
        sf.write(tmp, wav, sr, subtype="PCM_16")
        tmp.replace(d / "final.wav")

    def _finish(self, job_id: str, result: Dict[str, Any]) -> None:
        status = "partial" if result["meta"]["suspect_chunks"] else "succeeded"
        (self.job_dir(job_id) / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        self.store.update_job(job_id, status=status, progress=1.0, result=result, finished_at=now())
