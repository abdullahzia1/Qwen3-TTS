"""Run the voicekit API + UI.

    QWEN_BASE_MODEL=/path/to/Qwen3-TTS-12Hz-1.7B-Base python -m server

Env:
    QWEN_BASE_MODEL     Base model dir or HF id (required)
    VOICEKIT_DATA       data root: voices/, jobs/, uploads/, voicekit.db (default ./data)
    VOICEKIT_API_KEY    if set, every /v1 call needs "Authorization: Bearer <key>"
    VOICEKIT_IDLE_STOP  unload the model after this many idle seconds (default 1200, 0 = never)
    VOICEKIT_AUTOSTART  load the model automatically when a job is queued (default 1)
    VOICEKIT_HOST/PORT  bind address (default 127.0.0.1:8100)
"""

import os
import sys
from pathlib import Path

import uvicorn

from .app import create_app
from .db import Store
from .engine import Engine
from .worker import Worker


def main() -> int:
    model = os.environ.get("QWEN_BASE_MODEL")
    if not model:
        print("QWEN_BASE_MODEL is required", file=sys.stderr)
        return 2
    data = Path(os.environ.get("VOICEKIT_DATA", "data")).resolve()
    (data / "voices").mkdir(parents=True, exist_ok=True)

    from voicekit.__main__ import _load_model

    store = Store(str(data / "voicekit.db"))
    engine = Engine(lambda: _load_model(model), idle_timeout_s=float(os.environ.get("VOICEKIT_IDLE_STOP", "1200")))
    worker = Worker(store, engine, data, autostart=os.environ.get("VOICEKIT_AUTOSTART", "1") == "1")
    web = Path(__file__).resolve().parent.parent / "web" / "dist"
    app = create_app(store, engine, worker, data, api_key=os.environ.get("VOICEKIT_API_KEY"), web_dist=web)
    worker.start()
    uvicorn.run(app, host=os.environ.get("VOICEKIT_HOST", "127.0.0.1"), port=int(os.environ.get("VOICEKIT_PORT", "8100")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
