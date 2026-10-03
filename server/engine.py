"""Model lifecycle: the local stand-in for the GPU VM's start/stop.

States: stopped -> loading -> ready <-> busy -> (stopping) -> stopped, or error.
Stop never interrupts a generation: while busy it only marks the engine to
unload once the current task finishes.
"""

import gc
import threading
import time
from typing import Any, Callable, Dict, Optional


class Engine:
    def __init__(self, loader: Callable[[], Any], idle_timeout_s: float = 0):
        self._loader = loader
        self._lock = threading.Lock()
        self.idle_timeout_s = idle_timeout_s
        self.state = "stopped"
        self.error: Optional[str] = None
        self.tts = None
        self.loaded_at: Optional[float] = None
        self.last_used: float = time.time()
        self._stop_after_task = False

    def status(self) -> Dict[str, Any]:
        info = None
        if self.tts is not None:
            from voicekit.voice import model_info

            info = model_info(self.tts)
        return {
            "state": self.state,
            "error": self.error,
            "model": info,
            "loaded_at": self.loaded_at,
            "idle_s": round(time.time() - self.last_used, 1) if self.state == "ready" else 0,
            "idle_timeout_s": self.idle_timeout_s,
            "stop_pending": self._stop_after_task,
        }

    def start(self) -> None:
        with self._lock:
            if self.state not in ("stopped", "error"):
                self._stop_after_task = False
                return
            self.state, self.error = "loading", None
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            tts = self._loader()
        except Exception as e:  # noqa: BLE001 - surface any load failure to the UI
            with self._lock:
                self.state, self.error = "error", f"{type(e).__name__}: {e}"
            return
        with self._lock:
            self.tts, self.state = tts, "ready"
            self.loaded_at = self.last_used = time.time()
            self._stop_after_task = False

    def stop(self) -> None:
        with self._lock:
            if self.state == "busy":
                self._stop_after_task = True
                return
            if self.state != "ready":
                return
        self._unload()

    def _unload(self) -> None:
        with self._lock:
            self.state = "stopping"
            self.tts = None
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cache release is best-effort
            pass
        with self._lock:
            self.state, self.loaded_at, self._stop_after_task = "stopped", None, False

    def acquire(self) -> bool:
        with self._lock:
            if self.state != "ready":
                return False
            self.state = "busy"
            return True

    def release(self) -> None:
        with self._lock:
            self.state = "ready"
            self.last_used = time.time()
            stop = self._stop_after_task
        if stop:
            self._unload()

    def maybe_idle_stop(self) -> None:
        if self.idle_timeout_s and self.state == "ready" and time.time() - self.last_used > self.idle_timeout_s:
            self.stop()
