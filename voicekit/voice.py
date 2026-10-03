"""Voice versions: an immutable bundle of reference audio, transcript, clone
prompt and default generation parameters.

Layout on disk (one directory per version, never modified after creation):

    <root>/<name>/v<N>/
        voice.json   metadata + default params
        ref.wav      the cleaned reference clip actually used (24 kHz mono)
        prompt.pt    clone prompt, same payload format as the Gradio demo's "Save Voice"

A job that records (name, version) can always be regenerated identically.
"""

import datetime as dt
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
import torch

from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

from .audio import normalize_loudness, pitch_profile, trim_silence

SAMPLE_RATE = 24000
REF_LUFS = -20.0
# The model conditions on the whole reference before every chunk. Long
# references were measured to destabilise pitch (a 57 s clip caused a +59%
# pitch jump that a 10 s excerpt of the same recording did not) and slow
# generation down, so warn above REF_WARN_S and refuse above REF_MAX_S.
REF_MIN_S = 3.0
REF_WARN_S = 20.0
REF_MAX_S = 30.0

# Starting point for the Phase 1 sweep, not a tuned result.
DEFAULT_PARAMS: Dict[str, Any] = {
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "subtalker_temperature": 0.9,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "non_streaming_mode": False,
}


@dataclass
class Voice:
    name: str
    version: int
    path: Path
    ref_text: str
    language: str
    mode: str
    chars_per_sec: float
    params: Dict[str, Any]
    # Median pitch of the reference; the yardstick for pitch-spike checks.
    f0_median_hz: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    prompt: List[VoiceClonePromptItem] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.name}/v{self.version}"


def _git_sha(repo: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def model_info(tts) -> Dict[str, Any]:
    m = tts.model
    return {
        "name_or_path": getattr(m.config, "_name_or_path", None),
        "tts_model_type": m.tts_model_type,
        "tts_model_size": m.tts_model_size,
        "dtype": str(m.dtype),
        "device": str(tts.device),
        "qwen_tts_git": _git_sha(Path(__file__).resolve().parent.parent),
    }


def prepare_reference(path: str) -> np.ndarray:
    """Load, downmix, resample to 24 kHz, trim silence, set a fixed loudness."""
    wav, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    wav = trim_silence(wav, SAMPLE_RATE)
    return normalize_loudness(wav, SAMPLE_RATE, REF_LUFS)


def _next_version(voice_dir: Path) -> int:
    versions = [int(p.name[1:]) for p in voice_dir.glob("v*") if p.name[1:].isdigit()]
    return max(versions, default=0) + 1


def create_voice(
    tts,
    root: str,
    name: str,
    ref_audio: str,
    ref_text: str,
    language: str = "English",
    x_vector_only: bool = False,
    params: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> Voice:
    """Build a clone prompt from a reference clip and save it as a new version."""
    if tts.model.tts_model_type != "base":
        raise ValueError("Voices must be built with the Base model (voice clone).")
    ref_text = " ".join(ref_text.split())
    if not x_vector_only and not ref_text:
        raise ValueError("ref_text is required unless x_vector_only=True.")

    wav = prepare_reference(ref_audio)
    duration = len(wav) / SAMPLE_RATE
    if duration < REF_MIN_S:
        raise ValueError(f"Reference is {duration:.1f}s after trimming; use at least {REF_MIN_S:.0f}s (8-15s recommended).")
    if duration > REF_MAX_S:
        raise ValueError(
            f"Reference is {duration:.1f}s; use at most {REF_MAX_S:.0f}s (8-15s recommended). Long references make "
            "pitch unstable and generation slow. Cut a clean excerpt and give its exact transcript."
        )
    warnings = []
    if duration > REF_WARN_S:
        warnings.append(f"Reference is {duration:.0f}s; 8-15s gives steadier pitch and faster generation.")
    f0_median, _ = pitch_profile(wav, SAMPLE_RATE)

    items = tts.create_voice_clone_prompt(
        ref_audio=(wav, SAMPLE_RATE), ref_text=ref_text or None, x_vector_only_mode=x_vector_only
    )

    voice_dir = Path(root) / name
    version = _next_version(voice_dir)
    vdir = voice_dir / f"v{version}"
    vdir.mkdir(parents=True, exist_ok=False)

    sf.write(vdir / "ref.wav", wav, SAMPLE_RATE, subtype="PCM_16")
    torch.save({"items": [asdict(it) for it in items]}, vdir / "prompt.pt")

    merged = {**DEFAULT_PARAMS, **(params or {})}
    meta = {
        "name": name,
        "version": version,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "xvec" if x_vector_only else "icl",
        "language": language,
        "ref_text": ref_text,
        "ref_source": str(ref_audio),
        "ref_sha256": hashlib.sha256((vdir / "ref.wav").read_bytes()).hexdigest(),
        "ref_duration_s": round(duration, 3),
        # Speaking rate of the reference; used to sanity-check generated chunk lengths.
        "chars_per_sec": round(len(ref_text) / duration, 3) if ref_text else None,
        "f0_median_hz": round(f0_median, 1) if np.isfinite(f0_median) else None,
        "warnings": warnings,
        "params": merged,
        "model": model_info(tts),
        "notes": notes,
    }
    (vdir / "voice.json").write_text(json.dumps(meta, indent=2) + "\n")
    return load_voice(root, name, version)


def load_voice(root: str, name: str, version: Optional[int] = None) -> Voice:
    """Load a voice version; `version=None` picks the latest."""
    voice_dir = Path(root) / name
    if version is None:
        version = _next_version(voice_dir) - 1
        if version < 1:
            raise FileNotFoundError(f"No versions for voice '{name}' in {root}")
    vdir = voice_dir / f"v{version}"
    meta = json.loads((vdir / "voice.json").read_text())
    payload = torch.load(vdir / "prompt.pt", map_location="cpu", weights_only=True)
    items = [
        VoiceClonePromptItem(
            ref_code=d.get("ref_code"),
            ref_spk_embedding=d["ref_spk_embedding"],
            x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
            icl_mode=bool(d.get("icl_mode", not d.get("x_vector_only_mode", False))),
            ref_text=d.get("ref_text"),
        )
        for d in payload["items"]
    ]
    if meta.get("ref_duration_s", 0) > REF_WARN_S and not meta.get("warnings"):
        # Versions made before the length guard existed.
        meta["warnings"] = [f"Reference is {meta['ref_duration_s']:.0f}s; 8-15s gives steadier pitch and faster generation."]
    f0 = meta.get("f0_median_hz")
    if f0 is None and (vdir / "ref.wav").exists():
        # Versions made before pitch tracking: measure now, in memory only
        # (voice versions are never rewritten on disk).
        wav, _ = sf.read(vdir / "ref.wav", dtype="float32")
        med, _ = pitch_profile(wav, SAMPLE_RATE)
        f0 = med if np.isfinite(med) else None
    return Voice(
        name=name,
        version=version,
        path=vdir,
        f0_median_hz=f0,
        ref_text=meta["ref_text"],
        language=meta["language"],
        mode=meta["mode"],
        chars_per_sec=meta.get("chars_per_sec") or 14.0,
        params=meta["params"],
        meta=meta,
        prompt=items,
    )
