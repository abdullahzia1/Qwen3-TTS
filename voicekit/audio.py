"""Post-processing: trim, loudness-match, stitch."""

from typing import List, Sequence

import librosa
import numpy as np
import pyloudnorm as pyln
from scipy.ndimage import maximum_filter1d, uniform_filter1d

PAUSE_MS = {"sentence": 250, "paragraph": 600, "end": 0}


def trim_silence(wav: np.ndarray, sr: int, threshold_db: float = -45.0, pad_ms: int = 40) -> np.ndarray:
    """Cut leading/trailing audio quieter than `threshold_db` relative to the peak."""
    if wav.size == 0:
        return wav
    frame = max(1, int(sr * 0.01))
    n = len(wav) // frame
    if n == 0:
        return wav
    rms = np.sqrt(np.mean(wav[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    level = 20 * np.log10(rms / (np.max(rms) + 1e-12))
    voiced = np.flatnonzero(level > threshold_db)
    if voiced.size == 0:
        return wav[:0]
    pad = int(sr * pad_ms / 1000)
    start = max(0, voiced[0] * frame - pad)
    end = min(len(wav), (voiced[-1] + 1) * frame + pad)
    return wav[start:end]


def pitch_profile(wav: np.ndarray, sr: int, window_s: float = 2.0, step_s: float = 0.5,
                  fmin: float = 60.0, fmax: float = 500.0):
    """Median F0 of the voiced frames, plus the median of every `window_s` window.

    Windows slide by `step_s`, so a pitch excursion is measured the same way no
    matter where it sits in the clip (fixed windows gave +36% or +64% for the
    same take depending on leading silence). Windows with too little voiced
    speech (pauses, breaths) are skipped so they can't read as jumps.
    Returns (nan, []) when nothing is voiced.
    """
    hop = 256
    f0, voiced, _ = librosa.pyin(wav, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=hop)
    f0 = np.where(voiced, f0, np.nan)
    if np.all(np.isnan(f0)):
        return float("nan"), []
    n = max(1, int(window_s * sr / hop))
    step = max(1, int(step_s * sr / hop))
    windows = []
    for i in range(0, max(1, len(f0) - n + 1), step):
        w = f0[i : i + n]
        if np.sum(~np.isnan(w)) >= 0.25 * len(w):
            windows.append(float(np.nanmedian(w)))
    return float(np.nanmedian(f0)), windows


def loudness(wav: np.ndarray, sr: int) -> float:
    """Integrated loudness in LUFS (BS.1770). -inf for clips shorter than 400 ms."""
    if len(wav) < int(0.4 * sr):
        return float("-inf")
    return float(pyln.Meter(sr).integrated_loudness(wav.astype(np.float64)))


def limit(wav: np.ndarray, sr: int, ceiling_dbfs: float = -1.0, lookahead_ms: float = 5.0) -> np.ndarray:
    """Look-ahead peak limiter: only the samples around a peak are turned down.

    The raw gain is min-held over 2x the look-ahead window and then averaged
    over 1x, so the smoothed gain at any peak is still at or below what that
    peak needs - no overshoot, no clicks.
    """
    ceiling = 10 ** (ceiling_dbfs / 20)
    la = max(1, int(sr * lookahead_ms / 1000))
    env = maximum_filter1d(np.abs(wav), size=2 * la + 1)
    gain = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))
    gain = uniform_filter1d(gain, size=la)
    return (wav * gain).astype(np.float32)


def normalize_loudness(wav: np.ndarray, sr: int, target_lufs: float, peak_dbfs: float = -1.0) -> np.ndarray:
    """Gain to `target_lufs`, limiting peaks to `peak_dbfs` instead of lowering the whole clip."""
    out = wav.astype(np.float32)
    # Limiting removes a little loudness, so correct once more after the first pass.
    for _ in range(2):
        current = loudness(out, sr)
        if not np.isfinite(current):
            return out
        out = limit(out * (10 ** ((target_lufs - current) / 20)), sr, peak_dbfs)
    return out


def stitch(
    wavs: Sequence[np.ndarray],
    boundaries: Sequence[str],
    sr: int,
    pause_ms: dict = PAUSE_MS,
    crossfade_ms: int = 15,
) -> np.ndarray:
    """Join chunks with fixed pauses. Fades each edge so joins never click."""
    fade = int(sr * crossfade_ms / 1000)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32) if fade else None
    parts: List[np.ndarray] = []
    for wav, boundary in zip(wavs, boundaries):
        w = wav.astype(np.float32).copy()
        if ramp is not None and len(w) > 2 * fade:
            w[:fade] *= ramp
            w[-fade:] *= ramp[::-1]
        parts.append(w)
        gap = int(sr * pause_ms.get(boundary, 0) / 1000)
        if gap:
            parts.append(np.zeros(gap, dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
