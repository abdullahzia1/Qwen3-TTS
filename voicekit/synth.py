"""Long-form synthesis with a fixed voice: chunk, generate, check, retry, stitch.

Every chunk is conditioned on the same voice prompt (never on the previous
chunk's audio), so drift cannot compound across a long script.
"""

import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .audio import PAUSE_MS, loudness, normalize_loudness, pitch_profile, stitch, trim_silence
from .text import Chunk, chunk, normalize
from .voice import Voice, model_info

CODEC_FPS = 12.5

# Duration bounds relative to the voice's measured speaking rate. Outside them,
# the chunk most likely skipped words or ran past the text. Starting values;
# tighten them once the benchmark shows the real spread.
MIN_RATIO = 0.5
MAX_RATIO = 1.7

# Pitch checks. All run on the trimmed speech with sliding 2 s windows, and
# only when there is >= MIN_PITCH_SPEECH_S of speech (shorter clips give the
# pitch tracker too few frames to be trusted). Thresholds calibrated on the
# 96-take baseline (2026-10-04) plus the jump users heard (+59%):
# SPIKE_MAX_DEV: worst window vs the voice's reference pitch. 45% flags 3.4%
#   of baseline takes; 40% also flagged natural rises on questions.
# REGISTER_MAX_DEV: whole-chunk median vs reference pitch (catches a line read
#   entirely higher, e.g. +46%). Flags 2.3% of baseline takes.
# CONSISTENCY_MAX_DEV: chunk median vs the median of the job's chunks (3+).
SPIKE_MAX_DEV = 0.45
REGISTER_MAX_DEV = 0.30
CONSISTENCY_MAX_DEV = 0.20
MIN_PITCH_SPEECH_S = 1.0
SLOW_RTF = 5.0


@dataclass
class ChunkResult:
    idx: int
    text: str
    boundary: str
    seed: int
    attempts: int
    duration_s: float
    expected_s: float
    status: str  # "ok" | "suspect" (kept after exhausting retries)
    start_s: float = 0.0
    end_s: float = 0.0
    f0_median_hz: Optional[float] = None
    issue: Optional[str] = None  # why the chunk is suspect, e.g. "pitch spike +59%"


@dataclass
class SynthResult:
    wav: np.ndarray
    sr: int
    chunks: List[ChunkResult]
    meta: Dict[str, Any] = field(default_factory=dict)
    # Raw model output per chunk, before trim/loudness, so one chunk can be
    # regenerated and the whole file re-assembled.
    chunk_wavs: List[np.ndarray] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"meta": self.meta, "chunks": [asdict(c) for c in self.chunks]}


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _expected_seconds(text: str, voice: Voice) -> float:
    return max(0.5, len(text) / voice.chars_per_sec)


def _token_cap(expected_s: float) -> int:
    """Hard stop for a runaway generation: generous, but nowhere near 8192."""
    return int(math.ceil(expected_s * (MAX_RATIO + 0.5) * CODEC_FPS)) + 25


def _generate(tts, voice: Voice, texts: List[str], seed: int, params: Dict[str, Any], max_new_tokens: int):
    gen = dict(params)
    non_streaming = bool(gen.pop("non_streaming_mode", False))
    set_seed(seed)
    wavs, sr = tts.generate_voice_clone(
        text=texts,
        language=[voice.language] * len(texts),
        voice_clone_prompt=voice.prompt * len(texts),
        non_streaming_mode=non_streaming,
        max_new_tokens=max_new_tokens,
        **gen,
    )
    return wavs, sr


def _inspect(wav: np.ndarray, sr: int, expected_s: float, voice: Voice, target_f0: Optional[float] = None):
    """Check one take. Returns (issue or None, median f0, badness) - lower badness is better."""
    ratio = (len(wav) / sr) / expected_s
    speech = trim_silence(wav, sr)
    med, windows = float("nan"), []
    if len(speech) / sr >= MIN_PITCH_SPEECH_S:
        med, windows = pitch_profile(speech, sr)
    f0 = med if np.isfinite(med) else None
    ref = voice.f0_median_hz

    spike = max((w / ref - 1 for w in windows), key=abs) if (ref and windows) else 0.0
    register = (f0 / ref - 1) if (ref and f0) else 0.0
    drift = (f0 / target_f0 - 1) if (target_f0 and f0) else 0.0
    len_bad = max(0.0, MIN_RATIO - ratio) / MIN_RATIO + max(0.0, ratio - MAX_RATIO) / MAX_RATIO

    issue = None
    if ratio < MIN_RATIO:
        issue = f"too short ({ratio:.0%} of expected)"
    elif ratio > MAX_RATIO:
        issue = f"too long ({ratio:.0%} of expected)"
    elif abs(register) > REGISTER_MAX_DEV:
        issue = f"pitch {register:+.0%} vs voice"
    elif abs(spike) > SPIKE_MAX_DEV:
        issue = f"pitch spike {spike:+.0%}"
    elif abs(drift) > CONSISTENCY_MAX_DEV:
        issue = f"pitch {drift:+.0%} vs other chunks"
    badness = (len_bad + max(0.0, abs(register) - REGISTER_MAX_DEV) + max(0.0, abs(spike) - SPIKE_MAX_DEV)
               + max(0.0, abs(drift) - CONSISTENCY_MAX_DEV))
    return issue, f0, badness


def _result(idx: int, c: Chunk, seed: int, attempts: int, wav, sr, expected: float, issue, f0) -> ChunkResult:
    return ChunkResult(
        idx=idx, text=c.text, boundary=c.boundary, seed=seed, attempts=attempts,
        duration_s=round(len(wav) / sr, 3), expected_s=round(expected, 3),
        status="suspect" if issue else "ok", issue=issue,
        f0_median_hz=round(f0, 1) if f0 else None,
    )


def render_chunk(
    tts,
    voice: Voice,
    idx: int,
    c: Chunk,
    seed: int,
    params: Dict[str, Any],
    max_attempts: int = 3,
    target_f0: Optional[float] = None,
):
    """Render one chunk on its own, retrying with derived seeds until a take passes.

    `target_f0` (the median pitch of the job's other chunks) adds the
    consistency check. Returns (wav, sr, ChunkResult) for the best take.
    """
    expected = _expected_seconds(c.text, voice)
    best = None  # (badness, wav, result)
    sr = 24000
    for attempt in range(1, max_attempts + 1):
        s = seed if attempt == 1 else seed + 10_000 * attempt
        wavs, sr = _generate(tts, voice, [c.text], s, params, _token_cap(expected))
        issue, f0, bad = _inspect(wavs[0], sr, expected, voice, target_f0)
        if best is None or bad < best[0] or (issue is None and best[2].issue is not None):
            best = (bad, wavs[0], _result(idx, c, s, attempt, wavs[0], sr, expected, issue, f0))
        best[2].attempts = attempt
        if issue is None:
            break
    return best[1], sr, best[2]


def assemble(raw: List[np.ndarray], results: List[ChunkResult], sr: int, target_lufs: float = -16.0) -> np.ndarray:
    """Trim, loudness-match, stitch and master raw chunk audio. Fills start_s/end_s on `results`."""
    cleaned = [normalize_loudness(trim_silence(w, sr), sr, target_lufs) for w in raw]
    out = stitch(cleaned, [r.boundary for r in results], sr)
    out = normalize_loudness(out, sr, target_lufs)
    pos = 0.0
    for res, w in zip(results, cleaned):
        res.start_s = round(pos, 3)
        pos += len(w) / sr
        res.end_s = round(pos, 3)
        pos += PAUSE_MS.get(res.boundary, 0) / 1000
    return out


def summarize(results: List[ChunkResult], out: np.ndarray, sr: int) -> Dict[str, Any]:
    duration = len(out) / sr
    lufs = loudness(out, sr)
    return {
        "duration_s": round(duration, 3),
        # Silence or sub-400ms audio has no measurable loudness; JSON has no -inf.
        "lufs": round(lufs, 2) if np.isfinite(lufs) else None,
        "suspect_chunks": [r.idx for r in results if r.status != "ok"],
        "retried_chunks": [r.idx for r in results if r.attempts > 1],
    }


def synthesize(
    tts,
    voice: Voice,
    text: str,
    seed: int = 0,
    params: Optional[Dict[str, Any]] = None,
    lexicon: Optional[Dict[str, str]] = None,
    max_chars: int = 400,
    batch_size: int = 1,
    max_attempts: int = 3,
    target_lufs: float = -16.0,
    on_progress=None,
) -> SynthResult:
    """Render `text` in `voice`.

    `params` overrides the voice's saved defaults for this call only.
    `batch_size` > 1 is faster on a GPU, but batch composition changes the
    sampled output, so benchmark with the same batch size you produce with.
    `on_progress(done, total)` is called after each first-pass batch.
    """
    t0 = time.time()
    gen_params = {**voice.params, **(params or {})}
    norm = normalize(text, lexicon)
    chunks: List[Chunk] = chunk(norm, max_chars=max_chars)
    if not chunks or not any(ch.isalnum() for ch in norm):
        raise ValueError("Nothing speakable in the text (no letters or digits).")

    expected = [_expected_seconds(c.text, voice) for c in chunks]
    audio: List[Optional[np.ndarray]] = [None] * len(chunks)
    results: List[Optional[ChunkResult]] = [None] * len(chunks)
    sr = 24000

    # First pass, batched. Seeds are derived from (seed, position) so a rerun
    # with the same inputs reproduces the same takes.
    for b in range(0, len(chunks), batch_size):
        idxs = list(range(b, min(b + batch_size, len(chunks))))
        cap = max(_token_cap(expected[i]) for i in idxs)
        batch_seed = seed * 100_003 + b
        wavs, sr = _generate(tts, voice, [chunks[i].text for i in idxs], batch_seed, gen_params, cap)
        for i, wav in zip(idxs, wavs):
            issue, f0, _ = _inspect(wav, sr, expected[i], voice)
            audio[i] = wav
            results[i] = _result(i, chunks[i], batch_seed, 1, wav, sr, expected[i], issue, f0)
        if on_progress:
            on_progress(idxs[-1] + 1, len(chunks))

    def retry(i: int, seed_base: int, attempts: int, target_f0: Optional[float]) -> None:
        old = results[i]
        wav, _, new = render_chunk(tts, voice, i, chunks[i], seed_base, gen_params, attempts, target_f0)
        new.attempts += old.attempts
        _, _, old_bad = _inspect(audio[i], sr, expected[i], voice, target_f0)
        _, _, new_bad = _inspect(wav, sr, expected[i], voice, target_f0)
        if new.issue is None or new_bad < old_bad:
            audio[i], results[i] = wav, new
        else:
            old.attempts = new.attempts

    # Pass 2: retry chunks that failed their own checks (length, pitch spike).
    if max_attempts > 1:
        for i, res in enumerate(results):
            if res.issue:
                retry(i, seed * 100_003 + 20_000 + i, max_attempts - 1, None)

    # Pass 3: consistency. A chunk whose median pitch sits far from the rest of
    # the job is re-rendered toward the job's median.
    target_f0 = None
    if len(chunks) >= 3:
        f0s = [r.f0_median_hz for r in results if r.f0_median_hz]
        if len(f0s) >= 3:
            target_f0 = float(np.median(f0s))
            for i, res in enumerate(results):
                if res.f0_median_hz and abs(res.f0_median_hz / target_f0 - 1) > CONSISTENCY_MAX_DEV:
                    results[i].issue = results[i].issue or f"pitch {res.f0_median_hz / target_f0 - 1:+.0%} vs other chunks"
                    results[i].status = "suspect"
                    if max_attempts > 1:
                        retry(i, seed * 100_003 + 30_000 + i, max_attempts - 1, target_f0)

    out = assemble(audio, results, sr, target_lufs)
    elapsed = time.time() - t0
    meta = {
        "voice": voice.id,
        "model": model_info(tts),
        "params": gen_params,
        "seed": seed,
        "batch_size": batch_size,
        "max_chars": max_chars,
        "target_lufs": target_lufs,
        "text_normalized": norm,
        "elapsed_s": round(elapsed, 2),
        **summarize(results, out, sr),
    }
    meta["rtf"] = round(elapsed / meta["duration_s"], 3) if meta["duration_s"] else None
    meta["pitch_target_hz"] = round(target_f0, 1) if target_f0 else None
    warnings = list(voice.meta.get("warnings", []))
    if meta["rtf"] and meta["rtf"] > SLOW_RTF:
        warnings.append(
            f"Generation ran at {meta['rtf']:.0f}x real time (normally ~2x on this Mac). "
            "The machine may be short on memory; check Memory Pressure in Activity Monitor."
        )
    meta["warnings"] = warnings
    return SynthResult(wav=out, sr=sr, chunks=results, meta=meta, chunk_wavs=audio)
