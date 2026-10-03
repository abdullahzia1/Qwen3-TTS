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

from .audio import PAUSE_MS, loudness, normalize_loudness, stitch, trim_silence
from .text import Chunk, chunk, normalize
from .voice import Voice, model_info

CODEC_FPS = 12.5

# Duration bounds relative to the voice's measured speaking rate. Outside them,
# the chunk most likely skipped words or ran past the text. Starting values;
# tighten them once the benchmark shows the real spread.
MIN_RATIO = 0.5
MAX_RATIO = 1.7


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


def _check(wav: np.ndarray, sr: int, expected_s: float) -> bool:
    dur = len(wav) / sr
    return MIN_RATIO * expected_s <= dur <= MAX_RATIO * expected_s


def render_chunk(
    tts,
    voice: Voice,
    idx: int,
    c: Chunk,
    seed: int,
    params: Dict[str, Any],
    max_attempts: int = 3,
):
    """Render one chunk on its own, retrying with derived seeds. Returns (wav, sr, ChunkResult)."""
    expected = _expected_seconds(c.text, voice)
    best, best_res = None, None
    for attempt in range(1, max_attempts + 1):
        s = seed if attempt == 1 else seed + 10_000 * attempt
        wavs, sr = _generate(tts, voice, [c.text], s, params, _token_cap(expected))
        wav = wavs[0]
        ok = _check(wav, sr, expected)
        if best is None or ok or abs(len(wav) / sr - expected) < abs(best_res.duration_s - expected):
            best = wav
            best_res = ChunkResult(
                idx=idx, text=c.text, boundary=c.boundary, seed=s, attempts=attempt,
                duration_s=round(len(wav) / sr, 3), expected_s=round(expected, 3), status="ok" if ok else "suspect",
            )
        best_res.attempts = attempt
        if ok:
            break
    return best, sr, best_res


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
    return {
        "duration_s": round(duration, 3),
        "lufs": round(loudness(out, sr), 2),
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
    if not chunks:
        raise ValueError("Nothing to synthesize after normalization.")

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
            ok = _check(wav, sr, expected[i])
            audio[i] = wav
            results[i] = ChunkResult(
                idx=i, text=chunks[i].text, boundary=chunks[i].boundary, seed=batch_seed, attempts=1,
                duration_s=round(len(wav) / sr, 3), expected_s=round(expected[i], 3), status="ok" if ok else "suspect",
            )
        if on_progress:
            on_progress(idxs[-1] + 1, len(chunks))

    # Retries one chunk at a time with fresh seeds; keep the take closest to
    # the expected length if none passes.
    for i, res in enumerate(results):
        if res.status == "ok" or max_attempts < 2:
            continue
        wav, sr, retry = render_chunk(tts, voice, i, chunks[i], seed * 100_003 + 20_000 + i, gen_params, max_attempts - 1)
        retry.attempts += 1
        if retry.status == "ok" or abs(retry.duration_s - expected[i]) < abs(res.duration_s - expected[i]):
            audio[i], results[i] = wav, retry
        else:
            res.attempts = retry.attempts

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
    return SynthResult(wav=out, sr=sr, chunks=results, meta=meta, chunk_wavs=audio)
