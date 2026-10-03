import numpy as np
import pytest

from voicekit.audio import loudness, normalize_loudness, stitch, trim_silence
from voicekit.text import chunk, normalize

SR = 24000


def test_normalize_applies_lexicon_whole_word_longest_first():
    lex = {"AI": "A I", "AI agents": "A I agents", "GCP": "G C P"}
    assert normalize("AI agents on GCP, not FAIR.", lex) == "A I agents on G C P, not FAIR."


def test_normalize_quotes_and_ellipsis():
    assert normalize("“Yep”... ok") == '"Yep"… ok'


def test_chunk_never_splits_mid_sentence_and_respects_limit():
    sents = [f"This is sentence number {i} with some words." for i in range(30)]
    chunks = chunk(" ".join(sents), max_chars=120)
    assert all(len(c.text) <= 120 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)
    assert " ".join(c.text for c in chunks) == " ".join(sents)


def test_chunk_paragraph_boundaries():
    chunks = chunk("First paragraph here.\n\nSecond one. Still second.")
    assert [c.boundary for c in chunks] == ["paragraph", "end"]


def test_chunk_splits_overlong_sentence_at_soft_breaks():
    s = "; ".join(["clause with several words in it"] * 20) + "."
    chunks = chunk(s, max_chars=100)
    assert len(chunks) > 1 and all(len(c.text) <= 100 for c in chunks)


def _tone(seconds, amp=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_trim_silence_removes_padding():
    wav = np.concatenate([np.zeros(SR), _tone(1.0), np.zeros(SR)])
    out = trim_silence(wav, SR, pad_ms=0)
    assert abs(len(out) / SR - 1.0) < 0.02


def test_normalize_loudness_hits_target():
    out = normalize_loudness(_tone(2.0, amp=0.05), SR, -16.0)
    assert loudness(out, SR) == pytest.approx(-16.0, abs=0.2)


def test_normalize_loudness_reaches_target_on_peaky_signal():
    # Quiet tone with sparse full-scale clicks: plain gain would clip, so the
    # old approach fell short of the target. The limiter must hold both.
    wav = _tone(3.0, amp=0.02)
    wav[:: SR // 4] = 0.99
    out = normalize_loudness(wav, SR, -16.0, peak_dbfs=-1.0)
    assert loudness(out, SR) == pytest.approx(-16.0, abs=0.5)
    assert np.max(np.abs(out)) <= 10 ** (-1.0 / 20) + 1e-4


def test_stitch_inserts_pauses():
    out = stitch([_tone(1.0), _tone(1.0)], ["paragraph", "end"], SR)
    assert len(out) == 2 * SR + int(0.6 * SR)
