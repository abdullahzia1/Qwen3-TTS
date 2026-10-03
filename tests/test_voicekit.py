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


def _glide(seconds, f0):
    """Voiced-ish test signal: a harmonic tone at f0."""
    phase = 2 * np.pi * f0 * np.arange(int(SR * seconds)) / SR
    return (0.3 * (np.sin(phase) + 0.5 * np.sin(2 * phase))).astype(np.float32)


def _voice(f0):
    from types import SimpleNamespace

    return SimpleNamespace(f0_median_hz=f0)


def test_pitch_profile_tracks_f0():
    from voicekit.audio import pitch_profile

    med, wins = pitch_profile(_glide(4.0, 130), SR)
    assert med == pytest.approx(130, rel=0.03)
    assert len(wins) >= 4  # 2 s windows sliding by 0.5 s


def test_inspect_flags_pitch_spike_but_not_steady_take():
    from voicekit.synth import _inspect

    steady = _glide(6.0, 128)
    issue, f0, _ = _inspect(steady, SR, 6.0, _voice(128))
    assert issue is None and f0 == pytest.approx(128, rel=0.03)

    # +55% opening on an otherwise normal line, like the chunk users heard.
    jumpy = np.concatenate([_glide(2.5, 198), _glide(6.0, 128)])
    issue, _, _ = _inspect(jumpy, SR, 8.5, _voice(128))
    assert issue and issue.startswith("pitch spike +"), issue


def test_inspect_flags_whole_line_in_wrong_register():
    from voicekit.synth import _inspect

    issue, _, _ = _inspect(_glide(4.0, 175), SR, 4.0, _voice(128))
    assert issue == "pitch +37% vs voice"


def test_inspect_skips_pitch_checks_on_sub_second_speech():
    from voicekit.synth import _inspect

    issue, f0, _ = _inspect(_glide(0.7, 250), SR, 0.7, _voice(128))  # would be +95%
    assert issue is None and f0 is None


def test_inspect_flags_chunk_off_from_job_median():
    from voicekit.synth import _inspect

    issue, _, _ = _inspect(_glide(6.0, 150), SR, 6.0, _voice(150), target_f0=120)
    assert issue == "pitch +25% vs other chunks"


def test_inspect_flags_length():
    from voicekit.synth import _inspect

    issue, _, _ = _inspect(_glide(1.0, 128), SR, 6.0, _voice(128))
    assert issue.startswith("too short")
