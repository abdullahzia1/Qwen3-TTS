from bench.__main__ import _word_errors, summarize


def test_word_errors_counts_sub_ins_del():
    ref = "the box changed the world".split()
    assert _word_errors(ref, ref) == 0
    assert _word_errors(ref, "the box changed a world".split()) == 1  # substitution
    assert _word_errors(ref, "the box changed the whole world".split()) == 1  # insertion
    assert _word_errors(ref, "the box the world".split()) == 1  # deletion
    assert _word_errors(ref, []) == len(ref)


def _item(id_, seed, f0, wer=0.0, words=10, sim=0.9, cat="medium", spike=0.1):
    return {"id": id_, "cat": cat, "seed": seed, "words": words, "errors": round(wer * words), "wer": wer,
            "sim": sim, "f0": f0, "spike": spike, "pace_wps": 2.5, "lufs": -16.0, "rtf": 2.0,
            "chunks": 1, "attempts": 1, "suspect": 0}


def test_summarize_corpus_wer_and_consistency():
    items = [_item("a", 0, 120), _item("a", 1, 120), _item("b", 0, 120, wer=0.5), _item("b", 1, 180, spike=0.5)]
    s = summarize(items + [{"id": "c", "cat": "medium", "seed": 0, "status": "failed", "wer": None}])
    assert s["failed"] == 1
    assert s["wer"] == 5 / 40
    assert s["items_wer_gt_10pct"] == ["b/s0"]
    assert s["spikes_gt_40pct"] == 1
    assert s["f0_cv_across_seeds"] > 0  # item b differs between seeds
