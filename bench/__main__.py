"""TTS benchmark: generate through the live API, score with independent judges, compare runs.

    # 1. generate (server must be running; uses the production code path)
    python -m bench run --voice v2/v2 --seeds 0,1,2 --label baseline
    # 2. score (stop the server's model first if memory is tight)
    python -m bench score bench/runs/baseline
    # 3. compare two scored runs
    python -m bench compare bench/runs/baseline bench/runs/temp07

Judges (override with env to use local copies; BENCH_DEVICE=cpu keeps the GPU free for generation):
    BENCH_ASR_MODEL  default openai/whisper-large-v3-turbo   -> word error rate
    BENCH_SV_MODEL   default microsoft/wavlm-base-plus-sv    -> speaker similarity
They are deliberately not Qwen components, so the TTS model is never grading itself.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent


# --- generation -------------------------------------------------------------
def cmd_run(args) -> int:
    import httpx

    items = [json.loads(line) for line in Path(args.set).read_text().splitlines() if line.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out) / args.label
    (out / "audio").mkdir(parents=True, exist_ok=True)
    c = httpx.Client(base_url=args.api, timeout=60)
    params = json.loads(args.params)

    pending = {}
    for seed in seeds:
        for it in items:
            body = {"voice": args.voice, "text": it["text"], "seed": seed, "params": params, "tag": "bench"}
            if args.max_chars:
                body["max_chars"] = args.max_chars
            r = c.post("/v1/tts/jobs", json=body)
            r.raise_for_status()
            pending[r.json()["id"]] = (it, seed)
    print(f"[run] queued {len(pending)} jobs ({len(items)} items x {len(seeds)} seeds)", flush=True)

    records, t0 = [], time.time()
    while pending:
        for jid in list(pending):
            j = c.get(f"/v1/jobs/{jid}").json()
            if j["status"] not in ("succeeded", "partial", "failed"):
                continue
            it, seed = pending.pop(jid)
            rec = {"id": it["id"], "cat": it["cat"], "text": it["text"], "seed": seed, "job": jid, "status": j["status"]}
            if j["status"] == "failed":
                rec["error"] = j["error"]
            else:
                wav = out / "audio" / f"{it['id']}__s{seed}.wav"
                wav.write_bytes(c.get(f"/v1/tts/jobs/{jid}/audio").content)
                rec.update(wav=str(wav.relative_to(out)), result=j["result"])
            records.append(rec)
            done = len(records)
            print(f"[run] {done}/{done + len(pending)}  {it['id']} s{seed}: {j['status']}  ({time.time() - t0:.0f}s)", flush=True)
        time.sleep(1)

    voice = c.get("/v1/voices").json()
    name, _, ver = records[0]["result"]["meta"]["voice"].partition("/v") if records and "result" in records[0] else (args.voice, "", "")
    meta = {"label": args.label, "voice": args.voice, "params": params, "seeds": seeds, "set": args.set,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": round(time.time() - t0, 1),
            "voice_resolved": f"{name}/v{ver}" if ver else args.voice}
    (out / "run.json").write_text(json.dumps({"meta": meta, "records": records}, indent=2) + "\n")
    print(f"[run] wrote {out / 'run.json'}")
    return 0


# --- scoring ----------------------------------------------------------------
def _word_errors(ref: List[str], hyp: List[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)]


class Judges:
    def __init__(self):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector, pipeline

        device = os.environ.get("BENCH_DEVICE") or (
            "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        asr = os.environ.get("BENCH_ASR_MODEL", "openai/whisper-large-v3-turbo")
        sv = os.environ.get("BENCH_SV_MODEL", "microsoft/wavlm-base-plus-sv")
        print(f"[score] judges: ASR={asr}  SV={sv}  device={device}", flush=True)
        self.asr = pipeline("automatic-speech-recognition", model=asr, torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                            device=device)
        import re

        # Whisper writes spoken times as "3.30" or "3:30" interchangeably; both
        # are the same words, so unify before scoring.
        self.normalize = lambda t: self.asr.tokenizer.normalize(re.sub(r"(\d)[:.](\d\d)\b", r"\1 \2", t))
        self.sv_fe = AutoFeatureExtractor.from_pretrained(sv)
        self.sv = WavLMForXVector.from_pretrained(sv).eval()
        self.torch = torch

    def transcribe(self, wav: np.ndarray, sr: int) -> str:
        kw = {"generate_kwargs": {"language": "english", "task": "transcribe"}}
        if len(wav) / sr > 28:
            # Sequential long-form decoding. The pipeline's chunk_length_s=30
            # mode was measured dropping a ~15 s passage at a window boundary,
            # which would score as false word errors.
            kw["return_timestamps"] = True
        return self.asr({"raw": wav, "sampling_rate": sr}, **kw)["text"].strip()

    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        import librosa

        x = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
        feats = self.sv_fe(x, sampling_rate=16000, return_tensors="pt")
        with self.torch.no_grad():
            e = self.sv(**feats).embeddings[0].numpy()
        return e / (np.linalg.norm(e) + 1e-9)


def cmd_score(args) -> int:
    import soundfile as sf

    sys.path.insert(0, str(HERE.parent))
    from voicekit.audio import loudness, pitch_profile, trim_silence

    run_dir = Path(args.run)
    run = json.loads((run_dir / "run.json").read_text())
    j = Judges()
    voices_root = Path(os.environ.get("VOICEKIT_DATA", "data")) / "voices"
    ref_cache: Dict[str, Any] = {}

    scored = []
    for rec in run["records"]:
        if rec["status"] == "failed":
            scored.append({**rec, "result": None})
            continue
        vid = rec["result"]["meta"]["voice"]
        if vid not in ref_cache:
            name, _, ver = vid.partition("/v")
            ref, rsr = sf.read(voices_root / name / f"v{ver}" / "ref.wav", dtype="float32")
            ref_cache[vid] = (j.embed(ref, rsr), pitch_profile(ref, rsr)[0])
        ref_emb, ref_f0 = ref_cache[vid]

        wav, sr = sf.read(run_dir / rec["wav"], dtype="float32")
        hyp = j.transcribe(wav, sr)
        rw, hw = j.normalize(rec["text"]).split(), j.normalize(hyp).split()
        errs = _word_errors(rw, hw)
        f0, wins = pitch_profile(wav, sr)
        speech = trim_silence(wav, sr)
        m = rec["result"]["meta"]
        s = {
            "id": rec["id"], "cat": rec["cat"], "seed": rec["seed"], "status": rec["status"],
            "words": len(rw), "errors": errs, "wer": errs / max(1, len(rw)), "hyp": hyp,
            "sim": float(np.dot(j.embed(wav, sr), ref_emb)),
            "f0": f0 if np.isfinite(f0) else None,
            "f0_vs_ref": (f0 / ref_f0 - 1) if np.isfinite(f0) and ref_f0 else None,
            "spike": max((abs(w / ref_f0 - 1) for w in wins), default=0.0) if ref_f0 else None,
            "pace_wps": len(rw) / max(0.1, len(speech) / sr),
            "lufs": loudness(wav, sr), "duration_s": len(wav) / sr, "rtf": m.get("rtf"),
            "chunks": len(rec["result"]["chunks"]),
            "attempts": sum(c["attempts"] for c in rec["result"]["chunks"]),
            "suspect": len(m.get("suspect_chunks", [])),
        }
        scored.append(s)
        flag = "  <-- check" if s["wer"] > 0.1 or s["sim"] < 0.8 else ""
        print(f"[score] {s['id']:9} s{s['seed']}  WER {s['wer']:5.1%}  SIM {s['sim']:.3f}  F0 {s['f0'] or 0:5.0f}  "
              f"pace {s['pace_wps']:.2f} w/s{flag}", flush=True)
        if s["wer"] > 0.1:
            print(f"          ref: {rec['text']}\n          hyp: {hyp}", flush=True)

    summary = summarize(scored)
    (run_dir / "scores.json").write_text(json.dumps({"meta": run["meta"], "summary": summary, "items": scored}, indent=2, default=float) + "\n")
    (run_dir / "summary.md").write_text(render(run["meta"], summary) + "\n")
    print("\n" + render(run["meta"], summary))
    return 0


# Speaker-similarity and pitch need about a second of speech; on shorter clips
# both judges were measured to be unreliable (0% voiced, similarity 0.76 on
# takes that ASR heard perfectly), so those clips only count toward WER.
MIN_ACOUSTIC_S = 1.0


def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [i for i in items if i.get("wer") is not None]
    acoustic = [i for i in ok if i.get("duration_s", MIN_ACOUSTIC_S) >= MIN_ACOUSTIC_S]
    cv = lambda xs: float(np.std(xs) / np.mean(xs)) if len(xs) > 1 and np.mean(xs) else None  # noqa: E731
    f0s = [i["f0"] for i in acoustic if i["f0"]]
    paces = [i["pace_wps"] for i in ok if i["cat"] != "short"]
    by_cat: Dict[str, List] = {}
    for i in ok:
        by_cat.setdefault(i["cat"], []).append(i)
    # Seed-to-seed variation per item: how differently the same line comes out.
    by_item: Dict[str, List] = {}
    for i in acoustic:
        by_item.setdefault(i["id"], []).append(i)
    seed_f0_cv = [cv([x["f0"] for x in v if x["f0"]]) for v in by_item.values() if len(v) > 1]
    seed_f0_cv = [x for x in seed_f0_cv if x is not None]
    return {
        "n": len(items), "failed": len(items) - len(ok),
        "wer": sum(i["errors"] for i in ok) / max(1, sum(i["words"] for i in ok)),
        "items_wer_gt_10pct": sorted({f"{i['id']}/s{i['seed']}" for i in ok if i["wer"] > 0.1}),
        "wer_by_cat": {k: sum(i["errors"] for i in v) / max(1, sum(i["words"] for i in v)) for k, v in sorted(by_cat.items())},
        "acoustic_excluded_short": len(ok) - len(acoustic),
        "sim_mean": float(np.mean([i["sim"] for i in acoustic])) if acoustic else None,
        "sim_min": float(np.min([i["sim"] for i in acoustic])) if acoustic else None,
        "sim_std": float(np.std([i["sim"] for i in acoustic])) if acoustic else None,
        "f0_median": float(np.median(f0s)) if f0s else None,
        "f0_cv_across_items": cv(f0s),
        "f0_cv_across_seeds": float(np.mean(seed_f0_cv)) if seed_f0_cv else None,
        "spike_max": max((i["spike"] or 0) for i in acoustic) if acoustic else None,
        "spikes_gt_40pct": sum(1 for i in acoustic if (i["spike"] or 0) > 0.4),
        "pace_mean_wps": float(np.mean(paces)) if paces else None,
        "pace_cv": cv(paces),
        "lufs_std": float(np.std([i["lufs"] for i in ok if np.isfinite(i["lufs"])])) if ok else None,
        "rtf_mean": float(np.mean([i["rtf"] for i in ok if i["rtf"]])) if ok else None,
        "retry_rate": sum(i["attempts"] - i["chunks"] for i in ok) / max(1, sum(i["chunks"] for i in ok)),
        "suspect_chunks": sum(i["suspect"] for i in ok),
    }


ROWS = [
    ("wer", "Word error rate", "{:.1%}", -1), ("sim_mean", "Speaker similarity (mean)", "{:.3f}", 1),
    ("sim_min", "Speaker similarity (worst)", "{:.3f}", 1), ("f0_cv_across_items", "Pitch variation across lines (CV)", "{:.1%}", -1),
    ("f0_cv_across_seeds", "Pitch variation across takes (CV)", "{:.1%}", -1), ("spikes_gt_40pct", "Lines with >40% pitch spike", "{}", -1),
    ("pace_cv", "Pace variation (CV)", "{:.1%}", -1), ("pace_mean_wps", "Pace (words/s)", "{:.2f}", 0),
    ("lufs_std", "Loudness spread (LU)", "{:.2f}", -1), ("retry_rate", "Retry rate per chunk", "{:.1%}", -1),
    ("suspect_chunks", "Chunks left suspect", "{}", -1), ("rtf_mean", "Speed (RTF, lower=faster)", "{:.2f}", -1),
    ("failed", "Failed jobs", "{}", -1), ("acoustic_excluded_short", "Clips < 1 s (WER only)", "{}", 0),
]


def _fmt(fmt, v):
    return "—" if v is None else fmt.format(v)


def render(meta, s) -> str:
    lines = [f"## {meta['label']} — {meta['voice']}  params={json.dumps(meta['params'])}  seeds={meta['seeds']}",
             "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {name} | {_fmt(fmt, s.get(k))} |" for k, name, fmt, _ in ROWS]
    lines += ["", "WER by category: " + ", ".join(f"{k} {v:.1%}" for k, v in s["wer_by_cat"].items())]
    if s["items_wer_gt_10pct"]:
        lines.append("Lines with WER > 10%: " + ", ".join(s["items_wer_gt_10pct"]))
    return "\n".join(lines)


def cmd_compare(args) -> int:
    runs = [json.loads((Path(r) / "scores.json").read_text()) for r in args.runs]
    head = "| Metric | " + " | ".join(r["meta"]["label"] for r in runs) + " |"
    print(head + "\n|---|" + "---|" * len(runs))
    for k, name, fmt, better in ROWS:
        vals = [r["summary"].get(k) for r in runs]
        cells = []
        for i, v in enumerate(vals):
            cell = _fmt(fmt, v)
            if i > 0 and v is not None and vals[0] is not None and better and v != vals[0]:
                cell += " ✓" if (v - vals[0]) * better > 0 else " ✗"
            cells.append(cell)
        print(f"| {name} | " + " | ".join(cells) + " |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--voice", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--seeds", default="0,1,2")
    r.add_argument("--params", default="{}")
    r.add_argument("--max-chars", type=int)
    r.add_argument("--set", default=str(HERE / "tts_set.jsonl"))
    r.add_argument("--out", default=str(HERE / "runs"))
    r.add_argument("--api", default=os.environ.get("VOICEKIT_API", "http://127.0.0.1:8100"))
    s = sub.add_parser("score")
    s.add_argument("run")
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+")
    args = ap.parse_args()
    return {"run": cmd_run, "score": cmd_score, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
