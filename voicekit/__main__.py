"""voicekit CLI.

    # 1. Turn a reference clip into a locked voice version (Base model):
    python -m voicekit make-voice --name narrator --ref clip.wav --ref-text "exact transcript"

    # 2. Render a script with it:
    python -m voicekit say --voice narrator --text-file script.txt --out out.wav

Model paths come from --model or QWEN_BASE_MODEL; device/dtype from
QWEN_DEVICE / QWEN_DTYPE (defaults: best available device, bfloat16).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf


def _load_model(path: str):
    import torch

    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.device_utils import get_attention_implementation, get_optimal_device

    device = os.environ.get("QWEN_DEVICE") or get_optimal_device()
    dtype = getattr(torch, os.environ.get("QWEN_DTYPE", "bfloat16"))
    kwargs = {"device_map": device, "dtype": dtype}
    attn = get_attention_implementation(device)
    if attn:
        kwargs["attn_implementation"] = attn
    print(f"[info] loading {path} on {device} ({dtype})", file=sys.stderr)
    return Qwen3TTSModel.from_pretrained(path, **kwargs)


def _model_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=os.environ.get("QWEN_BASE_MODEL"), help="Base model dir or HF id (or QWEN_BASE_MODEL)")
    p.add_argument("--voices", default=os.environ.get("VOICEKIT_VOICES", "voices"), help="Voice library root")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="voicekit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    mv = sub.add_parser("make-voice", help="Create a new voice version from a reference clip")
    _model_arg(mv)
    mv.add_argument("--name", required=True)
    mv.add_argument("--ref", required=True, help="Reference audio (8-15s, clean, single speaker)")
    mv.add_argument("--ref-text", required=True, help="Exact transcript of the reference")
    mv.add_argument("--language", default="English")
    mv.add_argument("--xvec", action="store_true", help="Speaker-embedding-only mode (no transcript needed, lower quality)")
    mv.add_argument("--params", default="{}", help="JSON overrides for default generation params")
    mv.add_argument("--notes", default="")

    say = sub.add_parser("say", help="Render text with a saved voice")
    _model_arg(say)
    say.add_argument("--voice", required=True, help="name or name/vN")
    src = say.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--text-file")
    say.add_argument("--out", required=True)
    say.add_argument("--seed", type=int, default=0)
    say.add_argument("--params", default="{}", help="JSON overrides for this render only")
    say.add_argument("--lexicon", help="JSON file mapping words to respellings")
    say.add_argument("--max-chars", type=int, default=400)
    say.add_argument("--batch-size", type=int, default=1)
    say.add_argument("--lufs", type=float, default=-16.0)

    args = ap.parse_args(argv)
    if not args.model:
        ap.error("--model or QWEN_BASE_MODEL is required")

    from .synth import synthesize
    from .voice import create_voice, load_voice

    tts = _load_model(args.model)

    if args.cmd == "make-voice":
        v = create_voice(
            tts, args.voices, args.name, args.ref, args.ref_text, language=args.language,
            x_vector_only=args.xvec, params=json.loads(args.params), notes=args.notes,
        )
        print(f"[ok] created {v.id} at {v.path} ({v.meta['ref_duration_s']}s ref, {v.chars_per_sec} chars/s)")
        return 0

    name, _, ver = args.voice.partition("/v")
    voice = load_voice(args.voices, name, int(ver) if ver else None)
    text = args.text if args.text is not None else Path(args.text_file).read_text(encoding="utf-8")
    lexicon = json.loads(Path(args.lexicon).read_text()) if args.lexicon else None

    res = synthesize(
        tts, voice, text, seed=args.seed, params=json.loads(args.params), lexicon=lexicon,
        max_chars=args.max_chars, batch_size=args.batch_size, target_lufs=args.lufs,
    )
    out = Path(args.out)
    sf.write(out, res.wav, res.sr, subtype="PCM_16")
    out.with_suffix(".json").write_text(json.dumps(res.to_dict(), indent=2) + "\n")
    m = res.meta
    print(f"[ok] {out}  {m['duration_s']}s  RTF {m['rtf']}  {m['lufs']} LUFS  "
          f"{len(res.chunks)} chunks  retried={m['retried_chunks']}  suspect={m['suspect_chunks']}")
    return 1 if m["suspect_chunks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
