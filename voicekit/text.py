"""Text preparation: deterministic normalization and sentence-aware chunking.

The model takes raw text (no SSML, no phonemes), so everything that affects
pronunciation has to be decided here, the same way every time.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")
_SOFT_BREAK = re.compile(r"(?<=[;:—–,])\s+")


@dataclass
class Chunk:
    text: str
    # Pause inserted after this chunk when stitching: "sentence", "paragraph" or "end".
    boundary: str


def normalize(text: str, lexicon: Optional[Dict[str, str]] = None) -> str:
    """Make the text deterministic before it reaches the model.

    `lexicon` maps a word or phrase to its respelling ("GCP" -> "G C P").
    Matching is whole-word and case-sensitive, so "Qwen" and "qwen" can differ.
    """
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[“”]", '"', t)
    t = re.sub(r"[‘’]", "'", t)
    t = re.sub(r"\.{3,}", "…", t)
    if lexicon:
        # Longest keys first so "AI agents" wins over "AI".
        for key in sorted(lexicon, key=len, reverse=True):
            t = re.sub(rf"(?<!\w){re.escape(key)}(?!\w)", lexicon[key], t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _split_long(sentence: str, max_chars: int) -> List[str]:
    """Split one over-long sentence at soft punctuation, then at spaces."""
    if len(sentence) <= max_chars:
        return [sentence]
    parts, cur = [], ""
    for piece in _SOFT_BREAK.split(sentence):
        if cur and len(cur) + 1 + len(piece) > max_chars:
            parts.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}" if cur else piece
    if cur:
        parts.append(cur)
    out = []
    for p in parts:
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


def chunk(text: str, max_chars: int = 400, min_chars: int = 80) -> List[Chunk]:
    """Split normalized text into chunks that never break mid-sentence.

    Sentences are packed greedily up to `max_chars`. A paragraph never shares a
    chunk with the next one, so paragraph pauses land exactly where the script
    has them. A trailing chunk shorter than `min_chars` is merged backwards when
    it fits, since very short inputs are where the model is least stable.
    """
    chunks: List[Chunk] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for p_idx, para in enumerate(paragraphs):
        para = re.sub(r"\s*\n\s*", " ", para)
        sentences = []
        for s in _SENTENCE_END.split(para):
            if s.strip():
                sentences.extend(_split_long(s.strip(), max_chars))

        packed: List[str] = []
        for s in sentences:
            if packed and len(packed[-1]) + 1 + len(s) <= max_chars:
                packed[-1] = f"{packed[-1]} {s}"
            else:
                packed.append(s)
        if len(packed) > 1 and len(packed[-1]) < min_chars and len(packed[-2]) + 1 + len(packed[-1]) <= max_chars:
            packed[-2] = f"{packed[-2]} {packed.pop()}"

        last_para = p_idx == len(paragraphs) - 1
        for i, s in enumerate(packed):
            if i < len(packed) - 1:
                boundary = "sentence"
            else:
                boundary = "end" if last_para else "paragraph"
            chunks.append(Chunk(text=s, boundary=boundary))
    return chunks
