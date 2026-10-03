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


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ((10**9, "billion"), (10**6, "million"), (1000, "thousand"))


def number_words(n: int) -> str:
    """0 <= n < 10**12 in words: 5 -> 'five', 1256 -> 'one thousand two hundred fifty-six'."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (" " + number_words(n % 100) if n % 100 else "")
    for size, name in _SCALES:
        if n >= size:
            return number_words(n // size) + " " + name + (" " + number_words(n % size) if n % size else "")
    raise ValueError(n)


_MONEY = re.compile(r"\$(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?!\d|,\d)(\s*(?:billion|million|thousand)\b)?")
_PERCENT = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s?%")


def _decimal_words(whole: int, frac: str) -> str:
    return number_words(whole) + (" point " + " ".join(_ONES[int(d)] for d in frac) if frac else "")


def _money(m: "re.Match") -> str:
    raw_dollars, frac, scale = m.group(1), m.group(2) or "", m.group(3)
    dollars = int(raw_dollars.replace(",", ""))
    if scale:  # "$2.5 billion" -> "two point five billion dollars"
        return f"{_decimal_words(dollars, frac)} {scale.strip()} dollars"
    if frac and len(frac) != 2:
        return m.group(0)  # not a currency amount we understand; leave it to the model
    parts = []
    if dollars or not frac:
        parts.append(f"{number_words(dollars)} dollar{'s' if dollars != 1 else ''}")
    if frac and int(frac):
        parts.append(f"{number_words(int(frac))} cent{'s' if int(frac) != 1 else ''}")
    return " and ".join(parts) or "zero dollars"


def _percent(m: "re.Match") -> str:
    whole, _, frac = m.group(1).partition(".")
    return _decimal_words(int(whole), frac) + " percent"


def expand_numbers(text: str) -> str:
    """Spell out money and percentages so every take reads the same words.

    Measured on the benchmark: "$0.16" was read as "16 cents" in some takes and
    "$0.16" (zero point one six dollars) in others, and "$5.86" was once slurred.
    Plain numbers, years and times are left to the model, which reads them
    consistently.
    """
    text = _MONEY.sub(_money, text)
    return _PERCENT.sub(_percent, text)


def normalize(text: str, lexicon: Optional[Dict[str, str]] = None, expand: bool = True) -> str:
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
        # Lexicon first, so a user respelling always wins over automatic expansion.
        # Longest keys first so "AI agents" wins over "AI".
        for key in sorted(lexicon, key=len, reverse=True):
            t = re.sub(rf"(?<!\w){re.escape(key)}(?!\w)", lexicon[key], t)
    if expand:
        t = expand_numbers(t)
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
