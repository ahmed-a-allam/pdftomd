"""
quran.py
========
Quran correction layer: detects Quranic verses quoted in the extracted text
(even when garbled by OCR) and replaces them with verified Uthmani text from a
local database.

Idea: no OCR reads the Uthmani script with 100% accuracy, but the Quran text is
fixed and known. So instead of fixing characters, we fuzzy-match the garbled
text (on a diacritics-stripped form) against the nearest verse and substitute
the correct one.

Requires:
  - data/quran-uthmani.json  (created by setup_quran.py)
  - rapidfuzz  (pip install rapidfuzz)

Usage:
  qc = QuranCorrector()
  if qc.ready:
      md = qc.correct(md)
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pdf2md.quran")

DEFAULT_CORPUS = Path(__file__).with_name("data") / "quran-uthmani.json"

# Diacritics / Quranic marks to strip before matching.
# Built from explicit codepoint ranges (ASCII source) so it can NEVER
# accidentally include the Arabic letter block U+0621..U+064A.
_DIACRITIC_RANGES = [
    (0x0610, 0x061A),   # Arabic signs (before the letters block)
    (0x064B, 0x065F),   # harakat / tashkeel (after the letters block)
    (0x0670, 0x0670),   # superscript alef
    (0x06D6, 0x06DC),   # Quranic annotation marks
    (0x06DF, 0x06E8),
    (0x06EA, 0x06ED),
    (0x0640, 0x0640),   # tatweel
]
_DIACRITICS = re.compile(
    "[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _DIACRITIC_RANGES) + "]"
)

# Letter-shape unification (alef/ya/waw/ta-marbuta variants)
_NORM_MAP = {
    0x0623: "ا", 0x0625: "ا", 0x0622: "ا", 0x0671: "ا",  # أ إ آ ٱ -> ا
    0x0649: "ي", 0x0626: "ي",                              # ى ئ -> ي
    0x0624: "و",                                            # ؤ -> و
    0x0629: "ه",                                            # ة -> ه
}

# Keep only Arabic block (U+0600..U+06FF) + whitespace
_KEEP_ARABIC = re.compile(r"[^؀-ۿ\s]")

# Candidate verse span. Verses are almost always quoted, but the quoting style
# depends on who produced the text: older OCR gave us ( ) and ﴿ ﴾, while the
# vision models quote with Arabic guillemets « » or curly quotes. Missing those
# meant every verse in a QARI-produced file was skipped, so all four are matched.
_SPAN = re.compile(
    r"\(\s*(?P<paren>[^()\n]{6,400}?)\s*\)"
    r"|﴿\s*(?P<orn>[^﴿﴾\n]{6,400}?)\s*﴾"
    r"|«\s*(?P<guill>[^«»\n]{6,400}?)\s*»"
    r"|“\s*(?P<curly>[^“”\n]{6,400}?)\s*”"
)
_SPAN_GROUPS = ("paren", "orn", "guill", "curly")

# Line prefixes that are structural and never treated as verses.
_SKIP_PREFIXES = ("#", "-", "*", ">", "|", "[^", "```", "---")


def normalize_ar(text: str) -> str:
    """Arabic normalization: strip diacritics/tatweel, unify letters, collapse spaces."""
    text = unicodedata.normalize("NFC", text)
    text = _DIACRITICS.sub("", text)
    text = text.translate(_NORM_MAP)
    text = _KEEP_ARABIC.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


class QuranCorrector:
    def __init__(
        self,
        corpus_path: Optional[Path] = None,
        threshold: int = 68,
        margin: int = 5,
        wrap: bool = True,
        add_reference: bool = True,
        min_words: int = 4,
    ):
        # threshold = minimum similarity of the best match.
        # margin    = how far the best match must beat the 2nd-best (guards
        #             against ambiguous matches on common words).
        self.threshold = threshold
        self.margin = margin
        self.wrap = wrap
        self.add_reference = add_reference
        self.min_words = min_words
        self.ready = False
        self._verses: list[dict] = []
        self._norm: list[str] = []
        self._load(corpus_path or DEFAULT_CORPUS)

    def _load(self, path: Path):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not load Quran corpus (%s). Layer disabled. "
                           "Run setup_quran.py first.", exc)
            return
        try:
            from rapidfuzz import process, fuzz  # noqa: F401
        except Exception:
            logger.warning("rapidfuzz not installed - Quran layer disabled. "
                           "pip install rapidfuzz")
            return
        self._verses = data
        self._norm = [normalize_ar(v["text"]) for v in data]
        self.ready = bool(self._verses)
        logger.info("Quran correction layer ready (%d verses).", len(self._verses))

    def _format(self, verse: dict) -> str:
        text = verse["text"]
        out = f"﴿ {text} ﴾" if self.wrap else text
        if self.add_reference:
            out += f" [Quran {verse['surah']}:{verse['ayah']}]"
        return out

    def _best(self, norm: str):
        """Return (idx, score, verse_word_count) for the best verse match, using
        a top-2 margin check to reject ambiguous matches."""
        from rapidfuzz import process, fuzz
        cand = process.extract(norm, self._norm, scorer=fuzz.token_set_ratio, limit=2)
        if not cand:
            return None
        best = cand[0]
        second = cand[1][1] if len(cand) > 1 else 0.0
        score, idx = best[1], best[2]
        if score < self.threshold or (score - second) < self.margin:
            return None
        return idx, score, len(self._norm[idx].split())

    def _try(self, fragment: str, min_ratio: float):
        """Match a fragment; return formatted verse or None."""
        norm = normalize_ar(fragment)
        words = norm.split()
        if len(words) < self.min_words:
            return None
        res = self._best(norm)
        if not res:
            return None
        idx, score, vlen = res
        ratio = min(len(words), vlen) / max(len(words), vlen)
        if ratio < min_ratio:
            return None
        return self._format(self._verses[idx])

    def _cite_partial(self, fragment: str, min_score: int = 92):
        """For a PARTIAL quote (a few words lifted out of a long verse), the
        length guard in `_try` correctly refuses to swap in the whole verse —
        that would put text on the page the author never quoted.

        But we can still verify it and add the citation. This looks for the
        fragment *inside* a verse and, when the match is near-exact, returns
        "surah:ayah" so the caller can append a reference and leave the author's
        wording untouched.
        """
        from rapidfuzz import process, fuzz
        norm = normalize_ar(fragment)
        words = norm.split()
        # Needs to be long enough that a coincidental match is implausible.
        if len(words) < max(5, self.min_words):
            return None
        cand = process.extract(norm, self._norm, scorer=fuzz.partial_ratio,
                               limit=2)
        if not cand:
            return None
        best = cand[0]
        second = cand[1][1] if len(cand) > 1 else 0.0
        if best[1] < min_score or (best[1] - second) < 2:
            return None
        v = self._verses[best[2]]
        return f"{v['surah']}:{v['ayah']}"

    def correct(self, text: str) -> str:
        """
        Replace Quranic quotes with verified Uthmani text.
        Pass 1: spans inside ( ) or ﴿ ﴾ (verses are usually quoted, even inline).
        Pass 2: standalone verse lines not caught by Pass 1.
        """
        if not self.ready:
            return text

        # Pass 1: quoted spans in any of the four styles (handles inline verses)
        def _span_repl(m):
            inner = next((m.group(g) for g in _SPAN_GROUPS
                          if m.group(g) is not None), None)
            if inner is None:
                return m.group(0)
            out = self._try(inner, min_ratio=0.5)
            if out:
                return out
            # Partial quote: keep the author's wording, just add the reference.
            if self.add_reference:
                ref = self._cite_partial(inner)
                if ref:
                    return f"{m.group(0)} [Quran {ref}]"
            return m.group(0)

        text = _SPAN.sub(_span_repl, text)

        # Pass 2: standalone verse lines (skip already-corrected or structural lines)
        out_lines: list[str] = []
        for line in text.split("\n"):
            if "﴿" in line or line.lstrip().startswith(_SKIP_PREFIXES):
                out_lines.append(line)
                continue
            out = self._try(line, min_ratio=0.6)
            out_lines.append(out if out else line)
        return "\n".join(out_lines)
