"""
engine.py
=========
PDF -> Markdown conversion logic.

Modular design:
  - Primary engine: Marker (Surya) — layout-aware, auto-detects per page whether
    it is digital text (native extraction) or scanned image (OCR on GPU).
  - Alternative OCR: Windows OCR (Windows.Media.Ocr) — better for printed Arabic.
  - Fallback: PyMuPDF for digital text + EasyOCR for scanned pages.
  - Post-processing layer after extraction:
      * Unicode normalization (NFC) + religious-symbol fixes (e.g. ﷺ).
      * De-hyphenation of words split across line breaks.
      * Footnote markers -> standard Markdown [^n].
      * Reversed RTL parentheses )2( -> (2).
      * Optional line reflow.
  - Optional Quran correction layer (see quran.py).

Exports: PdfToMarkdownEngine, EngineConfig, ConversionResult, processing helpers.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("pdf2md")

ProgressCB = Optional[Callable[[int, int, str], None]]


class PauseRequested(Exception):
    """Raised when a stop_event is set mid-conversion; progress is saved."""
    def __init__(self, done: int, total: int):
        self.done, self.total = done, total
        super().__init__(f"Paused at page {done}/{total} (progress saved).")


# --------------------------------------------------------------------------- #
# Engine configuration
# --------------------------------------------------------------------------- #
# The four supported engines, in the order shown to the user.
BACKENDS = ["fast", "arabic", "english", "windows"]

# Old / technical names kept working so existing config.json files and scripts
# don't break after the rename.
BACKEND_ALIASES = {
    "qari": "arabic",
    "hybrid": "fast",
    "hybrid-qari": "fast",
    "marker": "english",
    "gemini": "arabic",     # removed engines -> best remaining equivalent
    "glm": "arabic",
}


@dataclass
class EngineConfig:
    languages: list[str] = field(default_factory=lambda: ["ar", "en"])
    use_gpu: bool = True
    text_threshold_chars: int = 20   # threshold to treat a page as digital text
    output_images: bool = False
    batch_multiplier: int = 1        # 1 suits 4GB VRAM

    # OCR backend — four free, fully-local options:
    #   "arabic"  = QARI-OCR local model. Best quality for classical/diacritized
    #               Arabic. Slowest (a vision LLM runs on every page).
    #   "fast"    = fast local pass (embedded text / Windows OCR) on every page,
    #               then QARI ONLY on the pages that fail a quality check.
    #   "english" = Marker/Surya. Layout-aware; best for English & digital books.
    #   "windows" = Windows OCR alone. Very fast, modern printed Arabic only.
    ocr_backend: str = "fast"

    # Local Arabic OCR model (used by the "arabic" and "fast" engines).
    # "katib" = 0.8B, fits in VRAM as fp16 -> fast. Default.
    # "qari"  = 2B, marginally more accurate but needs slow 4-bit on a 4GB card.
    arabic_model: str = "katib"
    # Speed knobs. Vision-model cost scales with the image's pixel count, so DPI
    # and the pixel cap are the levers that matter for wall-clock time.
    arabic_dpi: int = 200
    arabic_max_pixels: int = 1280 * 28 * 28   # ~1280 visual tokens per page
    arabic_max_new_tokens: int = 0            # 0 = use the model's own default

    # Convert only part of a book (1-based, inclusive). 0 = no limit.
    # A vision model can take minutes per page, so being able to try pages 1-3
    # before committing to a 600-page book matters.
    page_start: int = 0
    page_end: int = 0

    # "fast" mode: fast local pass (native text / Windows OCR) on every page,
    # then the QARI vision model ONLY on pages whose quality score is bad.
    hybrid_threshold: float = 0.10      # 0..1 — pages scoring above this get AI-fixed
    hybrid_native_text_penalty: float = 0.5   # extra suspicion multiplier applied to
        # the threshold when the "fast pass" came from the PDF's own embedded text
        # layer (tag="text"). Old scanned/converted Arabic PDFs often carry a baked-in
        # corrupted text layer (bad font/cmap encoding) that LOOKS like normal words
        # (real letters, just the wrong ones) and slips past arabic_quality_score's
        # symbol-based heuristic. Windows-OCR-sourced text doesn't have this failure
        # mode as often, so it keeps the normal threshold.

    # Quran correction layer (needs data/quran-uthmani.json + rapidfuzz)
    correct_quran: bool = False
    quran_threshold: int = 68        # min similarity of best match (0-100)
    quran_margin: int = 5            # best must beat 2nd-best by this much
    quran_wrap: bool = True          # wrap verse in ﴿ ﴾ + surah:ayah reference

    # Pause/resume: cache each page so a stopped run can continue later
    resume: bool = True

    # Text post-processing toggles
    fix_symbols: bool = True         # Unicode NFC + religious-symbol fixes
    fix_hyphenation: bool = True     # merge words split across line breaks
    fix_parentheses: bool = True     # fix reversed parentheses )2( -> (2)
    fix_footnotes: bool = True       # convert (n) -> [^n]
    strip_footnotes: bool = False    # delete footnotes + reference numbers entirely
    clean_artifacts: bool = True     # remove scan gutter lines / rules / dot leaders
    reflow_lines: bool = False       # merge lines within a paragraph (heuristic)


@dataclass
class ConversionResult:
    source: Path
    output: Path
    engine_used: str
    pages: int
    seconds: float


# --------------------------------------------------------------------------- #
# (1) Religious-symbol fixes + Unicode normalization
# --------------------------------------------------------------------------- #
# Correct Unicode symbols (Presentation Forms-A):
#   ﷺ U+FDFA  |  ﷻ U+FDFB  |  ﷲ U+FDF2  |  ﷴ U+FDF4
#
# Extensible map: add any corrupted sequence you observe in real OCR output.
# key = corrupted text as produced by OCR, value = correct symbol.
RELIGIOUS_SYMBOL_FIXES: dict[str, str] = {
    # Observed OCR misreads of the ﷺ ligature (add more as you see them)
    "لمخي!": "ﷺ",
    "لمج!": "ﷺ",
    "ع!يم": "ﷺ",
    "!و:": "ﷺ",
    # Textual form -> unified ligature (enabled via unify_salla)
    # "صلى الله عليه وسلم": "ﷺ",
}


# --- Context-aware ﷺ recovery ---------------------------------------------- #
# After an honorific antecedent (النبي / الرسول / محمد ...), OCR often turns the
# ﷺ ligature into a short garbage token like "لمخي!", "ع!يم", "!و:".
# We detect that positionally instead of relying on a fixed lookup table.
_HONORIFIC_TRIGGERS = (
    r"(?:النبيّ|النبي|الرسول|رسول\s+الله|المصطفى|نبيّنا|نبينا|نبيه|"
    r"محمدًا|محمدا|محمدٍ|محمد)"
)
# Characters that never belong inside a normal Arabic word
_SUSPECT = set("!:;/\\|~^*_<>=+`؟?@#$%&") | set("0123456789") | set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_QUOTES = set('"\'«»“”')
_ARABIC_RE = re.compile(r"[ء-ي]")
# Real words that legitimately follow these triggers — never treat as garbage
_SAFE_FOLLOWERS = {
    "صلى", "عليه", "قال", "كان", "أنه", "انه", "رضي", "بن", "ابن", "عن",
    "في", "من", "إلى", "الى", "وقال", "وعن", "لما", "حين", "أن", "ان",
    # extra guards for the frequency-based pass below
    "وسلم", "وآله", "واله", "الله", "يقول", "وهو", "هو", "قد", "ثم", "أو",
    "او", "لم", "لا", "ما", "إذا", "اذا", "حتى", "على", "عند", "بعد", "قبل",
    "أبي", "ابي", "أبو", "ابو", "عبد", "سيدنا", "نبينا", "كما", "وكان",
    "وقد", "فقال", "يا", "أيها", "ايها", "به", "له", "بها", "منه",
}

# Diacritics (harakat) — stripped before comparing tokens.
_DIACRITICS_RE = re.compile(
    "[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in [
        (0x0610, 0x061A), (0x064B, 0x065F), (0x0670, 0x0670),
        (0x06D6, 0x06DC), (0x06DF, 0x06E8), (0x06EA, 0x06ED),
        (0x0640, 0x0640),
    ]) + "]"
)


def _bare(token: str) -> str:
    """Token with diacritics and edge punctuation removed."""
    return _DIACRITICS_RE.sub("", token).strip(".,،؛:!؟?\"'«»“”()[]{}")


# A capturing version of the trigger, for the frequency pass.
_AFTER_HONORIFIC = re.compile(rf"({_HONORIFIC_TRIGGERS})(\s+)(\S{{1,16}})")


def find_honorific_corruptions(text: str, min_hits: int = 2) -> dict:
    """Find tokens that are almost certainly a mangled ﷺ.

    Why this exists: `fix_honorific_symbols` only catches garbage containing an
    odd character (!, digits, latin...). Vision models like QARI instead emit a
    *well-formed-looking* Arabic word for the ﷺ ligature (it reads the
    calligraphy as letters), so that check never fires.

    We can't hard-code those words — they differ per model and per book. Instead
    we use the document itself as evidence: a short Arabic token that keeps
    appearing right after "النبي" / "رسول الله" / ... and is not a real word
    that belongs there is, with high confidence, the ligature.

    Returns {token: count} for the tokens judged to be corruptions.
    """
    counts: dict[str, int] = {}
    for m in _AFTER_HONORIFIC.finditer(text):
        tok = m.group(3)
        core = _bare(tok)
        if not core or not _ARABIC_RE.search(core):
            continue
        if len(core) > 6:                 # ﷺ never reads as a long word
            continue
        if core in _SAFE_FOLLOWERS:
            continue
        if core.startswith("ال"):         # adjectives: النبي الكريم / الأعظم
            continue
        if core == "ﷺ":
            continue
        counts[core] = counts.get(core, 0) + 1
    return {t: c for t, c in counts.items() if c >= min_hits}


def fix_honorific_by_frequency(text: str, min_hits: int = 2) -> tuple[str, dict]:
    """Replace repeated post-honorific junk tokens with ﷺ.
    Returns (fixed_text, {token: count}) so the caller can report what changed."""
    bad = find_honorific_corruptions(text, min_hits)
    if not bad:
        return text, {}

    def _repl(m):
        trigger, gap, tok = m.group(1), m.group(2), m.group(3)
        core = _bare(tok)
        if core not in bad:
            return m.group(0)
        # keep any punctuation that trailed the token (e.g. ':' before a quote)
        trail = ""
        stripped = _DIACRITICS_RE.sub("", tok)
        while stripped and stripped[-1] in ':؛،.!؟?"\'«»“”':
            trail = stripped[-1] + trail
            stripped = stripped[:-1]
        return f"{trigger} ﷺ{trail}"

    return _AFTER_HONORIFIC.sub(_repl, text), bad


def fix_honorific_symbols(text: str) -> str:
    """Replace OCR-garbled ﷺ tokens that follow an honorific with the real ﷺ."""
    def _repl(m):
        trigger, gap, token = m.group(1), m.group(2), m.group(3)
        # keep a leading footnote number if the OCR glued one on
        lead = re.match(r"^[\d٠-٩]+", token)
        num = lead.group(0) if lead else ""
        rest = token[len(num):]
        # Cut the garbage core at the first quote / sentence punctuation so that
        # things like  !يم:"لكل داء"  are handled without eating the quote.
        core = re.match(r"[^\s\"'«»“”،.]*", rest).group(0)
        trail = rest[len(core):]
        # trailing ':' or ';' belongs to the sentence, not the symbol
        punct = ""
        while core and core[-1] in ":؛":
            punct = core[-1] + punct
            core = core[:-1]
        if not core or len(core) > 8:
            return m.group(0)
        if not (_SUSPECT & set(core)):               # no odd character -> real word
            return m.group(0)
        arabic = "".join(_ARABIC_RE.findall(core))
        if len(arabic) > 6 or arabic in _SAFE_FOLLOWERS:
            return m.group(0)
        return f"{trigger}{num} ﷺ{punct}{trail}"

    # stray suspect char glued directly before the honorific: "عن!النبي" -> "عن النبي"
    text = re.sub(rf"(?<=[ء-ي])[!:;~^*]+(?={_HONORIFIC_TRIGGERS})", " ", text)
    return re.sub(rf"({_HONORIFIC_TRIGGERS})(\s*)(\S{{2,12}})", _repl, text)


def fix_religious_symbols(text: str, unify_salla: bool = False) -> str:
    """Normalize to Unicode NFC and repair corrupted religious symbols."""
    text = unicodedata.normalize("NFC", text)

    for bad, good in RELIGIOUS_SYMBOL_FIXES.items():
        text = text.replace(bad, good)

    if unify_salla:
        # Unify the textual form "صلى الله عليه وسلم" (flexible spacing) to ﷺ
        text = re.sub(r"صلى\s+الله\s+عليه\s+و?\s*سلم", "ﷺ", text)
    return text


# --------------------------------------------------------------------------- #
# (2) De-hyphenation at line ends
# --------------------------------------------------------------------------- #
def dehyphenate(text: str) -> str:
    """Merge words split by a hyphen at line end (Latin), drop soft hyphens."""
    text = text.replace("­", "")                  # SOFT HYPHEN
    text = re.sub(r"(\w)[-‐]\n(\w)", r"\1\2", text)
    return text


# --------------------------------------------------------------------------- #
# (3a) Fix reversed parentheses in RTL context:  )2(  ->  (2)
# --------------------------------------------------------------------------- #
def fix_rtl_parentheses(text: str) -> str:
    """
    In Arabic (RTL) text, parentheses get mirrored during extraction/OCR,
    producing )2( instead of (2). Fix the common numeric case safely.
    """
    text = re.sub(r"\)(\s*(?:\d{1,3}|[٠-٩]{1,3})\s*)\(", r"(\1)", text)
    return text


# --------------------------------------------------------------------------- #
# (3b) Convert footnotes to standard Markdown
# --------------------------------------------------------------------------- #
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩",
                              "0123456789")


def convert_footnotes(text: str) -> str:
    """
    - Footnote definition at line start:  (1) text...  ->  [^1]: text...
    - Inline marker attached to a word:   word(1)       ->  word[^1]
    Handles both normal and reversed brackets; Arabic-Indic digits normalized.
    """
    num = r"(\d{1,3}|[٠-٩]{1,3})"

    def _def(m):
        return f"[^{m.group(1).translate(_ARABIC_INDIC)}]: "

    def _mark(m):
        return f"[^{m.group(1).translate(_ARABIC_INDIC)}]"

    # Definitions (line starting with a bracketed number) — normal & reversed
    text = re.sub(rf"(?m)^\s*\({num}\)\s*", _def, text)
    text = re.sub(rf"(?m)^\s*\){num}\(\s*", _def, text)
    # Inline markers attached to a word — normal & reversed
    text = re.sub(rf"(?<=\S)\({num}\)", _mark, text)
    text = re.sub(rf"(?<=\S)\){num}\(", _mark, text)
    return text


# --------------------------------------------------------------------------- #
# (4) Line reflow (optional)
# --------------------------------------------------------------------------- #
# Lines starting with these are structural and never merged by reflow.
# "﴿" protects corrected Quran verses so each stays on its own line.
_BLOCK_PREFIXES = ("#", "-", "*", ">", "|", "[^", "```", "﴿")


def reflow_lines(text: str) -> str:
    """
    Merge consecutive prose lines while keeping structural lines
    (headings/lists/tables/footnote-defs/Quran verses) on their own line.
    Works line-by-line so a single protected line no longer blocks the
    whole paragraph from reflowing.
    """
    out_paras: list[str] = []
    for para in text.split("\n\n"):
        result: list[str] = []
        buf: list[str] = []
        for ln in para.split("\n"):
            if ln.lstrip().startswith(_BLOCK_PREFIXES):
                if buf:
                    result.append(" ".join(x.strip() for x in buf if x.strip()))
                    buf = []
                result.append(ln)                       # keep structural line as-is
            else:
                buf.append(ln)                          # accumulate prose
        if buf:
            result.append(" ".join(x.strip() for x in buf if x.strip()))
        out_paras.append("\n".join(result))
    return "\n\n".join(out_paras)


# --------------------------------------------------------------------------- #
# Final cleanup + processing pipeline
# --------------------------------------------------------------------------- #
def clean_page_artifacts(text: str) -> str:
    """Remove marks the OCR picked up from the physical page rather than the
    text: the shadow in a scanned book's centre gutter, rules and borders.

    These come out as stray '|' or repeated punctuation, and once every line
    carries one they stack up into a vertical line down the rendered page.
    Markdown tables are left alone (a real table row has 2+ pipes).
    """
    out = []
    for line in text.split("\n"):
        stripped = line.strip()
        # Leave real Markdown horizontal rules / page separators alone.
        if stripped in ("---", "***", "___", "..."):
            out.append(line)
            continue
        if line.count("|") == 1:                 # lone pipe = gutter, not a table
            line = line.replace("|", " ")
        line = re.sub(r"[│┃┆┇┊┋‖]", " ", line)   # box-drawing / bar characters
        line = re.sub(r"[.ـ]{4,}", " ", line)    # dot leaders / kashida runs
        out.append(re.sub(r"[ \t]{2,}", " ", line).rstrip())
    return "\n".join(out)


# Inline footnote markers produced by convert_footnotes, plus the raw forms that
# survive when footnote conversion is off.
_FN_MARKER = re.compile(r"\[\^[^\]\s]+\]")
_FN_DEF = re.compile(r"^\s*\[\^[^\]\s]+\]:.*$")
_BRACKET_NUM = re.compile(r"\[\s*[\d٠-٩]{1,3}\s*\]")
_PAREN_NUM = re.compile(r"\(\s*[\d٠-٩]{1,3}\s*\)|\)\s*[\d٠-٩]{1,3}\s*\(")


def strip_footnotes(text: str, drop_numbers: bool = True) -> str:
    """Delete footnotes entirely: the definition blocks at the bottom of each
    page and the little reference markers inside the prose.

    A definition runs until the next definition or a blank line, so wrapped
    footnote text is removed too — not just its first line.
    """
    lines = text.split("\n")
    kept: list[str] = []
    in_def = False
    for line in lines:
        if _FN_DEF.match(line):
            in_def = True
            continue
        if in_def:
            # continuation of a footnote definition?
            if line.strip() == "" or line.lstrip().startswith(("#", "---")):
                in_def = False
            else:
                continue
        kept.append(line)

    body = "\n".join(kept)
    body = _FN_MARKER.sub("", body)              # [^3] inside sentences
    if drop_numbers:
        body = _BRACKET_NUM.sub("", body)        # [١٨]
        body = _PAREN_NUM.sub("", body)          # (٤٥٣) and reversed )٤٥٣(
    # tidy the gaps the removals leave behind
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"\s+([،.:؛؟!])", r"\1", body)
    return body


def _whitespace_cleanup(md: str) -> str:
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = "\n".join(line.rstrip() for line in md.splitlines())
    return md.strip() + "\n"


def postprocess(text: str, cfg: EngineConfig, apply_reflow: bool = True) -> str:
    """
    Apply text-processing layers according to the config.

    Reflow is kept separate (apply_reflow) so the Quran correction layer can run
    BETWEEN the rest of the processing and reflow — verses are matched line-by-line
    while still separated, then reflow merges the remaining prose.
    """
    if cfg.clean_artifacts:
        text = clean_page_artifacts(text)
    if cfg.fix_symbols:
        text = fix_religious_symbols(text)
        text = fix_honorific_symbols(text)   # odd-character ﷺ garbage
        # Vision models emit a normal-looking Arabic word for ﷺ, which the check
        # above cannot see. This second pass uses repetition as the evidence.
        text, found = fix_honorific_by_frequency(text)
        if found:
            listed = ", ".join(f"{t}({c})" for t, c in
                               sorted(found.items(), key=lambda kv: -kv[1])[:6])
            print(f"    ﷺ restored from repeated tokens: {listed}", flush=True)
    if cfg.fix_hyphenation:
        text = dehyphenate(text)
    if cfg.fix_parentheses:
        text = fix_rtl_parentheses(text)
    if cfg.strip_footnotes:
        # Remove them outright — do this INSTEAD of converting them.
        text = strip_footnotes(text)
    elif cfg.fix_footnotes:
        text = convert_footnotes(text)
    if apply_reflow and cfg.reflow_lines:
        text = reflow_lines(text)
    return _whitespace_cleanup(text)


def finalize_reflow(text: str, cfg: EngineConfig) -> str:
    """Run reflow as a final step (used after Quran correction)."""
    if cfg.reflow_lines:
        text = reflow_lines(text)
    return _whitespace_cleanup(text)


# --------------------------------------------------------------------------- #
# Arabic OCR quality score (0 = clean, 1 = garbage) — used by hybrid mode
# --------------------------------------------------------------------------- #
_AR_LETTER = re.compile(r"[ء-ي]")
_BAD_TOKEN = re.compile(
    r"!|"                       # stray exclamation inside a word (OCR artifact)
    r"[ء-ي]\d|\d[ء-ي]|"         # digit glued to an Arabic letter
    r"[A-Za-z][ء-ي]|[ء-ي][A-Za-z]"  # Latin glued to Arabic
)


def arabic_quality_score(text: str) -> float:
    """
    Heuristic garbage score for extracted Arabic text.
    Counts suspicious tokens (embedded '!', digits/Latin glued to Arabic,
    reversed bracket pairs) as a fraction of all tokens.
    """
    tokens = [t for t in text.split() if t.strip()]
    if len(tokens) < 5:
        return 0.0                      # too little text to judge
    bad = 0
    for t in tokens:
        if _BAD_TOKEN.search(t):
            bad += 1
        elif len(t) == 1 and _AR_LETTER.search(t) and t not in "وأياب":
            bad += 1                    # isolated single letters (broken words)
    # reversed parentheses pairs are a strong garbage signal
    bad += 2 * len(re.findall(r"\)\s*(?:\d{1,3}|[٠-٩]{1,3})\s*\(", text))
    return min(1.0, bad / len(tokens))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def detect_device(prefer_gpu: bool = True) -> str:
    try:
        import torch

        if prefer_gpu and torch.cuda.is_available():
            logger.info("GPU detected: %s", torch.cuda.get_device_name(0))
            return "cuda"
        if prefer_gpu:
            ver = getattr(torch, "__version__", "?")
            logger.warning("torch cannot see a GPU (version: %s). Possibly CPU-only.", ver)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not check CUDA: %s", exc)
    logger.info("Using CPU (much slower for scanned books).")
    return "cpu"


def count_pages(pdf_path: Path) -> int:
    try:
        import fitz
        with fitz.open(str(pdf_path)) as doc:
            return doc.page_count
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# Per-page checkpoint (pause/resume) — append-only JSONL
# --------------------------------------------------------------------------- #
class _Checkpoint:
    def __init__(self, out_dir: Path, stem: str, backend: str):
        safe = re.sub(r"[^\w.-]", "_", backend)
        self.path = Path(out_dir) / ".cache" / f"{stem}.{safe}.jsonl"

    def load(self) -> dict[int, str]:
        pages: dict[int, str] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    pages[int(d["i"])] = d["t"]
                except Exception:
                    pass
        return pages

    def append(self, i: int, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"i": i, "t": text}, ensure_ascii=False) + "\n")

    def clear(self) -> None:
        try:
            self.path.unlink()
        except Exception:
            pass


def _emit(cb: ProgressCB, i: int, total: int, stage: str) -> None:
    if cb:
        try:
            cb(i, total, stage)
        except Exception:
            pass
    if total:
        print(f"    {stage} {i}/{total}  ({i / total * 100:5.1f}%)", flush=True)
    else:
        print(f"    {stage} {i}", flush=True)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class PdfToMarkdownEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        # Auto-set CUDA memory config (reduces fragmentation on 4GB)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        self.device = detect_device(self.config.use_gpu)
        os.environ.setdefault("TORCH_DEVICE", self.device)
        self._marker_converter = None
        self._quran = None  # lazy-loaded Quran correction layer

    # ---- Quran correction layer (lazy) --------------------------------- #
    def _quran_corrector(self):
        if self._quran is not None:
            return self._quran
        from quran import QuranCorrector
        self._quran = QuranCorrector(
            threshold=self.config.quran_threshold,
            margin=self.config.quran_margin,
            wrap=self.config.quran_wrap,
        )
        return self._quran

    # ---- Page range ------------------------------------------------------ #
    def _range(self, total: int) -> tuple[int, int]:
        """Return the 0-based [lo, hi] page window to convert (inclusive)."""
        last = (total or 10 ** 9) - 1
        lo = max(0, (self.config.page_start or 1) - 1)
        hi = last if not self.config.page_end else min(self.config.page_end - 1, last)
        return lo, max(lo, hi)

    def _assemble(self, parts: dict, lo: int, hi: int) -> str:
        """Join only the pages inside the selected window, in order."""
        return "\n\n---\n\n".join(parts[i] for i in range(lo, hi + 1) if i in parts)

    # ---- Load Marker models once --------------------------------------- #
    def _load_marker(self):
        if self._marker_converter is not None:
            return self._marker_converter
        logger.info("Loading Marker/Surya models (first run may take minutes)...")
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser

        cfg = ConfigParser({
            "output_format": "markdown",
            "languages": ",".join(self.config.languages),
            "extract_images": self.config.output_images,
        })
        self._marker_converter = PdfConverter(
            config=cfg.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=cfg.get_processors(),
            renderer=cfg.get_renderer(),
        )
        return self._marker_converter

    # ---- Primary path: Marker ------------------------------------------ #
    # Marker processes the whole document in one call, so it cannot resume
    # mid-file; stop is only honored before it starts.
    def _convert_with_marker(self, pdf_path, total, cb, ckpt=None, stop=None):
        from marker.output import text_from_rendered

        if stop is not None and stop.is_set():
            raise PauseRequested(0, total)
        if self.config.page_start or self.config.page_end:
            lo, hi = self._range(total)
            print(f"    Note: Marker converts the whole document; the page "
                  f"range {lo + 1}-{hi + 1} is applied after conversion.",
                  flush=True)
        converter = self._load_marker()
        _emit(cb, 0, total, "Marker: starting")
        rendered = converter(str(pdf_path))
        text, _, _ = text_from_rendered(rendered)
        pages = getattr(rendered, "metadata", {}).get("page_count", 0) or total
        _emit(cb, pages or total, total, "Marker: done")
        return text, pages

    # ---- Last-resort path: the PDF's own embedded text layer ------------- #
    # No OCR here on purpose. EasyOCR used to sit in this slot but scored worst
    # of everything we measured on Arabic (CER ~0.65-0.79), while dragging in a
    # large dependency — so it was removed. If a page has no embedded text this
    # path simply leaves it empty rather than filling it with garbage.
    def _convert_with_fallback(self, pdf_path, total, cb, ckpt=None, stop=None):
        import fitz

        done = ckpt.load() if ckpt else {}
        parts: dict[int, str] = dict(done)
        doc = fitz.open(str(pdf_path))
        total = total or doc.page_count
        lo, hi = self._range(doc.page_count)
        n_sel, done_n = hi - lo + 1, 0
        for i in range(lo, hi + 1):
            if i in parts:                                # already done in a prior run
                done_n += 1
                _emit(cb, done_n, n_sel, f"page {i + 1} [cached]")
                continue
            if stop is not None and stop.is_set():
                doc.close()
                raise PauseRequested(done_n, n_sel)
            native = doc[i].get_text("text").strip()
            if len(native) >= self.config.text_threshold_chars:
                text = native
                tag = "page [text]"
            else:
                text = ""
                tag = "page [no text layer - skipped]"
            parts[i] = text
            if ckpt:
                ckpt.append(i, text)
            done_n += 1
            _emit(cb, done_n, n_sel, tag)
        doc.close()
        return self._assemble(parts, lo, hi), n_sel

    # ---- Windows OCR path (best for printed Arabic) -------------------- #
    def _convert_with_windows(self, pdf_path, total, cb, ckpt=None, stop=None):
        import fitz
        from PIL import Image
        from ocr_windows import ocr_image, windows_ocr_available

        ok, msg = windows_ocr_available()
        if not ok:
            raise RuntimeError(f"Windows OCR unavailable: {msg}")

        done = ckpt.load() if ckpt else {}
        parts: dict[int, str] = dict(done)
        doc = fitz.open(str(pdf_path))
        total = total or doc.page_count
        lo, hi = self._range(doc.page_count)
        n_sel, done_n = hi - lo + 1, 0
        for i in range(lo, hi + 1):
            if i in parts:                                # already done in a prior run
                done_n += 1
                _emit(cb, done_n, n_sel, f"page {i + 1} [cached]")
                continue
            if stop is not None and stop.is_set():
                doc.close()
                raise PauseRequested(done_n, n_sel)
            native = doc[i].get_text("text").strip()
            if len(native) >= self.config.text_threshold_chars:
                text = native                             # digital page: direct extract
                tag = "page [text]"
            else:
                pix = doc[i].get_pixmap(dpi=300)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text = ocr_image(img, "ar")               # scanned page: Windows OCR
                tag = "page [WinOCR]"
            parts[i] = text
            if ckpt:
                ckpt.append(i, text)
            done_n += 1
            _emit(cb, done_n, n_sel, tag)
        doc.close()
        return self._assemble(parts, lo, hi), n_sel

    def _arabic_ocr(self, img):
        """Run one page image through the configured local Arabic model."""
        from ocr_local import ocr_image as _local_ocr
        return _local_ocr(img, self.config.arabic_model,
                          self.config.arabic_max_pixels,
                          self.config.arabic_max_new_tokens)

    def _convert_with_arabic(self, pdf_path, total, cb, ckpt=None, stop=None):
        """Local Arabic vision-model OCR. Offline, no key/quota.
        Page-by-page with checkpoint + pause support, because a page can take a
        while on a small GPU and resuming mid-book matters."""
        import fitz
        from PIL import Image
        from ocr_local import local_ocr_available, MODELS

        ok, msg = local_ocr_available()
        if not ok:
            raise RuntimeError(f"Local model unavailable: {msg}")

        name = self.config.arabic_model
        done = ckpt.load() if ckpt else {}
        parts: dict[int, str] = dict(done)
        doc = fitz.open(str(pdf_path))
        total = total or doc.page_count
        lo, hi = self._range(doc.page_count)
        n_sel = hi - lo + 1
        _emit(cb, 0, n_sel,
              f"Loading {name} model (first run downloads it once)")

        done_n = 0
        t_start = time.time()
        for i in range(lo, hi + 1):
            if i in parts:
                done_n += 1
                _emit(cb, done_n, n_sel, f"page {i + 1} [cached]")
                continue
            if stop is not None and stop.is_set():
                doc.close()
                raise PauseRequested(done_n, n_sel)
            t0 = time.time()
            pix = doc[i].get_pixmap(dpi=self.config.arabic_dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            parts[i] = self._arabic_ocr(img)
            if ckpt:
                ckpt.append(i, parts[i])
            done_n += 1
            dt = time.time() - t0
            # Show the per-page cost and a live estimate, so a slow model looks
            # like a slow model rather than a frozen program.
            avg = (time.time() - t_start) / max(done_n, 1)
            left = (n_sel - done_n) * avg
            _emit(cb, done_n, n_sel,
                  f"page {i + 1} [{name}] {dt:.0f}s (~{left / 60:.0f} min left)")
        doc.close()
        return self._assemble(parts, lo, hi), n_sel

    # ---- Hybrid path: fast local pass + AI fixer for bad pages ---------- #
    def _pick_fixer(self):
        """Return (name, callable(img)->text) for the page fixer used by "fast"
        mode: the local Arabic model (free, offline)."""
        from ocr_local import local_ocr_available
        ok, msg = local_ocr_available()
        if not ok:
            return None, msg
        return self.config.arabic_model, self._arabic_ocr

    def _convert_with_hybrid(self, pdf_path, total, cb, ckpt=None, stop=None):
        import fitz
        from PIL import Image

        fixer_name, fixer = self._pick_fixer()
        if fixer is None:
            raise RuntimeError(f"Fast mode needs the QARI model: {fixer_name}")

        # Local OCR for scanned pages: Windows OCR if available, else None
        win_ok = False
        try:
            from ocr_windows import ocr_image, windows_ocr_available
            win_ok, _ = windows_ocr_available()
        except Exception:
            pass

        done = ckpt.load() if ckpt else {}
        parts: dict[int, str] = dict(done)
        fixed = 0
        doc = fitz.open(str(pdf_path))
        total = total or doc.page_count
        lo, hi = self._range(doc.page_count)
        n_sel = hi - lo + 1
        done_n = 0
        for i in range(lo, hi + 1):
            page = doc[i]
            if i in parts:
                done_n += 1
                _emit(cb, done_n, n_sel, f"page {i + 1} [cached]")
                continue
            if stop is not None and stop.is_set():
                doc.close()
                raise PauseRequested(done_n, n_sel)

            # 1) fast local pass
            text = page.get_text("text").strip()
            tag = "text"
            if len(text) < self.config.text_threshold_chars and win_ok:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text = ocr_image(img, "ar")
                tag = "WinOCR"

            # 2) score, and AI-fix only if bad. Native PDF text gets a stricter
            # (lower) effective threshold: it can hide font/cmap-encoding corruption
            # that still LOOKS like valid words, which the symbol-based heuristic misses.
            score = arabic_quality_score(text)
            effective_threshold = self.config.hybrid_threshold
            if tag == "text":
                effective_threshold *= self.config.hybrid_native_text_penalty
            if score > effective_threshold:
                pix = page.get_pixmap(dpi=self.config.arabic_dpi)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text = fixer(img)
                tag = f"{fixer_name}-fix"
                fixed += 1

            parts[i] = text
            if ckpt:
                ckpt.append(i, text)
            done_n += 1
            _emit(cb, done_n, n_sel, f"page {i + 1} [{tag}] q={score:.2f}")
        doc.close()
        print(f"    Fast mode: {fixed}/{n_sel} pages needed {fixer_name}; "
              f"{n_sel - fixed} were fine from the quick local pass.", flush=True)
        return self._assemble(parts, lo, hi), n_sel

    # ---- Public API ---------------------------------------------------- #
    def convert(self, pdf_path, out_dir, progress_cb: ProgressCB = None,
                stop_event=None) -> ConversionResult:
        pdf_path = Path(pdf_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        total = count_pages(pdf_path)

        # A partial run gets its own filename, so trying pages 1-3 never
        # overwrites a finished full conversion of the same book.
        lo, hi = self._range(total)
        partial = (lo, hi) != (0, max(total - 1, 0))
        suffix = f" (p{lo + 1}-{hi + 1})" if partial else ""
        out_file = out_dir / (pdf_path.stem + suffix + ".md")

        if partial:
            print(f"    Pages: {lo + 1}-{hi + 1} of {total or '?'} | "
                  f"Device: {self.device}", flush=True)
        else:
            print(f"    Pages: {total or 'unknown'} | Device: {self.device}",
                  flush=True)

        t0 = time.time()
        # Backend chain with graceful fallback: if the chosen engine can't run,
        # try the next-best rather than failing the whole book.
        runners = {
            "arabic-model": self._convert_with_arabic,
            "fast": self._convert_with_hybrid,
            "windows-ocr": self._convert_with_windows,
            "marker": self._convert_with_marker,
            "text-layer": self._convert_with_fallback,
        }
        backend = BACKEND_ALIASES.get(self.config.ocr_backend,
                                      self.config.ocr_backend)

        if backend == "arabic":
            chain = ["arabic-model", "windows-ocr", "marker", "text-layer"]
        elif backend == "fast":
            chain = ["fast", "windows-ocr", "marker", "text-layer"]
        elif backend == "windows":
            chain = ["windows-ocr", "marker", "text-layer"]
        else:                                    # "english"
            chain = ["marker", "windows-ocr", "text-layer"]

        text = pages = None
        engine_used = ""
        last_exc = None
        for name in chain:
            ckpt = _Checkpoint(out_dir, pdf_path.stem, name) if self.config.resume else None
            # Say plainly how much work is actually left. Without this, a run that
            # is mostly served from an earlier run's checkpoint looks misleadingly
            # fast — and the checkpoint is deleted on success, so the NEXT run
            # redoes everything and looks misleadingly slow.
            if ckpt:
                _cached = len(ckpt.load())
                if _cached:
                    print(f"    Resuming: {_cached}/{total} pages already done, "
                          f"{max(total - _cached, 0)} left to convert", flush=True)
                elif total:
                    print(f"    Starting fresh: {total} pages to convert",
                          flush=True)
            try:
                text, pages = runners[name](pdf_path, total, progress_cb, ckpt, stop_event)
                engine_used = name
                if ckpt:
                    ckpt.clear()          # completed fully: drop the checkpoint
                break
            except PauseRequested:
                raise                     # propagate up; progress is saved
            except Exception as exc:
                last_exc = exc
                logger.warning("Backend '%s' failed: %s", name, exc)
                print(f"    ! {name} failed ({exc}) - trying next", flush=True)
        if engine_used == "":
            raise RuntimeError(f"All OCR backends failed. Last error: {last_exc}")

        # Post-processing WITHOUT reflow (verses must stay line-separated first)
        text = postprocess(text, self.config, apply_reflow=False)

        # Optional Quran correction layer (runs on line-separated text)
        if self.config.correct_quran:
            try:
                qc = self._quran_corrector()
                if qc.ready:
                    _emit(progress_cb, total, total, "Correcting Quran verses")
                    text = qc.correct(text)
                    engine_used += "+quran"
                else:
                    print("    ! Quran layer not ready (run setup_quran.py).",
                          flush=True)
            except Exception as exc:
                logger.warning("Quran correction failed: %s", exc)

        # Reflow LAST, so prose flows while corrected verses stay on their own line
        text = finalize_reflow(text, self.config)

        front = (
            "---\n"
            f"title: {pdf_path.stem}\n"
            f"source: {pdf_path.name}\n"
            f"engine: {engine_used}\n"
            "tags: [pdf-import]\n"
            "---\n\n"
        )
        out_file.write_text(front + text, encoding="utf-8")

        return ConversionResult(
            source=pdf_path,
            output=out_file,
            engine_used=engine_used,
            pages=pages,
            seconds=round(time.time() - t0, 1),
        )
