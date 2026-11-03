"""
ocr_windows.py
==============
OCR backend based on Windows.Media.Ocr (the same engine used by Snipping Tool).
Much better than Surya/EasyOCR for modern printed Arabic.

Requires:
  - Windows 10/11
  - The Arabic OCR recognizer pack (install via: Settings > Time & Language >
    Language > add Arabic > Language options > Optical character recognition).
  - Package: winocr  (pip install winocr)

Provides:
  windows_ocr_available()  -> (bool, message)
  ocr_image(pil_img, lang) -> str
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("pdf2md.winocr")

_LANG = "ar"   # try "ar-SA" if Arabic is not detected


def _import_winocr():
    import winocr  # type: ignore
    return winocr


def windows_ocr_available() -> tuple[bool, str]:
    """Check that winocr and the Arabic language pack are available."""
    try:
        _import_winocr()
    except Exception as exc:
        return False, (f"winocr not installed or unavailable ({exc}). "
                       f"Install it: pip install winocr")
    try:
        from winrt.windows.media.ocr import OcrEngine
        langs = OcrEngine.available_recognizer_languages
        tags = [l.language_tag.lower() for l in langs]
        if not any(t.startswith("ar") for t in tags):
            return False, ("Arabic OCR pack not installed in Windows. "
                           "Add it via: Settings > Language > Arabic > OCR. "
                           f"Available: {tags}")
        return True, "available"
    except Exception as exc:
        # If the check fails, assume available and let recognition surface errors
        logger.warning("Could not enumerate OCR languages: %s", exc)
        return True, "unverified (check skipped)"


async def _recognize(img, lang: str):
    winocr = _import_winocr()
    return await winocr.recognize_pil(img, lang)


def ocr_image(pil_img, lang: str = _LANG) -> str:
    """Run Windows OCR on a PIL image and return the text."""
    try:
        result = asyncio.run(_recognize(pil_img, lang))
    except RuntimeError:
        # If an event loop is already running (rare inside a thread), make a new one
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_recognize(pil_img, lang))
        finally:
            loop.close()
    text = getattr(result, "text", None)
    if text is None and hasattr(result, "lines"):
        text = "\n".join(line.text for line in result.lines)
    return text or ""
