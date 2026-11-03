"""
branding.py
===========
Author/attribution constants and the one-time first-run prompt.

Everything brand-related lives here so there is a single place to change it.
This module does NOT touch conversion behaviour.

Privacy: nothing here contacts the network. The only thing that can open a
connection is the user explicitly clicking "Yes" in the first-run dialog, which
hands SITE_URL to the operating system's default browser. No telemetry, no
tracking parameters, no analytics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# --- Single source of truth -------------------------------------------------- #
SITE_URL = "https://ahmed-a-allam.github.io"

APP_NAME = "PDFtoMD"
VERSION = "1.0"
AUTHOR = "Ahmed Adel Allam"

# Shown in the title bar.
WINDOW_TITLE = f"{APP_NAME} by {AUTHOR}"

# Display form of the URL (no scheme), used in the footer and CLI line.
SITE_LABEL = SITE_URL.split("://", 1)[-1]

FOOTER_TEXT = f"Made by {AUTHOR} · {SITE_LABEL}"
CLI_LINE = f"{APP_NAME} by {AUTHOR} · {SITE_LABEL}"

FIRST_RUN_QUESTION = (
    f"{APP_NAME} is made by {AUTHOR}.\n\nVisit the author's website?"
)

# Third-party work this tool depends on.
CREDITS = [
    ("KATIB (Arabic OCR model)", "oddadmix/Katib-Qwen3.5-0.8B-0.1"),
    ("QARI-OCR (Arabic OCR model)", "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct"),
    ("Marker (layout analysis)", "github.com/VikParuchuri/marker"),
    ("Uthmani Quran corpus", "alquran.cloud"),
]

ABOUT_TEXT = "\n".join([
    f"{APP_NAME} {VERSION}",
    f"by {AUTHOR}",
    SITE_URL,
    "",
    "Converts PDF books into clean Markdown, built for classical Arabic",
    "with diacritics, Quranic verses and scholarly footnotes.",
    "",
    "Credits:",
    *[f"  • {name} — {ref}" for name, ref in CREDITS],
])

# Environment escape hatch for scripts and CI.
NO_BRANDING_ENV = "PDFTOMD_NO_BRANDING"


def branding_disabled() -> bool:
    return os.environ.get(NO_BRANDING_ENV, "").strip() not in ("", "0", "false")


# --- First-run flag ---------------------------------------------------------- #
# Stored in the user's app-data folder, deliberately NOT in the project folder,
# so it is never committed and survives moving or re-cloning the tool.

def settings_path() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / APP_NAME / "settings.json"


def _read_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_settings(data: dict) -> None:
    try:
        p = settings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    except Exception:
        pass        # a settings file we cannot write is not worth failing over


def already_prompted() -> bool:
    return bool(_read_settings().get("first_run_prompted"))


def mark_prompted() -> None:
    data = _read_settings()
    data["first_run_prompted"] = True
    _write_settings(data)


def should_prompt() -> bool:
    """True only on the very first launch, and never under PDFTOMD_NO_BRANDING."""
    return not branding_disabled() and not already_prompted()


def open_site() -> None:
    """Hand SITE_URL to the default browser. Called only on explicit consent."""
    import webbrowser
    try:
        webbrowser.open_new_tab(SITE_URL)
    except Exception:
        pass
