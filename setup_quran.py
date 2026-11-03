"""
setup_quran.py
==============
Downloads the verified Uthmani Quran text and saves it to
data/quran-uthmani.json in the format:
    [{"surah": 1, "ayah": 1, "text": "..."}, ...]

Source: AlQuran Cloud API (quran-uthmani edition).
Runs once; skips download if the file exists (unless --force).

Run:
    python setup_quran.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

API = "https://api.alquran.cloud/v1/quran/quran-uthmani"
OUT = Path(__file__).with_name("data") / "quran-uthmani.json"


def download() -> list[dict]:
    print(f"Downloading Quran from: {API} ...")
    req = urllib.request.Request(API, headers={"User-Agent": "pdf2md-setup"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    verses: list[dict] = []
    for surah in raw["data"]["surahs"]:
        s = surah["number"]
        for ayah in surah["ayahs"]:
            verses.append({
                "surah": s,
                "ayah": ayah["numberInSurah"],
                "text": ayah["text"].strip(),
            })
    if len(verses) < 6000:
        raise RuntimeError(f"Unexpected verse count: {len(verses)}")
    return verses


def main(argv=None) -> int:
    force = "--force" in (argv or sys.argv[1:])
    if OUT.exists() and not force:
        print(f"Already exists: {OUT}  (use --force to re-download)")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        verses = download()
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print("Check your internet connection and try again.", file=sys.stderr)
        return 1
    OUT.write_text(json.dumps(verses, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {OUT}  ({len(verses)} verses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
