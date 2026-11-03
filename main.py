"""
main.py
=======
Command-line interface for the PDF -> Markdown tool.

Examples:
  python main.py "book.pdf" -o vault/Imported
  python main.py ./books -o vault/Imported --ocr windows --quran
  python main.py ./books --cpu
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Set CUDA memory config before importing torch (reduces fragmentation on 4GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import branding
from engine import EngineConfig, PdfToMarkdownEngine, BACKENDS


def collect_pdfs(target: Path) -> list[Path]:
    if target.is_file() and target.suffix.lower() == ".pdf":
        return [target]
    if target.is_dir():
        return sorted(target.rglob("*.pdf"))
    return []


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert PDF books to clean Markdown for Obsidian (Marker + fallback)."
    )
    p.add_argument("input", help="PDF file or a folder containing PDFs")
    p.add_argument("-o", "--output", default="output_md", help="output folder")
    p.add_argument("--cpu", action="store_true", help="disable GPU")
    p.add_argument("--langs", default="ar,en", help="OCR languages (default: ar,en)")
    p.add_argument("--ocr", choices=BACKENDS, default="fast",
                   help="fast (quick pass + AI only where needed) | "
                        "arabic (best quality, slowest) | "
                        "english (layout-aware) | windows (fastest, basic)")
    p.add_argument("--model", choices=["katib", "qari"], default="katib",
                   help="local Arabic model: katib (0.8B, fast) | qari (2B, slower)")
    p.add_argument("--dpi", type=int, default=200,
                   help="render DPI for the Arabic model (lower = faster)")
    p.add_argument("--pages", default="",
                   help="page range, 1-based inclusive: '5-12', '3-' or '-10'")
    p.add_argument("--quran", action="store_true",
                   help="enable Quran correction layer (needs setup_quran.py)")
    p.add_argument("--images", action="store_true", help="extract embedded images")
    p.add_argument("--reflow", action="store_true", help="merge lines within a paragraph")
    p.add_argument("--no-symbols", action="store_true", help="disable symbol fixes")
    p.add_argument("--no-footnotes", action="store_true", help="disable footnote conversion")
    p.add_argument("--no-hyphen", action="store_true", help="disable de-hyphenation")
    p.add_argument("--strip-footnotes", action="store_true",
                   help="delete footnotes and reference numbers entirely")
    p.add_argument("--keep-artifacts", action="store_true",
                   help="keep scan gutter lines / rules (they are removed by default)")
    p.add_argument("--no-resume", action="store_true",
                   help="disable page checkpointing / resume")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    p.add_argument("--quiet", action="store_true",
                   help="suppress the attribution line at the end of a run")
    return p


def parse_pages(spec: str) -> tuple[int, int]:
    """'5-12' -> (5, 12) | '3-' -> (3, 0) | '-10' -> (0, 10) | '7' -> (7, 7)."""
    spec = (spec or "").strip()
    if not spec:
        return 0, 0
    if "-" not in spec:
        n = int(spec)
        return n, n
    a, _, b = spec.partition("-")
    return (int(a) if a.strip() else 0), (int(b) if b.strip() else 0)


def make_config(args) -> EngineConfig:
    start, end = parse_pages(args.pages)
    return EngineConfig(
        page_start=start,
        page_end=end,
        languages=[s.strip() for s in args.langs.split(",") if s.strip()],
        use_gpu=not args.cpu,
        ocr_backend=args.ocr,
        arabic_model=args.model,
        arabic_dpi=args.dpi,
        correct_quran=args.quran,
        resume=not args.no_resume,
        output_images=args.images,
        reflow_lines=args.reflow,
        fix_symbols=not args.no_symbols,
        fix_footnotes=not args.no_footnotes,
        strip_footnotes=args.strip_footnotes,
        clean_artifacts=not args.keep_artifacts,
        fix_hyphenation=not args.no_hyphen,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(message)s",
    )

    target = Path(args.input).expanduser()
    pdfs = collect_pdfs(target)
    if not pdfs:
        print(f"No PDF files found in: {target}", file=sys.stderr)
        return 1

    engine = PdfToMarkdownEngine(make_config(args))
    out_dir = Path(args.output).expanduser()
    print(f"Files: {len(pdfs)} | Device: {engine.device} | Output: {out_dir}\n")

    ok, fail, skipped = 0, 0, 0
    for idx, pdf in enumerate(pdfs, 1):
        # Resume: skip files already fully converted (their .md exists)
        if not args.no_resume and (out_dir / (pdf.stem + ".md")).exists():
            skipped += 1
            print(f"[{idx}/{len(pdfs)}] {pdf.name} - already done, skipping")
            continue
        print(f"[{idx}/{len(pdfs)}] {pdf.name} ...", flush=True)
        try:
            res = engine.convert(pdf, out_dir)
            ok += 1
            print(f"    OK  {res.output.name} | {res.engine_used} | "
                  f"{res.pages} pages | {res.seconds}s")
        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved - re-run to resume.", file=sys.stderr)
            return 130
        except Exception as exc:
            fail += 1
            print(f"    FAIL: {exc}", file=sys.stderr)

    print(f"\nDone. Succeeded: {ok} | Skipped: {skipped} | Failed: {fail}")
    # Attribution only — the CLI never opens a browser and never asks anything.
    if not args.quiet and not branding.branding_disabled():
        print(branding.CLI_LINE)
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
