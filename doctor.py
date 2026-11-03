"""
doctor.py
=========
Check the setup and measure real speed. Run it and paste the output.

    .venv\\Scripts\\python.exe doctor.py
    .venv\\Scripts\\python.exe doctor.py "path\\to\\book.pdf"
    .venv\\Scripts\\python.exe doctor.py "path\\to\\book.pdf" 5      (test page 5)

With a PDF it runs one real page through each local model and reports seconds
and characters produced, so you can pick the model on evidence instead of guesswork.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _hdr(t):
    print("\n" + t)
    print("-" * len(t))


def main() -> int:
    _hdr("1. PyTorch / GPU")
    try:
        import torch
    except Exception as exc:
        print(f"  FAIL: torch not importable: {exc}")
        return 1
    print(f"  torch            : {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"  CUDA available   : {cuda}")
    vram = 0.0
    if cuda:
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU              : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total       : {vram:.2f} GB")
        if vram < 6:
            print("  >> Small GPU. The 0.8B 'katib' model is the right choice:")
            print("     it fits as fp16, so it avoids slow 4-bit kernels.")
    else:
        print("  >> No CUDA. Everything runs on the CPU and will be very slow.")

    _hdr("2. 4-bit support (only needed by the 2B 'qari' model)")
    try:
        import bitsandbytes as bnb
        print(f"  bitsandbytes     : {bnb.__version__}  OK")
    except Exception as exc:
        print(f"  bitsandbytes     : not usable -> {exc}")
        print("  >> Only matters if you pick 'qari'. 'katib' does not need it.")
    try:
        import triton  # noqa: F401
        print("  triton           : present")
    except Exception:
        print("  triton           : MISSING (normal on Windows)")
        print("  >> Without triton, 4-bit inference is very slow — which is why")
        print("     the 2B 'qari' model crawls here and 'katib' does not.")

    _hdr("3. Supporting libraries")
    for mod in ("transformers", "accelerate", "peft", "fitz", "rapidfuzz",
                "PIL"):
        try:
            m = __import__(mod)
            print(f"  {mod:<16} : {getattr(m, '__version__', 'ok')}")
        except Exception as exc:
            print(f"  {mod:<16} : MISSING ({exc})")

    _hdr("4. Quran corpus")
    corpus = Path(__file__).with_name("data") / "quran-uthmani.json"
    print(f"  {'found' if corpus.exists() else 'MISSING - run setup_quran.py'}"
          f"  ({corpus})")

    pdf = sys.argv[1] if len(sys.argv) > 1 else None
    if not pdf:
        _hdr("5. Page test")
        print("  Skipped. Re-run with a PDF path to measure a real page:")
        print('    .venv\\Scripts\\python.exe doctor.py "book.pdf"')
        return 0

    _hdr("5. Page test")
    try:
        import fitz
        from PIL import Image
        from engine import EngineConfig
        from ocr_local import ocr_image, MODELS

        cfg = EngineConfig()
        doc = fitz.open(pdf)

        # Test a page that actually has content, not a near-blank cover.
        page_no = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else None
        if page_no is None:
            best_i, best_len = min(2, doc.page_count - 1), -1
            for i in range(min(12, doc.page_count)):
                n = len(doc[i].get_text("text").strip())
                if n > best_len:
                    best_i, best_len = i, n
            page_no = best_i
        page_no = max(0, min(page_no, doc.page_count - 1))
        page = doc[page_no]
        print(f"  testing page     : {page_no + 1} of {doc.page_count}")
        pix = page.get_pixmap(dpi=cfg.arabic_dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        print(f"  page image       : {pix.width}x{pix.height} @ {cfg.arabic_dpi} dpi")

        _hdr("6. Model comparison on this page")
        print("  Each model is downloaded once on first use.\n")
        print(f"  {'model':<8} {'size':<6} {'seconds':>9} {'chars':>8}  note")
        results = []
        for name, spec in MODELS.items():
            try:
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                out = ocr_image(img, name, cfg.arabic_max_pixels,
                                cfg.arabic_max_new_tokens)
                dt = time.time() - t0
                vram_used = (torch.cuda.max_memory_allocated() / 1e9) if cuda else 0
                size = "0.8B" if name == "katib" else "2B"
                results.append((name, dt, len(out), out, vram_used))
                print(f"  {name:<8} {size:<6} {dt:>8.1f}s {len(out):>8}  "
                      f"VRAM {vram_used:.2f} GB")
            except Exception as exc:
                print(f"  {name:<8} FAILED: {exc}")

        if results:
            fastest = min(results, key=lambda r: r[1])
            richest = max(results, key=lambda r: r[2])
            _hdr("7. Sample output")
            print(f"  (from '{richest[0]}')\n")
            sample = richest[3][:600].replace("\n", "\n  ")
            print("  " + (sample if sample.strip() else "*** EMPTY OUTPUT ***"))

            _hdr("Verdict")
            for name, dt, chars, _out, _v in results:
                print(f"  {name}: ~{dt:.0f}s/page  ->  "
                      f"~{dt * doc.page_count / 60:.0f} min for this {doc.page_count}-page file")
            print(f"\n  Fastest: {fastest[0]}. Most text read: {richest[0]}.")
            if fastest[0] != richest[0]:
                ratio = richest[1] / max(fastest[1], 0.01)
                print(f"  '{richest[0]}' is {ratio:.1f}x slower for "
                      f"{richest[2] - fastest[2]:+d} characters.")
                print("  Unless that difference matters to you, use "
                      f"'{fastest[0]}'.")
        doc.close()
    except Exception as exc:
        print(f"  Page test failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
