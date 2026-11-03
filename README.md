# PDFtoMD

**Arabic-first PDF to Markdown, built for Obsidian.**

Turns Arabic books into clean Markdown while keeping diacritics (tashkeel), Quranic verses, ﷺ and footnotes intact. It runs entirely on your own computer: free, private, no accounts.

**PDFtoMD**: تحويل كتب PDF العربية إلى Markdown نظيف لـ Obsidian. يحافظ على التشكيل والآيات القرآنية ورمز ﷺ والحواشي. يعمل بالكامل على جهازك: مجاني وخاص وبلا حسابات.

---

## Quick start (Windows)

1. Install [Python](https://www.python.org/downloads/) (tick *"Add Python to PATH"*).
2. Download this repo and double-click **`run.bat`**.

That's it. On first launch `run.bat` builds its own environment and downloads
what it needs. Later launches start immediately.

> **Note on size:** the repo itself is tiny — no models are included. The Arabic
> OCR model (~2 GB) is downloaded automatically the first time you use it, and
> cached in your HuggingFace folder, not in this project. After that it works
> completely offline.

---

## Choosing an engine

| Engine | Best for | Speed | Quality on Arabic |
|---|---|---|---|
| **fast** *(default)* | Everyday use | Fast | High — AI is used only where needed |
| **arabic** | Classical/diacritized Arabic, scanned books | Slow | Highest |
| **english** | English books, digital PDFs, complex layout | Fast | n/a |
| **windows** | Quick drafts, modern printed Arabic | Fastest | Basic |

**Start with `fast`.** It reads each page the cheap way first, checks the result
for signs of garbling, and only pays the cost of the AI model on pages that
actually need it. Switch to `arabic` when you want maximum fidelity and don't
mind waiting.

### Arabic model: `katib` (default) or `qari`

| Model | Size | Speed on a 4 GB GPU | Accuracy (CER) |
|---|---|---|---|
| **katib** | 0.8B | Fast | 0.065 |
| qari | 2B | Much slower | 0.050 |

The size difference is not the whole story. A 0.8B model runs in float16 and
fits in a 4 GB card; a 2B model does not, so it must be loaded in 4-bit through
`bitsandbytes` — and on Windows there are no Triton kernels for that, so every
page pays a heavy de-quantization cost. **`katib` is therefore several times
faster for a small accuracy difference.** Pick `qari` only if you have a larger
GPU or you need the last fraction of accuracy and can wait.

### Why these and not a cloud API?

Cloud vision APIs were tested and removed. Free tiers were either tiny (~20
pages/day), geo-restricted, or the models hallucinated — inventing fluent Arabic
text that was never on the page, which is far more dangerous than obvious
garbling because it reads as correct. The local model is free, unlimited, and
private.

---

## What it fixes automatically

- **`ﷺ` recovery** — detects the corrupted glyph clusters OCR produces after
  honorifics (النبي، الرسول، رسول الله، المصطفى…) and restores the symbol.
- **Quranic verses** — fuzzy-matches quoted verses against a verified Uthmani
  corpus and replaces them with the correct text plus a `[Quran surah:ayah]`
  reference. Uses a confidence margin, so it won't "correct" ordinary text.
- **Footnotes** — converts `(1)` / `)1(` markers into Markdown `[^1]` footnotes,
  handling Arabic-Indic digits.
- **Reversed brackets** — repairs `)2(` → `(2)` from RTL text extraction.
- **De-hyphenation** — rejoins Latin words split across line breaks.
- **Repetition guard** — if the AI model gets stuck looping a phrase, the page is
  flagged for review rather than silently saved as if it were correct.

---

## Try a few pages first

A vision model can take minutes per page on a modest GPU, so convert a small
range before committing to a whole book:

```bash
python main.py "book.pdf" --ocr arabic --pages 1-3
```

In the GUI, use the **Pages from / to** boxes (leave them empty for the whole
book). A partial run is saved under its own name — `book (p1-3).md` — so it
never overwrites a completed full conversion.

## Pause & resume

Long books can be interrupted at any time — press **Pause**, or just close the
window. Every finished page is checkpointed, so re-running picks up exactly where
it stopped. Already-converted books are skipped automatically.

---

## If it's slow

Run the built-in diagnostic:

```
.venv\Scripts\python.exe doctor.py "path\to\book.pdf"
```

It reports whether the model is genuinely running on your GPU in 4-bit, times a
real page, and tells you what to change. The usual culprit is a missing or broken
`bitsandbytes`, which forces part of the model onto the CPU.

Other levers:

- Use the **fast** engine instead of **arabic**.
- Lower the render resolution: `--qari-dpi 150`.

---

## Command line

```bash
python main.py "book.pdf" -o vault/Imported
python main.py ./books --ocr arabic --quran
python main.py ./books --ocr fast --qari-dpi 150
```

| Flag | Meaning |
|---|---|
| `--ocr` | `fast` \| `arabic` \| `english` \| `windows` |
| `--model` | `katib` (0.8B, fast) \| `qari` (2B, slower) |
| `--pages` | Page range, e.g. `5-12`, `3-`, `-10`, `7` |
| `--quran` | Enable Quranic verse correction |
| `--strip-footnotes` | Delete footnotes and reference numbers |
| `--dpi` | Render resolution for the Arabic model (lower = faster) |
| `--reflow` | Merge wrapped lines into flowing paragraphs |
| `--cpu` | Disable GPU |
| `--no-resume` | Ignore checkpoints and start fresh |

---

## Requirements

- Windows (the `windows` engine is Windows-only; everything else is portable)
- Python 3.10+
- An NVIDIA GPU is strongly recommended for the Arabic model. 4 GB VRAM is
  enough — the model is loaded in 4-bit specifically to fit.

---

## About the author

Made by **Ahmed Adel Allam** — [ahmed-a-allam.github.io](https://ahmed-a-allam.github.io)

On its **first launch only**, the GUI asks once whether you'd like to visit the
author's website, with a *Don't ask again* box. It never opens a browser without
you clicking Yes, and it never asks again afterwards. The answer is remembered in
a small settings file in your user app-data folder (`%APPDATA%\PDFtoMD\settings.json`
on Windows) — not in the project folder, and not in the repository.

To skip it entirely, for scripts or CI:

```
set PDFTOMD_NO_BRANDING=1
```

This also suppresses the attribution line the CLI prints at the end of a run;
`--quiet` does the same for a single run.

The tool sends no telemetry and makes no network calls of its own. The only
downloads are the OCR model and the Quran corpus, both described above.

## Third-party licenses

The OCR models and the Quran corpus are **not bundled with this package** — they
are downloaded at runtime, each under its own licence terms. Several Python
dependencies are copyleft (PyMuPDF is AGPL-3.0, Marker and Surya are GPL-3.0),
which affects how this tool as a whole may be redistributed.

You are responsible for complying with the terms of every model, dataset and
library you download and use. See **[THIRD_PARTY.md](THIRD_PARTY.md)** for the
full list, the licence of each, and the items that remain unverified.

## Credits

Arabic OCR by [KATIB](https://huggingface.co/oddadmix/Katib-Qwen3.5-0.8B-0.1)
(oddadmix) and [QARI-OCR](https://huggingface.co/NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct)
(NAMAA-Space), Qwen vision models fine-tuned for Arabic — see
[arXiv:2506.02295](https://arxiv.org/abs/2506.02295). Layout analysis by
[Marker](https://github.com/VikParuchuri/marker). Quran text from
[alquran.cloud](https://alquran.cloud/).

## License

PDFtoMD is released under the GNU Affero General Public License v3.0
(AGPL-3.0). See the LICENSE file. Third-party components keep their own
licenses: see [THIRD_PARTY.md](THIRD_PARTY.md).
