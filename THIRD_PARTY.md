# Third-party dependencies and licenses

Licenses below were read from the **installed package metadata** in the
development environment (`.dist-info/METADATA`, classifiers, and the bundled
LICENSE text), and from each model's own model card. Versions are the ones this
tool was developed and tested against.

Anything that could not be confirmed from those sources is marked **UNVERIFIED**
rather than guessed.

---

## ⚠️ License flags that affect what you can license PDFtoMD as

Three dependencies are copyleft. They are not optional extras — PyMuPDF is used
on every code path, and Marker pulls in Surya.

| Package | License | Why it matters |
|---|---|---|
| **PyMuPDF** 1.28.0 | **AGPL-3.0** or a paid Artifex commercial license | AGPL is the strongest copyleft in this list. Used by every engine for reading and rendering PDFs. |
| **marker-pdf** 1.10.2 | **GPL-3.0-or-later** | The `english` engine. |
| **surya-ocr** 0.17.1 | **GPL-3.0-or-later** | Pulled in by marker-pdf. |

**Practical consequence:** if you distribute PDFtoMD with these dependencies as
a combined work, a permissive license (MIT/BSD/Apache) for your own code does
not make the whole thing permissive — the copyleft terms still reach the
distributed combination. The usual options are:

1. Release PDFtoMD under **AGPL-3.0** (the strictest of the three, which keeps
   you compatible with all of them), or
2. Buy a **commercial PyMuPDF license** from Artifex and drop or isolate
   marker/surya, if you want a permissive or proprietary release, or
3. Replace PyMuPDF with a permissively licensed PDF library (e.g. pypdfium2,
   BSD-3-Clause) and make the `english` engine an optional add-on the user
   installs themselves.

I am not a lawyer and this is not legal advice — please confirm with a
qualified source before you publish, especially if money is involved.

No non-commercial or OpenRAIL-style license was found among these dependencies.

---

## Python packages

| Package | Version | License | Source of the license text |
|---|---|---|---|
| PyMuPDF | 1.28.0 | AGPL-3.0 **or** Artifex commercial | METADATA `License:` field |
| marker-pdf | 1.10.2 | GPL-3.0-or-later | METADATA + OSI classifier |
| surya-ocr | 0.17.1 | GPL-3.0-or-later | METADATA + OSI classifier |
| transformers | 4.57.6 | Apache-2.0 | METADATA + classifier |
| accelerate | 1.14.0 | Apache-2.0 | METADATA + classifier |
| peft | 0.19.1 | Apache-2.0 | METADATA + classifier |
| qwen-vl-utils | 0.0.14 | Apache-2.0 | METADATA + classifier |
| torch | 2.11.0+cu128 | BSD-3-Clause | METADATA |
| torchvision | 0.26.0+cu128 | BSD | METADATA |
| Pillow | 10.4.0 | HPND (permissive) | METADATA + classifier |
| bitsandbytes | 0.49.2 | MIT | bundled LICENSE text |
| rapidfuzz | 3.14.5 | MIT-style permission grant | bundled LICENSE text |
| CustomTkinter | 6.0.0 | MIT | bundled LICENSE text |
| winocr | 0.0.15 | MIT | OSI classifier |

Note on CustomTkinter: its `License:` metadata field reads "Creative Commons
Zero v1.0" while its classifier and bundled LICENSE file both say MIT. The
bundled license text is authoritative here, so it is recorded as MIT.

Python itself and the Tcl/Tk libraries behind tkinter ship under their own
permissive licenses (PSF and Tcl/Tk BSD-style).

---

## Models (downloaded at runtime, not bundled)

| Model | Used for | License |
|---|---|---|
| [KATIB 0.8B v0.1](https://huggingface.co/oddadmix/Katib-Qwen3.5-0.8B-0.1) | Arabic OCR (default) | Apache-2.0, per the model card |
| [QARI-OCR 0.2.2.1](https://huggingface.co/NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct) | Arabic OCR (higher accuracy) | Apache-2.0 on the model card; the card also states it "follows the licensing terms of the original Qwen2-VL model" — **check the upstream Qwen2-VL terms before commercial use** |
| Qwen2-VL-2B / Qwen3.5-0.8B (base models) | Backbones of the two above | **UNVERIFIED** — not read directly; check the upstream Qwen repositories |
| Marker / Surya layout models | `english` engine | **UNVERIFIED** — model weights may carry terms separate from the GPL-3.0 code |

Neither model is included in this package. Both download on first use into the
user's HuggingFace cache.

## Data

| Source | Used for | License |
|---|---|---|
| [alquran.cloud](https://alquran.cloud/) Uthmani text | Quranic verse verification | **UNVERIFIED** — no license statement was read from the API or site during development |

The corpus is **not** included in this package; `setup_quran.py` downloads it on
first run.

---

## Attribution

The academic work behind QARI-OCR:

> Wasfy, A., Nacar, O., Elkhateb, A., Reda, M., Elshehy, O., Ammar, A., &
> Boulila, W. (2025). *QARI-OCR: High-Fidelity Arabic Text Recognition through
> Multimodal Large Language Model Adaptation.* arXiv:2506.02295
