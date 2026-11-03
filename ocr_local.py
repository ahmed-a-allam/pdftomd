"""
ocr_local.py
============
Local, offline Arabic OCR using a small vision-language model.
No API key, no quota, no internet after the one-time model download.

Two models are available:

  katib  (default) - oddadmix/Katib-Qwen3.5-0.8B-0.1, 0.8B parameters.
  qari             - NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct, 2B parameters.

Why KATIB is the default even though QARI scores slightly better:

  A 0.8B model in float16 needs ~1.6 GB and fits entirely in a 4 GB GPU.
  A 2B model in float16 needs ~4.4 GB and does NOT, so it has to be loaded in
  4-bit through bitsandbytes. On Windows bitsandbytes has no Triton kernels, so
  every forward pass pays a slow de-quantization cost — measured at roughly
  1-2 generated tokens/second, i.e. many minutes for one dense page.
  KATIB avoids that path completely: smaller model AND fast fp16 kernels.
  Published accuracy gap is small (CER 0.065 vs 0.050).

Each model is called with the exact prompt from its own model card. These are
fine-tuned models; paraphrasing the prompt measurably degrades their output.

Provides:
  local_ocr_available()        -> (bool, message)
  ocr_image(pil_img, ...)      -> str
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger("pdf2md.local")

MODELS: dict[str, dict] = {
    "katib": {
        "repo": "oddadmix/Katib-Qwen3.5-0.8B-0.1",
        "base": "unsloth/Qwen3.5-0.8B",
        # Exact prompt from the KATIB model card.
        "prompt": "Free OCR",
        # 0.8B in fp16 fits in 4 GB -> never quantize, keep the fast path.
        "quantize": False,
        "max_new_tokens": 1024,
        "note": "0.8B - fast, fits in VRAM as fp16",
    },
    "qari": {
        "repo": "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct",
        "base": "unsloth/Qwen2-VL-2B-Instruct-unsloth-bnb-4bit",
        # Exact prompt from the QARI model card - do not shorten it.
        "prompt": ("Below is the image of one page of a document, as well as "
                   "some raw textual content that was previously extracted for "
                   "it. Just return the plain text representation of this "
                   "document as if you were reading it naturally. Do not "
                   "hallucinate."),
        "quantize": True,
        "max_new_tokens": 1600,
        "note": "2B - slightly more accurate, much slower on a small GPU",
    },
}
DEFAULT_MODEL = "katib"

# Qwen vision models use dynamic resolution: compute scales with the image's
# pixel count, so this cap is the main speed/quality dial. 28*28 = 1 visual token.
_MIN_PIXELS = 256 * 28 * 28
_MAX_PIXELS = 1280 * 28 * 28

_lock = threading.Lock()
_loaded: dict = {}          # (name, max_pixels) -> (model, processor, cfg)


def local_ocr_available() -> tuple[bool, str]:
    """Check the required libraries are importable (weights download lazily)."""
    missing = [m for m in ("torch", "transformers")
               if not _importable(m)]
    if missing:
        return False, ("Missing: " + ", ".join(missing) +
                       ". Install with: pip install transformers accelerate")
    return True, "available"


def _importable(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _quant_config():
    """4-bit NF4 - only used for models too big to fit in fp16."""
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def _load(name: str, max_pixels: int):
    ck = (name, max_pixels)
    if ck in _loaded:
        return _loaded[ck]

    import torch
    from transformers import AutoProcessor

    spec = MODELS[name]
    on_gpu = torch.cuda.is_available()
    dtype = torch.float16 if on_gpu else torch.float32

    load_kw: dict = {"trust_remote_code": True}
    if on_gpu and spec["quantize"]:
        try:
            load_kw["quantization_config"] = _quant_config()
            load_kw["device_map"] = {"": 0}
        except Exception as exc:
            logger.warning("bitsandbytes unavailable (%s); using fp16 + offload.",
                           exc)
            load_kw["torch_dtype"] = dtype
            load_kw["device_map"] = "auto"
    else:
        load_kw["torch_dtype"] = dtype
        if on_gpu:
            load_kw["device_map"] = {"": 0}     # keep it all on the GPU
    load_kw["attn_implementation"] = "sdpa"

    proc_kw = {"trust_remote_code": True,
               "min_pixels": _MIN_PIXELS, "max_pixels": max_pixels}

    model = processor = None
    errors = []

    # 1) Generic image-text-to-text loader (works for Qwen2-VL and Qwen3-VL).
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(spec["repo"],
                                                            **load_kw)
        processor = AutoProcessor.from_pretrained(spec["repo"], **proc_kw)
    except Exception as exc:
        errors.append(f"AutoModelForImageTextToText: {exc}")

    # 2) Qwen2-VL specific class.
    if model is None:
        try:
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                spec["repo"], **load_kw)
            processor = AutoProcessor.from_pretrained(spec["repo"], **proc_kw)
        except Exception as exc:
            errors.append(f"Qwen2VL: {exc}")

    # 3) Adapter on top of its base model.
    if model is None:
        try:
            from peft import PeftModel
            from transformers import AutoModelForImageTextToText
            base = AutoModelForImageTextToText.from_pretrained(spec["base"],
                                                               **load_kw)
            model = PeftModel.from_pretrained(base, spec["repo"])
            try:
                processor = AutoProcessor.from_pretrained(spec["repo"], **proc_kw)
            except Exception:
                processor = AutoProcessor.from_pretrained(spec["base"], **proc_kw)
        except Exception as exc:
            errors.append(f"PEFT: {exc}")

    if model is None:
        raise RuntimeError(f"Could not load '{name}' ({spec['repo']}). "
                           + " | ".join(errors))

    model.eval()
    try:
        dev = next(model.parameters()).device
        quant = "4-bit" if spec["quantize"] else "fp16"
        print(f"    Model: {name} ({quant}) on {dev}", flush=True)
    except Exception:
        pass

    _loaded[ck] = (model, processor, spec)
    return _loaded[ck]


def is_degenerate(text: str, min_repeat: int = 6, min_share: float = 0.10) -> bool:
    """Detect a repetition collapse - the model looping one phrase instead of
    transcribing. Real pages never repeat a 4-word window this often."""
    words = text.split()
    if len(words) < 24:
        return False
    n = 4
    shingles = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not shingles:
        return False
    counts: dict[str, int] = {}
    for s in shingles:
        counts[s] = counts.get(s, 0) + 1
    top = max(counts.values())
    return top >= min_repeat and (top / len(shingles)) > min_share


def ocr_image(pil_img, model_name: str = DEFAULT_MODEL,
              max_pixels: int = _MAX_PIXELS,
              max_new_tokens: int = 0) -> str:
    """Transcribe one page image locally. Returns the text."""
    import torch

    if model_name not in MODELS:
        model_name = DEFAULT_MODEL

    with _lock:                      # one small GPU: one page at a time
        model, processor, spec = _load(model_name, max_pixels)
        limit = max_new_tokens or spec["max_new_tokens"]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": spec["prompt"]},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil_img],
                           return_tensors="pt").to(model.device)

        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=limit,
                                 do_sample=False, use_cache=True)
        out = processor.decode(gen[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()

    # Note: no automatic retry here. Re-running generation would double the cost
    # of a page that is already the slow case, so the page is flagged instead and
    # the user can re-do just that one.
    if is_degenerate(out):
        logger.warning("Repetition collapse on a page - flagged for review.")
        return ("[⚠ OCR-FAIL: the model got stuck repeating a phrase on this "
                "page - needs a manual re-check]\n" + out[:300])
    return out
