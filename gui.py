"""
gui.py
======
Simple graphical interface for the PDF -> Markdown tool.

Uses CustomTkinter if installed, otherwise falls back to standard tkinter.
Conversion runs in a separate thread so the UI stays responsive, showing a
progress bar and a live log via a callback from the engine.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

_CONFIG = Path(__file__).with_name("config.json")


def _load_cfg() -> dict:
    try:
        return json.loads(_CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cfg(data: dict) -> None:
    try:
        _CONFIG.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

# Set CUDA memory config before any torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --- Choose UI library ------------------------------------------------------ #
try:
    import customtkinter as ctk
    _CTK = True
except Exception:
    import tkinter as ctk  # type: ignore
    _CTK = False

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import branding
from engine import (EngineConfig, PdfToMarkdownEngine,
                    BACKENDS, BACKEND_ALIASES)

_ENGINE_HELP = {
    "fast":    "fast: quick local pass, AI model only on pages that need it.",
    "arabic":  "arabic: best quality for classical/diacritized Arabic. Slowest.",
    "english": "english: layout-aware, best for English & digital books.",
    "windows": "windows: very fast, modern printed Arabic only. Lowest quality.",
}

def _as_int(s: str) -> int:
    """Parse a page-range box; blank or junk means 'no limit'."""
    try:
        return max(0, int(str(s).strip()))
    except Exception:
        return 0


_MODEL_HELP = {
    "katib": "0.8B - fast (recommended for 4GB GPUs)",
    "qari":  "2B - slightly better, much slower on a small GPU",
}


class App:
    def __init__(self):
        if _CTK:
            ctk.set_appearance_mode("System")
            ctk.set_default_color_theme("blue")
            self.root = ctk.CTk()
        else:
            self.root = tk.Tk()
        self.root.title(branding.WINDOW_TITLE)
        self.root.geometry("640x600")

        self.q: "queue.Queue[tuple]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop = threading.Event()   # set by Pause; checked by the engine
        self.cfg = _load_cfg()

        self._build_menu()
        self._build()
        self.root.after(120, self._poll)
        # Ask once, after the window exists so the dialog has a parent.
        self.root.after(400, self._first_run_prompt)

    # ---- Branding ------------------------------------------------------ #
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self._about)
        helpmenu.add_command(label="Author's website",
                             command=branding.open_site)
        menubar.add_cascade(label="Help", menu=helpmenu)
        try:
            self.root.config(menu=menubar)
        except Exception:
            pass

    def _about(self):
        messagebox.showinfo(f"About {branding.APP_NAME}",
                            branding.ABOUT_TEXT, parent=self.root)

    def _first_run_prompt(self):
        """Ask once, on the very first launch only. Never opens the browser
        on its own — only if the user answers Yes."""
        if not branding.should_prompt():
            return
        win = tk.Toplevel(self.root)
        win.title(branding.APP_NAME)
        win.transient(self.root)
        win.resizable(False, False)
        win.grab_set()

        tk.Label(win, text=branding.FIRST_RUN_QUESTION, justify="left",
                 padx=16, pady=12).pack(anchor="w")
        dont_ask = tk.BooleanVar(value=False)
        tk.Checkbutton(win, text="Don't ask again", variable=dont_ask).pack(
            anchor="w", padx=16)

        def close(visit: bool):
            # The flag is written either way: the prompt is once-only, so the
            # "Don't ask again" box is really just an explicit confirmation.
            branding.mark_prompted()
            win.destroy()
            if visit:
                branding.open_site()

        row = tk.Frame(win)
        row.pack(pady=12)
        tk.Button(row, text="Yes", width=10,
                  command=lambda: close(True)).grid(row=0, column=0, padx=6)
        tk.Button(row, text="No", width=10,
                  command=lambda: close(False)).grid(row=0, column=1, padx=6)
        win.protocol("WM_DELETE_WINDOW", lambda: close(False))

    # ---- Build UI ------------------------------------------------------ #
    def _build(self):
        pad = {"padx": 12, "pady": 6}
        Frame = ctk.CTkFrame if _CTK else tk.Frame
        Label = ctk.CTkLabel if _CTK else tk.Label
        Button = ctk.CTkButton if _CTK else tk.Button
        Entry = ctk.CTkEntry if _CTK else tk.Entry

        top = Frame(self.root)
        top.pack(fill="x", **pad)
        Label(top, text="PDF file / folder:").grid(row=0, column=0, sticky="w")
        self.in_var = tk.StringVar()
        Entry(top, textvariable=self.in_var, width=380).grid(row=0, column=1, padx=6)
        Button(top, text="File", width=60, command=self._pick_file).grid(row=0, column=2)
        Button(top, text="Folder", width=60, command=self._pick_dir).grid(row=0, column=3)

        Label(top, text="Output folder:").grid(row=1, column=0, sticky="w", pady=6)
        self.out_var = tk.StringVar(value=str(Path.cwd() / "output_md"))
        Entry(top, textvariable=self.out_var, width=380).grid(row=1, column=1, padx=6)
        Button(top, text="Browse", width=60, command=self._pick_out).grid(row=1, column=2)

        # Options
        opts = Frame(self.root)
        opts.pack(fill="x", **pad)
        self.gpu_var = tk.BooleanVar(value=True)
        self.reflow_var = tk.BooleanVar(value=False)
        self.sym_var = tk.BooleanVar(value=True)
        self.fn_var = tk.BooleanVar(value=True)
        self.quran_var = tk.BooleanVar(value=False)
        self.striplfn_var = tk.BooleanVar(value=False)
        Check = ctk.CTkCheckBox if _CTK else tk.Checkbutton
        Check(opts, text="Use GPU", variable=self.gpu_var).grid(row=0, column=0, padx=8)
        Check(opts, text="Fix symbols (ﷺ)", variable=self.sym_var).grid(row=0, column=1, padx=8)
        Check(opts, text="Footnotes [^n]", variable=self.fn_var).grid(row=0, column=2, padx=8)
        Check(opts, text="Reflow lines", variable=self.reflow_var).grid(row=0, column=3, padx=8)
        Check(opts, text="Remove footnotes & ref numbers",
              variable=self.striplfn_var).grid(
            row=2, column=2, columnspan=2, padx=8, sticky="w")
        Check(opts, text="Correct Quran verses", variable=self.quran_var).grid(
            row=2, column=0, columnspan=2, padx=8, sticky="w")

        # OCR backend. Four free, fully-local engines.
        Label(opts, text="OCR engine:").grid(row=1, column=2, sticky="e")
        _saved = self.cfg.get("ocr", "fast")
        _saved = BACKEND_ALIASES.get(_saved, _saved)      # migrate old names
        if _saved not in BACKENDS:
            _saved = "fast"
        self.ocr_var = tk.StringVar(value=_saved)
        if _CTK:
            ctk.CTkOptionMenu(opts, values=BACKENDS, variable=self.ocr_var,
                              command=self._on_engine).grid(
                row=1, column=3, sticky="w")
        else:
            tk.OptionMenu(opts, self.ocr_var, *BACKENDS,
                          command=self._on_engine).grid(
                row=1, column=3, sticky="w")

        Label(opts, text="Languages:").grid(row=1, column=0, sticky="e", pady=6)
        self.lang_var = tk.StringVar(value="ar,en")
        Entry(opts, textvariable=self.lang_var, width=120).grid(row=1, column=1, sticky="w")

        # Page range — try a few pages before committing to a whole book.
        Label(opts, text="Pages from:").grid(row=5, column=0, sticky="e", pady=6)
        self.pfrom_var = tk.StringVar(value="")
        Entry(opts, textvariable=self.pfrom_var, width=60).grid(
            row=5, column=1, sticky="w", padx=6)
        Label(opts, text="to:").grid(row=5, column=2, sticky="e")
        self.pto_var = tk.StringVar(value="")
        Entry(opts, textvariable=self.pto_var, width=60).grid(
            row=5, column=3, sticky="w", padx=6)
        Label(opts, text="(leave empty for the whole book)").grid(
            row=6, column=1, columnspan=3, sticky="w", padx=6)

        # Arabic model choice (speed vs last-bit-of-accuracy)
        Label(opts, text="Arabic model:").grid(row=4, column=0, sticky="e", pady=6)
        self.armodel_var = tk.StringVar(
            value=self.cfg.get("arabic_model", "katib"))
        _models = ["katib", "qari"]
        if _CTK:
            ctk.CTkOptionMenu(opts, values=_models, variable=self.armodel_var,
                              command=self._on_model).grid(
                row=4, column=1, sticky="w")
        else:
            tk.OptionMenu(opts, self.armodel_var, *_models,
                          command=self._on_model).grid(row=4, column=1, sticky="w")
        self.model_hint = Label(opts, text=_MODEL_HELP[self.armodel_var.get()])
        self.model_hint.grid(row=4, column=2, columnspan=2, sticky="w", padx=6)

        # One-line explanation of the selected engine (updates on change).
        self.engine_hint = Label(opts, text=_ENGINE_HELP[_saved])
        self.engine_hint.grid(row=3, column=0, columnspan=4, sticky="w", padx=8, pady=(2, 6))

        # Start / Pause buttons
        btns = Frame(self.root)
        btns.pack(**pad)
        self.start_btn = Button(btns, text="Start / Resume", command=self._start)
        self.start_btn.grid(row=0, column=0, padx=6)
        self.pause_btn = Button(btns, text="Pause", command=self._pause)
        self.pause_btn.grid(row=0, column=1, padx=6)
        self.pause_btn.configure(state="disabled")

        self.bar = ttk.Progressbar(self.root, mode="determinate", maximum=100)
        self.bar.pack(fill="x", padx=12, pady=4)
        self.status = Label(self.root, text="Ready")
        self.status.pack(anchor="w", padx=12)

        # Log
        self.log = tk.Text(self.root, height=16, wrap="word")
        self.log.pack(fill="both", expand=True, padx=12, pady=(8, 2))
        self._say("Choose a file or folder, then click 'Start conversion'.")

        # Footer: plain text plus a clickable link to the author's site.
        foot = tk.Frame(self.root)
        foot.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(foot, text=f"Made by {branding.AUTHOR} · ").pack(side="left")
        link = tk.Label(foot, text=branding.SITE_LABEL,
                        fg="#1a6fd4", cursor="hand2")
        link.pack(side="left")
        f = link.cget("font")
        try:                      # underline it so it reads as a link
            import tkinter.font as tkfont
            uf = tkfont.Font(font=f)
            uf.configure(underline=True)
            link.configure(font=uf)
        except Exception:
            pass
        link.bind("<Button-1>", lambda _e: branding.open_site())

    # ---- File pickers -------------------------------------------------- #
    def _pick_file(self):
        p = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if p:
            self.in_var.set(p)

    def _pick_dir(self):
        p = filedialog.askdirectory()
        if p:
            self.in_var.set(p)

    def _pick_out(self):
        p = filedialog.askdirectory()
        if p:
            self.out_var.set(p)

    def _say(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    # ---- Run ----------------------------------------------------------- #
    def _collect(self, target: Path) -> list[Path]:
        if target.is_file() and target.suffix.lower() == ".pdf":
            return [target]
        if target.is_dir():
            return sorted(target.rglob("*.pdf"))
        return []

    def _on_model(self, *_):
        try:
            self.model_hint.configure(
                text=_MODEL_HELP.get(self.armodel_var.get(), ""))
        except Exception:
            pass

    def _on_engine(self, *_):
        """Update the one-line hint under the engine dropdown."""
        try:
            self.engine_hint.configure(
                text=_ENGINE_HELP.get(self.ocr_var.get(), ""))
        except Exception:
            pass

    def _pause(self):
        self.stop.set()
        self.pause_btn.configure(state="disabled")
        self.status.configure(text="Pausing after current page...")

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        src = Path(self.in_var.get().strip())
        pdfs = self._collect(src)
        if not pdfs:
            self._say("! No PDF files found.")
            return
        self.stop.clear()
        self.start_btn.configure(state="disabled")
        self.pause_btn.configure(state="normal")
        # Persist choices for next time
        _save_cfg({"ocr": self.ocr_var.get(),
                   "arabic_model": self.armodel_var.get()})
        cfg = EngineConfig(
            languages=[s.strip() for s in self.lang_var.get().split(",") if s.strip()],
            use_gpu=self.gpu_var.get(),
            ocr_backend=self.ocr_var.get(),
            arabic_model=self.armodel_var.get(),
            page_start=_as_int(self.pfrom_var.get()),
            page_end=_as_int(self.pto_var.get()),
            correct_quran=self.quran_var.get(),
            reflow_lines=self.reflow_var.get(),
            fix_symbols=self.sym_var.get(),
            fix_footnotes=self.fn_var.get(),
            strip_footnotes=self.striplfn_var.get(),
        )
        out_dir = Path(self.out_var.get().strip() or "output_md")
        self.worker = threading.Thread(
            target=self._run, args=(pdfs, out_dir, cfg), daemon=True
        )
        self.worker.start()

    def _run(self, pdfs, out_dir, cfg):
        from engine import PauseRequested

        def cb(i, total, stage):
            self.q.put(("progress", i, total, stage))

        try:
            engine = PdfToMarkdownEngine(cfg)
            self.q.put(("log", f"Device: {engine.device} | Files: {len(pdfs)}"))
            for idx, pdf in enumerate(pdfs, 1):
                # Resume: skip files already fully converted
                if cfg.resume and (out_dir / (pdf.stem + ".md")).exists():
                    self.q.put(("log", f"[{idx}/{len(pdfs)}] {pdf.name} - already done"))
                    continue
                self.q.put(("log", f"[{idx}/{len(pdfs)}] {pdf.name} ..."))
                try:
                    res = engine.convert(pdf, out_dir, progress_cb=cb,
                                         stop_event=self.stop)
                except PauseRequested as pr:
                    self.q.put(("paused", str(pr)))
                    return
                self.q.put(("log", f"    OK  {res.output.name} | {res.engine_used} "
                                   f"| {res.pages} pages | {res.seconds}s"))
            self.q.put(("done", str(out_dir)))
        except Exception as exc:
            self.q.put(("error", str(exc)))

    # ---- Update UI from the main thread -------------------------------- #
    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                kind = item[0]
                if kind == "progress":
                    _, i, total, stage = item
                    if total:
                        self.bar.configure(mode="determinate")
                        self.bar["value"] = i / total * 100
                        self.status.configure(text=f"{stage} - {i}/{total}")
                    else:
                        self.status.configure(text=stage)
                elif kind == "log":
                    self._say(item[1])
                elif kind == "done":
                    self.bar["value"] = 100
                    self.status.configure(text="Done")
                    self._say(f"Saved to: {item[1]}")
                    self.start_btn.configure(state="normal")
                    self.pause_btn.configure(state="disabled")
                elif kind == "paused":
                    self.status.configure(text="Paused - progress saved")
                    self._say(f"Paused: {item[1]}  (click Start / Resume to continue)")
                    self.start_btn.configure(state="normal")
                    self.pause_btn.configure(state="disabled")
                elif kind == "error":
                    self.status.configure(text="Error")
                    self._say(f"! Error: {item[1]}")
                    self.start_btn.configure(state="normal")
                    self.pause_btn.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(120, self._poll)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
