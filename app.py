"""
app.py — Sep-R-Ator GUI
Podcast crosstalk removal powered by SepReformer (NeurIPS 2024).

Two modes:
  Tab A — Separate Mixed File: split one mixed audio file into per-speaker tracks.
  Tab B — Clean Individual Tracks: remove mic bleed from per-mic recordings.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from processor import SepReformerProcessor

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Accepted audio extensions for file dialogs
_AUDIO_TYPES = (
    ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.aiff *.aif *.opus"),
    ("All files", "*.*"),
)


# ---------------------------------------------------------------------------
# First-run setup modal
# ---------------------------------------------------------------------------

class SetupDialog(ctk.CTkToplevel):
    """Blocking modal shown on first launch while SepReformer is being installed."""

    def __init__(self, parent: "App") -> None:
        super().__init__(parent)
        self.title("First-Time Setup")
        self.geometry("420x180")
        self.resizable(False, False)
        self.grab_set()
        self.lift()  # prevents white-flash on Windows dark mode (customtkinter #2469)
        self.protocol("WM_DELETE_WINDOW", lambda: None)  # prevent close

        ctk.CTkLabel(
            self,
            text="Setting up SepReformer",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(24, 4))

        self._status = ctk.CTkLabel(self, text="Starting…", text_color="gray")
        self._status.pack(pady=4)

        self._bar = ctk.CTkProgressBar(self, mode="indeterminate", width=340)
        self._bar.pack(pady=12)
        self._bar.start()

    def update_status(self, msg: str) -> None:
        self._status.configure(text=msg)
        self.update_idletasks()

    def close(self) -> None:
        self._bar.stop()
        self.grab_release()
        self.destroy()


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class App(ctk.CTk):

    def __init__(self) -> None:
        super().__init__()
        self.title("Sep-R-Ator")
        self.geometry("700x560")
        self.resizable(False, False)

        self.processor = SepReformerProcessor(device="auto")

        self._build_ui()

        # Kick off first-run setup after the window is visible
        self.after(100, self._maybe_setup)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Header
        header = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(16, 0))

        ctk.CTkLabel(
            header,
            text="Sep-R-Ator",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(side="left")

        ctk.CTkLabel(
            header,
            text="Podcast Crosstalk Removal",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        ).pack(side="left", padx=(10, 0), pady=(6, 0))

        # Device selector (top-right)
        self._device_var = ctk.StringVar(value="Auto")
        device_frame = ctk.CTkFrame(header, fg_color="transparent")
        device_frame.pack(side="right")
        ctk.CTkLabel(device_frame, text="Device:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 6))
        ctk.CTkSegmentedButton(
            device_frame,
            values=["Auto", "CPU", "GPU"],
            variable=self._device_var,
            width=160,
            command=self._on_device_change,
        ).pack(side="left")
        # Shows what "Auto" actually resolved to
        _resolved = self.processor.device.upper()
        self._device_status = ctk.CTkLabel(
            device_frame,
            text=f"({_resolved})",
            font=ctk.CTkFont(size=11),
            text_color="#4CAF50" if _resolved == "CUDA" else "gray",
        )
        self._device_status.pack(side="left", padx=(6, 0))

        # Tab view
        self._tabs = ctk.CTkTabview(self, width=660, height=320)
        self._tabs.pack(padx=20, pady=12, fill="both")

        self._tabs.add("Separate Mixed File")
        self._tabs.add("Clean Individual Tracks")

        self._build_tab_a(self._tabs.tab("Separate Mixed File"))
        self._build_tab_b(self._tabs.tab("Clean Individual Tracks"))

        # Shared bottom panel (progress + results)
        bottom = ctk.CTkFrame(self, corner_radius=8)
        bottom.pack(padx=20, pady=(0, 16), fill="x")

        self._status_label = ctk.CTkLabel(
            bottom, text="Ready.", font=ctk.CTkFont(size=12), text_color="gray"
        )
        self._status_label.pack(anchor="w", padx=14, pady=(10, 2))

        self._progress = ctk.CTkProgressBar(bottom, mode="indeterminate")
        self._progress.pack(fill="x", padx=14, pady=(0, 6))
        self._progress.set(0)

        self._result_frame = ctk.CTkScrollableFrame(bottom, height=80, label_text="Output Files")
        self._result_frame.pack(fill="x", padx=14, pady=(0, 10))

        self._open_btn = ctk.CTkButton(
            bottom,
            text="Open Output Folder",
            width=180,
            state="disabled",
            command=self._open_output_folder,
        )
        self._open_btn.pack(anchor="e", padx=14, pady=(0, 10))

        self._last_output_dir: str | None = None

    # ---- Tab A -----------------------------------------------------------

    def _build_tab_a(self, tab: ctk.CTkFrame) -> None:
        tab.grid_columnconfigure(1, weight=1)

        # Input file
        ctk.CTkLabel(tab, text="Input file:").grid(row=0, column=0, sticky="w", padx=12, pady=(16, 6))
        self._a_file_label = ctk.CTkLabel(tab, text="No file selected", text_color="gray", anchor="w")
        self._a_file_label.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkButton(tab, text="Browse…", width=90, command=self._a_pick_file).grid(
            row=0, column=2, padx=(0, 12), pady=(16, 6)
        )

        # Output folder
        ctk.CTkLabel(tab, text="Output folder:").grid(row=1, column=0, sticky="w", padx=12, pady=6)
        self._a_outdir_label = ctk.CTkLabel(tab, text="No folder selected", text_color="gray", anchor="w")
        self._a_outdir_label.grid(row=1, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkButton(tab, text="Browse…", width=90, command=self._a_pick_outdir).grid(
            row=1, column=2, padx=(0, 12), pady=6
        )

        # Speakers — SepReformer_Base_WSJ0 is a 2-source model; label is informational only
        ctk.CTkLabel(tab, text="Speakers:").grid(row=2, column=0, sticky="w", padx=12, pady=6)
        ctk.CTkLabel(tab, text="2  (model maximum)", text_color="gray").grid(
            row=2, column=1, sticky="w", pady=6
        )

        # Process button
        self._a_btn = ctk.CTkButton(
            tab,
            text="Separate Speakers",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=self._on_separate_click,
        )
        self._a_btn.grid(row=3, column=0, columnspan=3, pady=(20, 10), padx=12, sticky="ew")

        self._a_file: str | None = None
        self._a_outdir: str | None = None


    # ---- Tab B -----------------------------------------------------------

    def _build_tab_b(self, tab: ctk.CTkFrame) -> None:
        tab.grid_columnconfigure(0, weight=1)

        # Track list
        ctk.CTkLabel(tab, text="Mic tracks (one per speaker):").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(16, 4)
        )

        self._b_listbox_frame = ctk.CTkScrollableFrame(tab, height=90)
        self._b_listbox_frame.grid(row=1, column=0, padx=(12, 4), pady=4, sticky="ew")

        btn_col = ctk.CTkFrame(tab, fg_color="transparent")
        btn_col.grid(row=1, column=1, padx=(0, 12), pady=4, sticky="n")
        ctk.CTkButton(btn_col, text="Add…", width=80, command=self._b_add_tracks).pack(pady=(0, 4))
        ctk.CTkButton(btn_col, text="Remove", width=80, command=self._b_remove_track).pack()

        # Output folder
        ctk.CTkLabel(tab, text="Output folder:").grid(row=2, column=0, sticky="w", padx=12, pady=6)
        outdir_row = ctk.CTkFrame(tab, fg_color="transparent")
        outdir_row.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12)
        outdir_row.grid_columnconfigure(0, weight=1)
        self._b_outdir_label = ctk.CTkLabel(outdir_row, text="No folder selected", text_color="gray", anchor="w")
        self._b_outdir_label.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(outdir_row, text="Browse…", width=90, command=self._b_pick_outdir).grid(row=0, column=1, padx=(8, 0))

        # Process button
        self._b_btn = ctk.CTkButton(
            tab,
            text="Remove Bleed",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=self._on_clean_click,
        )
        self._b_btn.grid(row=4, column=0, columnspan=2, pady=(12, 10), padx=12, sticky="ew")

        self._b_tracks: list[str] = []
        self._b_track_labels: list[ctk.CTkLabel] = []
        self._b_outdir: str | None = None

    # ------------------------------------------------------------------
    # Tab A actions
    # ------------------------------------------------------------------

    def _a_pick_file(self) -> None:
        path = filedialog.askopenfilename(title="Select podcast audio file", filetypes=_AUDIO_TYPES)
        if path:
            self._a_file = path
            self._a_file_label.configure(text=Path(path).name, text_color="white")

    def _a_pick_outdir(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self._a_outdir = path
            self._a_outdir_label.configure(text=path, text_color="white")

    def _on_separate_click(self) -> None:
        if not self._a_file:
            self._show_error("Please select an input audio file.")
            return
        if not self._a_outdir:
            self._show_error("Please select an output folder.")
            return

        self._run_in_thread(
            self._a_btn,
            self.processor.separate,
            self._a_file,
            self._a_outdir,
        )

    # ------------------------------------------------------------------
    # Tab B actions
    # ------------------------------------------------------------------

    def _b_add_tracks(self) -> None:
        paths = filedialog.askopenfilenames(title="Select mic track files", filetypes=_AUDIO_TYPES)
        for p in paths:
            if p not in self._b_tracks:
                self._b_tracks.append(p)
                lbl = ctk.CTkLabel(
                    self._b_listbox_frame,
                    text=Path(p).name,
                    anchor="w",
                    cursor="hand2",
                )
                lbl.pack(fill="x", padx=4, pady=1)
                self._b_track_labels.append(lbl)

    def _b_remove_track(self) -> None:
        if self._b_tracks:
            self._b_tracks.pop()
            lbl = self._b_track_labels.pop()
            lbl.destroy()

    def _b_pick_outdir(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self._b_outdir = path
            self._b_outdir_label.configure(text=path, text_color="white")

    def _on_clean_click(self) -> None:
        if len(self._b_tracks) < 2:
            self._show_error("Add at least 2 mic tracks for bleed removal.")
            return
        if not self._b_outdir:
            self._show_error("Please select an output folder.")
            return

        self._run_in_thread(
            self._b_btn,
            self.processor.clean_tracks,
            list(self._b_tracks),
            self._b_outdir,
        )

    # ------------------------------------------------------------------
    # Device selector
    # ------------------------------------------------------------------

    def _on_device_change(self, value: str) -> None:
        import torch as _torch
        if value == "GPU":
            if _torch.cuda.is_available():
                self.processor.device = "cuda"
            else:
                self._show_error("No CUDA GPU detected — falling back to CPU.")
                self.processor.device = "cpu"
        elif value == "CPU":
            self.processor.device = "cpu"
        else:  # Auto
            self.processor.device = "cuda" if _torch.cuda.is_available() else "cpu"
        resolved = self.processor.device.upper()
        self._device_status.configure(
            text=f"({resolved})",
            text_color="#4CAF50" if resolved == "CUDA" else "gray",
        )

    # ------------------------------------------------------------------
    # Generic threaded runner
    # ------------------------------------------------------------------

    def _run_in_thread(self, trigger_btn: ctk.CTkButton, fn, *args) -> None:
        """Disable button, animate progress bar, call fn(*args) in a thread."""
        self._clear_results()
        self._set_processing(True, trigger_btn)

        def worker():
            try:
                def cb(msg: str, fraction=None):
                    self.after(0, lambda m=msg, f=fraction: self._set_progress(m, f))

                result = fn(*args, cb)
                self.after(0, lambda r=result: self._on_done(r))
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_error(e))
            finally:
                self.after(0, lambda: self._set_processing(False, trigger_btn))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # UI state helpers
    # ------------------------------------------------------------------

    def _set_processing(self, active: bool, btn: ctk.CTkButton) -> None:
        if active:
            btn.configure(state="disabled")
            self._progress.configure(mode="indeterminate")
            self._progress.start()
        else:
            btn.configure(state="normal")
            self._progress.stop()
            self._progress.configure(mode="determinate")
            self._progress.set(1)

    def _set_progress(self, msg: str, fraction=None) -> None:
        """Update status text and, if fraction is given, switch bar to determinate."""
        self._status_label.configure(text=msg, text_color="gray")
        if fraction is not None:
            self._progress.stop()
            self._progress.configure(mode="determinate")
            self._progress.set(max(0.0, min(1.0, fraction)))

    def _set_status(self, msg: str, color: str = "gray") -> None:
        self._status_label.configure(text=msg, text_color=color)

    def _show_error(self, msg: str) -> None:
        self._set_status(f"Error: {msg}", color="#FF6B6B")

    def _clear_results(self) -> None:
        for widget in self._result_frame.winfo_children():
            widget.destroy()
        self._open_btn.configure(state="disabled")
        self._last_output_dir = None

    def _on_done(self, output_paths: list[str]) -> None:
        self._set_status(f"Done — {len(output_paths)} file(s) saved.", color="#6BCB77")

        for path in output_paths:
            row = ctk.CTkFrame(self._result_frame, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=Path(path).name, anchor="w").pack(side="left", padx=4)

        if output_paths:
            self._last_output_dir = str(Path(output_paths[0]).parent)
            self._open_btn.configure(state="normal")

    def _on_error(self, exc: Exception) -> None:
        self._show_error(str(exc))
        messagebox.showerror("Processing Error", str(exc))

    def _open_output_folder(self) -> None:
        if not self._last_output_dir:
            return
        if sys.platform == "win32":
            os.startfile(self._last_output_dir)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", self._last_output_dir])
        else:
            subprocess.Popen(["xdg-open", self._last_output_dir])

    # ------------------------------------------------------------------
    # First-run setup
    # ------------------------------------------------------------------

    def _maybe_setup(self) -> None:
        if self.processor.is_installed():
            self.processor._patch_py39_compat()  # no-op on 3.10+; fixes slots=True on 3.9
            self._set_status("Ready.")
            return

        dialog = SetupDialog(self)

        def do_setup():
            try:
                self.processor.ensure_installed(
                    progress_cb=lambda msg: self.after(0, lambda m=msg: dialog.update_status(m))
                )
                self.after(0, dialog.close)
                self.after(0, lambda: self._set_status("Ready."))
            except Exception as exc:
                self.after(0, dialog.close)
                self.after(
                    0,
                    lambda e=exc: messagebox.showerror(
                        "Setup Failed",
                        f"Could not install SepReformer:\n{e}\n\n"
                        "Make sure git and git-lfs are installed, then restart the app.",
                    ),
                )

        threading.Thread(target=do_setup, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = App()
    app.mainloop()
