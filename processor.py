"""
processor.py — SepReformer inference wrapper for Sep-R-Ator.

Audio pipeline:
  - WAV input  → file-copy to temp dir (no re-encoding), run SepReformer
  - Other formats → decode in-memory with torchaudio, write temp WAV, run SepReformer
  - Outputs resampled back to original sample rate before saving.

SepReformer writes outputs to the same directory as the input sample:
  {stem}_out_0.wav, {stem}_out_1.wav, {stem}_in.wav
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
# Suppress torchaudio's "will change to torchcodec in 2.9" deprecation notice —
# the old API continues to work; we'll migrate when the change actually lands.
warnings.filterwarnings(
    "ignore",
    message=".*torchaudio.load_with_torchcodec.*",
    category=UserWarning,
)
import torchaudio

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).parent
SEPREFORMER_DIR = APP_DIR / "SepReformer"
MODEL_NAME = "SepReformer_Base_WSJ0"
SEPREFORMER_REPO = "https://github.com/dmlguq456/SepReformer"

# SepReformer_Base_WSJ0 operates at 8 kHz (WSJ0 dataset standard)
MODEL_SR = 8000


# ---------------------------------------------------------------------------
# Processor class
# ---------------------------------------------------------------------------

class SepReformerProcessor:

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    # ------------------------------------------------------------------
    # Installation helpers
    # ------------------------------------------------------------------

    def is_installed(self) -> bool:
        return (SEPREFORMER_DIR / "run.py").exists()

    def ensure_installed(self, progress_cb=None) -> None:
        """Clone SepReformer and install its dependencies if not present."""
        if self.is_installed():
            self._patch_py39_compat()  # safe to re-run; no-op if already patched
            return

        if progress_cb:
            progress_cb("Cloning SepReformer repository (this may take a while)…")

        subprocess.run(
            ["git", "clone", SEPREFORMER_REPO, str(SEPREFORMER_DIR)],
            check=True,
        )

        if progress_cb:
            progress_cb("Installing SepReformer dependencies…")

        # Install inference-only deps rather than SepReformer's pinned requirements.txt,
        # which contains exact CUDA/Linux-specific pins (e.g. networkx==3.4.1) that
        # don't resolve on Windows or non-CUDA systems.
        _SEPREFORMER_DEPS = [
            "librosa>=0.10.0",
            "soxr>=0.3.0",
            "scipy>=1.10.0",
            "loguru>=0.6.0",
            "mir-eval>=0.7",
            "matplotlib>=3.6.0",
            "pandas>=1.5.0",
            "scikit-learn>=1.1.0",
            "ptflops>=0.7",
            "ultralytics-thop>=2.0",  # actively maintained fork of unmaintained thop
            "torchinfo>=1.7",
            "tensorboard>=2.10",
            "tqdm>=4.60.0",
        ]
        subprocess.run(
            [sys.executable, "-m", "pip", "install"] + _SEPREFORMER_DEPS,
            check=True,
        )

        # Patch Python 3.9 compatibility: @dataclass(slots=True) requires 3.10+
        self._patch_py39_compat()

    def _write_cpu_wrapper(self) -> None:
        """Write a small shim into SepReformer/ for CPU-only torch.load safety.

        The heavy lifting (device creation) is handled by the source-level
        patch in _patch_py39_compat(). This wrapper only needs to ensure
        GPU-saved checkpoints load onto CPU via a torch.load redirect.
        """
        wrapper = SEPREFORMER_DIR / "_sep_r_ator_cpu_wrapper.py"
        wrapper.write_text(
            '"""CPU shim: patch torch.load for GPU-saved checkpoints, then run."""\n'
            "import runpy, sys, torch as _torch\n"
            "\n"
            "if not _torch.cuda.is_available():\n"
            "    _orig_load = _torch.load\n"
            "    def _cpu_load(f, map_location=None, **kwargs):\n"
            "        if map_location is None or (\n"
            "            isinstance(map_location, _torch.device) and map_location.type == 'cuda'\n"
            "        ) or (isinstance(map_location, str) and map_location.startswith('cuda')):\n"
            "            map_location = 'cpu'\n"
            "        # weights_only=False: SepReformer checkpoints contain optimizer/scheduler\n"
            "        # state; torch 2.6+ defaults to True which breaks loading.\n"
            "        kwargs.setdefault('weights_only', False)\n"
            "        return _orig_load(f, map_location=map_location, **kwargs)\n"
            "    _torch.load = _cpu_load\n"
            "\n"
            "sys.argv[0] = 'run.py'\n"
            "runpy.run_path('run.py', run_name='__main__')\n",
            encoding="utf-8",
        )

    def _patch_py39_compat(self) -> None:
        """Apply source-level patches to SepReformer for compatibility.

        1. CPU device patch — SepReformer unconditionally creates a CUDA device:
               gpuid = tuple(map(int, config["engine"]["gpuid"].split(',')))
               device = torch.device(f'cuda:{gpuid[0]}')
           Replace with a fallback that uses CPU when CUDA is unavailable.
           (Setting gpuid:'' in configs.yaml doesn't work — int('') → ValueError.)

        2. slots= patch (Python 3.9) — strip @dataclass(slots=True/False) since
           the slots= kwarg was added in Python 3.10.

        Both patches are idempotent (safe to re-run).
        """
        if not SEPREFORMER_DIR.exists():
            return

        # Always (re)write the CPU wrapper.
        self._write_cpu_wrapper()

        import re as _re

        # --- Source-level CPU device patch ---
        _CUDA_DEVICE_OLD = (
            "gpuid = tuple(map(int, config[\"engine\"][\"gpuid\"].split(',')))\n"
            "    device = torch.device(f'cuda:{gpuid[0]}')"
        )
        _CUDA_DEVICE_NEW = (
            "_gpuid_str = config[\"engine\"][\"gpuid\"]\n"
            "    if torch.cuda.is_available() and _gpuid_str.strip():\n"
            "        gpuid = tuple(map(int, _gpuid_str.split(',')))\n"
            "        device = torch.device(f'cuda:{gpuid[0]}')\n"
            "    else:\n"
            "        gpuid = (0,)\n"
            "        device = torch.device('cpu')"
        )

        for py_file in SEPREFORMER_DIR.rglob("*.py"):
            try:
                text = py_file.read_text(encoding="utf-8")
                changed = False

                # CPU device patch — replace hard-coded CUDA device creation
                if _CUDA_DEVICE_OLD in text:
                    text = text.replace(_CUDA_DEVICE_OLD, _CUDA_DEVICE_NEW)
                    changed = True

                # slots= patch (Python 3.9 compat)
                if "slots=" in text:
                    import sys as _sys
                    if _sys.version_info < (3, 10):
                        patched = _re.sub(r',\s*slots=(?:True|False)', '', text)
                        patched = _re.sub(r'slots=(?:True|False),\s*', '', patched)
                        patched = _re.sub(r'\(slots=(?:True|False)\)', '()', patched)
                        if patched != text:
                            text = patched
                            changed = True

                if changed:
                    py_file.write_text(text, encoding="utf-8")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _load_audio(self, path: str | Path) -> tuple[torch.Tensor, int]:
        """Load any supported audio format. Returns (mono float32 tensor, original_sr)."""
        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, sr

    def _resample(self, waveform: torch.Tensor, from_sr: int, to_sr: int) -> torch.Tensor:
        if from_sr == to_sr:
            return waveform
        return torchaudio.functional.resample(waveform, from_sr, to_sr)

    def _prepare_input(self, input_path: str | Path, tmp_dir: Path) -> Path:
        """
        Copy or convert input audio to a WAV file inside tmp_dir.

        - WAV files are copied as-is (no re-encoding).
        - All other formats are decoded in-memory and written as WAV.

        Returns the path to the WAV file in tmp_dir.
        """
        input_path = Path(input_path)
        suffix = input_path.suffix.lower()

        dest = tmp_dir / (input_path.stem + ".wav")

        if suffix == ".wav":
            # Straight file copy — no decoding/re-encoding
            shutil.copy2(input_path, dest)
        else:
            # Decode in-memory, write WAV
            waveform, sr = self._load_audio(input_path)
            torchaudio.save(str(dest), waveform.float(), sr)

        return dest

    # ------------------------------------------------------------------
    # SepReformer subprocess
    # ------------------------------------------------------------------

    def _run_sepreformer(self, input_wav: Path, progress_cb=None) -> list[Path]:
        """
        Run SepReformer inference on input_wav (must live inside a temp dir).

        SepReformer writes its outputs next to the input file:
            {stem}_out_0.wav, {stem}_out_1.wav

        Returns a sorted list of those output paths.
        """
        if progress_cb:
            progress_cb("Running SepReformer separation…")

        env = os.environ.copy()
        if self.device == "cpu":
            # Prevent CUDA use even if a GPU is present
            env["CUDA_VISIBLE_DEVICES"] = ""

        # Use the CPU wrapper when device is CPU so that SepReformer's hard-coded
        # cuda:0 device (from configs.yaml gpuid:'0') is transparently redirected
        # to CPU — avoids AssertionError on CPU-only PyTorch builds.
        entry_script = (
            "_sep_r_ator_cpu_wrapper.py"
            if self.device == "cpu"
            else "run.py"
        )

        subprocess.run(
            [
                sys.executable, entry_script,
                "--model", MODEL_NAME,
                "--engine-mode", "infer_sample",
                "--sample-file", str(input_wav),
            ],
            cwd=str(SEPREFORMER_DIR),
            env=env,
            check=True,
        )

        stem = input_wav.stem
        out_dir = input_wav.parent
        outputs = sorted(out_dir.glob(f"{stem}_out_*.wav"))
        if not outputs:
            raise RuntimeError(
                f"SepReformer produced no output files in {out_dir}. "
                "Check that the model weights are downloaded (git-lfs required)."
            )
        return outputs

    # ------------------------------------------------------------------
    # Mode A — separate a mixed file
    # ------------------------------------------------------------------

    def separate(
        self,
        input_path: str,
        output_dir: str,
        n_speakers: int = 2,
        progress_cb=None,
    ) -> list[str]:
        """
        Separate a single mixed audio file into per-speaker tracks.

        Returns a list of output file paths.
        """
        if progress_cb:
            progress_cb("Preparing audio…")

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Capture original sample rate for the final output
        _, original_sr = self._load_audio(input_path)

        with tempfile.TemporaryDirectory(prefix="sepr_a_") as tmpdir:
            tmp_dir = Path(tmpdir)
            input_wav = self._prepare_input(input_path, tmp_dir)

            sep_wavs = self._run_sepreformer(input_wav, progress_cb)

            if progress_cb:
                progress_cb("Saving separated tracks…")

            result_paths: list[str] = []
            for i, sep_wav in enumerate(sep_wavs, 1):
                waveform, sep_sr = torchaudio.load(str(sep_wav))
                waveform = self._resample(waveform, sep_sr, original_sr)
                out_path = output_dir / f"{input_path.stem}_speaker{i}.wav"
                torchaudio.save(str(out_path), waveform.float(), original_sr)
                result_paths.append(str(out_path))

        return result_paths

    # ------------------------------------------------------------------
    # Mode B — clean individual mic tracks (bleed removal)
    # ------------------------------------------------------------------

    def clean_tracks(
        self,
        track_paths: list[str],
        output_dir: str,
        progress_cb=None,
    ) -> list[str]:
        """
        Remove crosstalk bleed from a set of per-mic tracks.

        For each mic track:
          1. Load it (any format; WAV copied, others decoded).
          2. Feed it directly to SepReformer to separate the dominant speaker
             from the bleed.
          3. Keep the separated output most correlated with the original track
             (= primary speaker, least bleed).

        Returns a list of output file paths (one per input track).
        """
        if len(track_paths) < 2:
            raise ValueError("Need at least 2 tracks for bleed removal.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result_paths: list[str] = []

        for idx, track_path in enumerate(track_paths):
            track_path = Path(track_path)
            if progress_cb:
                progress_cb(
                    f"Processing track {idx + 1}/{len(track_paths)}: {track_path.name}…"
                )

            original_waveform, original_sr = self._load_audio(track_path)

            with tempfile.TemporaryDirectory(prefix=f"sepr_b{idx}_") as tmpdir:
                tmp_dir = Path(tmpdir)
                input_wav = self._prepare_input(track_path, tmp_dir)

                sep_wavs = self._run_sepreformer(input_wav, progress_cb=None)

                # Pick the separated output that best correlates with the input
                # (= the primary speaker on this mic)
                orig_np = original_waveform.squeeze().numpy()

                best_idx = 0
                best_corr = -2.0

                for si, sep_wav in enumerate(sep_wavs):
                    sw, sep_sr = torchaudio.load(str(sep_wav))
                    sw = self._resample(sw, sep_sr, original_sr)
                    sw_np = sw.squeeze().numpy()

                    min_len = min(len(orig_np), len(sw_np))
                    if min_len < 2:
                        continue
                    corr = float(
                        np.corrcoef(orig_np[:min_len], sw_np[:min_len])[0, 1]
                    )
                    if corr > best_corr:
                        best_corr = corr
                        best_idx = si

                best_wav_path = sep_wavs[best_idx]
                best_waveform, best_sr = torchaudio.load(str(best_wav_path))
                best_waveform = self._resample(best_waveform, best_sr, original_sr)

                out_path = output_dir / f"{track_path.stem}_clean.wav"
                torchaudio.save(str(out_path), best_waveform.float(), original_sr)
                result_paths.append(str(out_path))

        return result_paths
