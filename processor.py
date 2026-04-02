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
from pathlib import Path

import numpy as np
import torch
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

# Formats torchaudio can load that are NOT WAV — need intermediate WAV for these
_NON_WAV_SUFFIXES = {".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif"}


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
            "thop>=0.1",
            "torchinfo>=1.7",
            "tensorboard>=2.10",
            "tqdm>=4.60.0",
        ]
        subprocess.run(
            [sys.executable, "-m", "pip", "install"] + _SEPREFORMER_DEPS,
            check=True,
        )

        if progress_cb:
            progress_cb("Setup complete.")

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

        subprocess.run(
            [
                sys.executable, "run.py",
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
