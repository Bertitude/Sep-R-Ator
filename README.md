# Sep-R-Ator

Podcast crosstalk removal powered by [SepReformer](https://github.com/dmlguq456/SepReformer) (NeurIPS 2024).

---

## What it does

| Mode | Use case |
|------|----------|
| **Separate Mixed File** | One audio file contains all speakers (e.g. a Zoom/Teams recording). Splits it into individual per-speaker WAV tracks. |
| **Clean Individual Tracks** | Each speaker was recorded on their own mic but has bleed from the others. Removes that bleed and outputs cleaner per-track files. |

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10 | Tested on 3.10; 3.11 may work |
| [git](https://git-scm.com/) | For cloning SepReformer on first launch |
| [git-lfs](https://git-lfs.github.com/) | **Required** — the pretrained model weights are stored via Git LFS |
| NVIDIA GPU (optional) | CUDA 12 accelerates inference. Falls back to CPU if absent. On Surface Laptop Studio with RTX 3050 Ti, inference is fast. On Intel Iris Xe (CPU-only), expect ~5–10× real-time for a typical podcast clip. |

Install git-lfs before first run:

```bash
# Windows (winget)
winget install GitHub.GitLFS

# macOS
brew install git-lfs && git lfs install

# Ubuntu/Debian
sudo apt install git-lfs && git lfs install
```

---

## Installation

```bash
# 1. Clone this repo
git clone <this-repo-url>
cd Sep-R-Ator

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install app dependencies
pip install -r requirements.txt

# 4. Launch — SepReformer will be cloned automatically on first run
python app.py
```

First launch clones SepReformer (~350 MB including model weights) and installs its dependencies. This takes a few minutes; a progress dialog is shown.

---

## Usage

### Separate Mixed File (Tab A)

1. Click **Browse…** next to *Input file* and pick your podcast recording (WAV, MP3, FLAC, M4A, etc.).
2. Click **Browse…** next to *Output folder*.
3. Choose the number of speakers (2 or 3).
4. Click **Separate Speakers**.

Outputs: `{filename}_speaker1.wav`, `{filename}_speaker2.wav` — each track contains predominantly one speaker.

### Clean Individual Tracks (Tab B)

1. Click **Add…** and select all per-mic recordings (one file per microphone).
2. Pick an output folder.
3. Click **Remove Bleed**.

Outputs: `{original_name}_clean.wav` for each input track, with bleed from other mics suppressed.

---

## Notes

- The underlying model (`SepReformer_Base_WSJ0`) separates up to **2 speakers** and operates at **8 kHz**. Outputs are automatically resampled back to the original file's sample rate.
- WAV input files are passed to SepReformer without re-encoding. Other formats are decoded in-memory and written to a temporary WAV for SepReformer, then discarded.
- For GPU acceleration, select **GPU** in the device selector (top-right). If no CUDA GPU is detected the app falls back to CPU automatically.

---

## Model credit

> Ui-Hyeop Shin, Sangyoun Lee, Taehan Kim, Hyung-Min Park — *"Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation"* — NeurIPS 2024  
> https://github.com/dmlguq456/SepReformer
