# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This directory contains two independent color detection approaches:

- **`color_detector_lab.py`** — OpenCV-based classical color detection in LAB color space (primary tool)
- **`testing.ipynb`** — PyTorch/VGG16 transfer learning notebook for image classification (experimental)
- **`main.cpp` / `CMakeLists.txt`** — Minimal C++ stub (placeholder)

## Python: Running the Color Detector

Run every Python command through `uv run`. Do not call bare `python`, `python3`, or `pip`.

```bash
# Analyze an image file
uv run color_detector_lab.py image <path-to-image>

# Analyze with a region of interest
uv run color_detector_lab.py image <path> --roi X Y W H

# Display annotated result window
uv run color_detector_lab.py image <path> --show

# Real-time webcam detection (press Q to quit)
uv run color_detector_lab.py webcam

# Tune K-Means cluster count (default 4)
uv run color_detector_lab.py image <path> --clusters 6
```

## Python: Package Management

```bash
uv add <package>      # add dependency
uv remove <package>   # remove dependency
uv sync               # install/sync environment
uv run <tool>         # run pytest, ruff, mypy, etc.
```

## C++ Build

```bash
mkdir -p build && cd build
cmake ..
make
./test
```

## Architecture: `color_detector_lab.py`

Detection pipeline (BGR image → color name + confidence):

1. **Preprocess** (`_preprocess`) — optional ROI crop → Gaussian blur → morphological opening
2. **LAB conversion** — `cv2.COLOR_BGR2LAB`
3. **K-Means clustering** (`_dominant_lab_cluster`) — subsample to 8 000 pixels, run K-Means (`k=4` default), pick the largest cluster centroid; convert OpenCV's `[0,255]` LAB encoding back to standard `L=[0,100], a/b=[-128,127]`
4. **Color matching** (`_match_color`) — tolerance-weighted Euclidean distance against `COLOR_PROFILES` (12 empirically tuned entries); confidence via `exp(-0.6 * distance)`; returns `"Unknown"` below `min_confidence=0.55`

**Key types:**
- `ColorProfile` (frozen dataclass) — holds `lab_center`, `tolerance`, `hex_display` for one named color
- `LabColorDetector` — stateful detector; pre-builds numpy arrays from `COLOR_PROFILES` at init for vectorized matching
- Return dict keys: `color`, `confidence`, `lab_value` (`{L, a, b}`), `hex_color`, `all_matches` (top 3)

**Adding a new color:** append a `ColorProfile` to `COLOR_PROFILES` with empirically measured LAB center and per-channel tolerance.
