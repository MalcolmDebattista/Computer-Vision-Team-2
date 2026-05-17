# Histogram Equalisation & CLAHE - Interactive Simulator

Interactive Principle Simulator for **ARI 2129 - Principles of Computer
Vision for AI**, Topic 7: *Histogram Equalisation and CLAHE*.

The simulator is a Streamlit app that lets users explore the behaviour of
Global Histogram Equalisation (HE) and Contrast Limited Adaptive Histogram
Equalisation (CLAHE) on both synthetic and uploaded images.

## What's in here

```
simulator/
├── app.py            # the Streamlit application
├── requirements.txt  # pinned dependencies
└── README.md         # this file
```

## How to run

From the `simulator/` folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will open a browser tab at `http://localhost:8501`. If it does
not open automatically, copy that URL into your browser.

> **Python version:** any 3.9+ should work. We have tested on Python 3.10
> and 3.11.

## Features

The app is organised into four tabs.

1. **Interactive demo** - load any image (one of five built-in synthetic
   scenes or an uploaded file) and compare the *Original*, *Global HE* and
   *CLAHE* outputs side by side. Histograms and a CDF overlay (with the
   ideal-uniform reference line) update live as you move the `clip limit`
   and `tile grid size` sliders.
2. **Worked example** - an 8x8 toy image is walked through the full CDF
   derivation step by step: histogram, probability density, cumulative
   distribution, the equalisation mapping `T(r) = round((L-1) * CDF(r))`,
   and the equalised matrix. This mirrors what you would compute by hand.
3. **Failure cases** - three isolated experiments showing where HE / CLAHE
   break down: noise amplification on near-flat images, posterisation on
   bimodal images under global HE, and visible tile artefacts when the
   CLAHE tile grid is too coarse.
4. **Theory** - compact mathematical background and a parameter cheat
   sheet.

## Controls (sidebar)

| Control | Range | Default | What it does |
|---|---|---|---|
| Image source | dropdown | Low contrast gradient | Pick a synthetic scene or upload your own image. |
| Keep colour | checkbox | on (for uploads) | If on, colour uploads are processed in LAB space - the L channel is equalised and chroma is preserved. |
| Clip limit | 1.0 - 40.0 | 2.0 | Maximum histogram-bin amplification per CLAHE tile. Low values = gentle, high values = closer to plain HE. |
| Tile grid size | 2 - 32 | 8 | Number of tiles per side. Low = very local enhancement, high = behaves more like global HE. |

The defaults (`clip = 2.0`, `tile = 8x8`) match the OpenCV and academic
references so the first view is the canonical CLAHE setup.

## Colour images

Histogram equalisation is undefined for vector-valued (RGB) pixels: applying
HE to each channel independently shifts the colours. We follow the standard
recipe instead: convert RGB to LAB, equalise the L channel, convert back.
That preserves hue and saturation while rebalancing luminance.

## Implementation notes

- All processing uses OpenCV (`cv2.equalizeHist`, `cv2.createCLAHE`),
  NumPy for histogram / CDF maths, and Matplotlib for the plots.
- Synthetic images are generated procedurally inside `app.py`, so the
  repository has no image-asset dependency.
- The Streamlit app is single-file by design - the assignment specifies
  `simulator/app.py` and we keep it that way.

## Troubleshooting

- **`ModuleNotFoundError: cv2`** - reinstall with
  `pip install opencv-python-headless`. The non-`-headless` variant
  sometimes clashes with Streamlit's GUI thread on Linux.
- **The browser tab doesn't open** - copy the URL printed in the terminal
  (e.g. `http://localhost:8501`) into your browser manually.
- **Slow first load** - Streamlit's first run compiles and caches a few
  things. Subsequent reloads are instant.
