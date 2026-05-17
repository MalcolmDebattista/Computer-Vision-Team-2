"""
Interactive Principle Simulator
ARI 2129 - Topic 7: Histogram Equalisation and CLAHE

A Streamlit app that lets users explore how Histogram Equalisation (HE) and
Contrast Limited Adaptive Histogram Equalisation (CLAHE) transform image
intensities. The app covers:

  1. Interactive Demo  - load any image, compare Original / Global HE / CLAHE,
                         inspect histograms and the cumulative distribution
                         function (CDF), and adjust CLAHE parameters live.
  2. Worked Example    - a small toy image walked through the CDF derivation
                         step by step, exactly as it would be computed by hand.
  3. Failure Cases     - demonstrations of when HE and CLAHE go wrong
                         (noise amplification, halos, saturation).
  4. Theory            - the underlying mathematics in compact form.

Run with:    streamlit run app.py
""" 

from __future__ import annotations

import io
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HE & CLAHE Simulator",
    page_icon=":bar_chart:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Synthetic test images
# ---------------------------------------------------------------------------
# These are generated procedurally so the app has no external image dependency.
# Each one is designed to highlight a different aspect of histogram-based
# contrast enhancement.

def make_low_contrast(size: int = 256) -> np.ndarray:
    """A horizontal gradient whose intensities are squeezed into [80, 170].

    Because the dynamic range is compressed, this is the textbook input for
    histogram equalisation: stretching the histogram should recover detail.
    """
    gradient = np.linspace(0, 255, size, dtype=np.float32)
    image = np.tile(gradient, (size, 1))
    # Compress the [0, 255] range to roughly [80, 170].
    image = image * (90.0 / 255.0) + 80.0
    # Add a faint repeating pattern so equalisation has something to reveal.
    pattern = (np.sin(np.linspace(0, 6 * np.pi, size)) * 8.0).astype(np.float32)
    image = image + pattern[None, :]
    return np.clip(image, 0, 255).astype(np.uint8)


def make_dark_scene(size: int = 256) -> np.ndarray:
    """A mostly-dark image with a few bright structures - mimics a night photo."""
    rng = np.random.default_rng(seed=0)
    background = rng.normal(loc=30, scale=8, size=(size, size))
    # Add a couple of brighter circular regions.
    yy, xx = np.mgrid[0:size, 0:size]
    for cy, cx, r, val in [(70, 80, 25, 110), (180, 170, 35, 160), (130, 200, 18, 200)]:
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        background[mask] = val
    return np.clip(background, 0, 255).astype(np.uint8)


def make_bright_scene(size: int = 256) -> np.ndarray:
    """A washed-out, overexposed-looking image (intensities clustered near 255)."""
    rng = np.random.default_rng(seed=1)
    background = rng.normal(loc=220, scale=8, size=(size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    for cy, cx, r, val in [(80, 90, 30, 150), (170, 180, 40, 120), (200, 60, 20, 180)]:
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        background[mask] = val
    return np.clip(background, 0, 255).astype(np.uint8)


def make_bimodal(size: int = 256) -> np.ndarray:
    """Two well-separated intensity modes - foreground and background."""
    rng = np.random.default_rng(seed=2)
    image = np.where(
        rng.random((size, size)) < 0.5,
        rng.normal(60, 10, (size, size)),
        rng.normal(190, 10, (size, size)),
    )
    return np.clip(image, 0, 255).astype(np.uint8)


def make_noisy_flat(size: int = 256) -> np.ndarray:
    """A nearly-flat grey image with weak gaussian noise.

    Useful for showing how naive HE can dramatically amplify noise when there
    is no real structure to enhance.
    """
    rng = np.random.default_rng(seed=3)
    image = rng.normal(loc=128, scale=4, size=(size, size))
    return np.clip(image, 0, 255).astype(np.uint8)


SYNTHETIC_IMAGES = {
    "Low contrast gradient": make_low_contrast,
    "Dark scene": make_dark_scene,
    "Washed-out / bright": make_bright_scene,
    "Bimodal (two regions)": make_bimodal,
    "Almost-flat noisy patch": make_noisy_flat,
}


# ---------------------------------------------------------------------------
# Core image processing
# ---------------------------------------------------------------------------

def histogram_equalisation(image: np.ndarray) -> np.ndarray:
    """Apply standard (global) histogram equalisation.

    For colour input we work in the LAB colour space and equalise only the
    L (lightness) channel, which preserves the colour but rebalances the
    luminance distribution.
    """
    if image.ndim == 2:
        return cv2.equalizeHist(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def clahe(image: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    """Apply CLAHE.

    CLAHE divides the image into tiles (tile_size x tile_size) and equalises
    each tile separately. Histograms are clipped at clip_limit before the
    transformation, then the excess is redistributed - this prevents the
    over-amplification that plagues plain histogram equalisation.
    """
    operator = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    if image.ndim == 2:
        return operator.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = operator.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def luminance(image: np.ndarray) -> np.ndarray:
    """Return the channel we want to plot histograms / CDFs for.

    Greyscale images are returned unchanged; for colour we use the LAB
    L channel, which is exactly what HE / CLAHE operated on.
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 0]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_histogram(ax, image: np.ndarray, title: str) -> None:
    """Draw an intensity histogram for a single-channel image."""
    ax.hist(image.ravel(), bins=256, range=(0, 256), color="#444", alpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 256)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixel count")


def plot_cdf_overlay(ax, images: dict) -> None:
    """Overlay the CDFs of several single-channel images on one axis."""
    for label, image in images.items():
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
        cdf = hist.cumsum().astype(np.float64)
        cdf /= cdf[-1]  # normalise to [0, 1]
        ax.plot(cdf, label=label, linewidth=1.8)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Intensity r")
    ax.set_ylabel("CDF(r)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")


# ---------------------------------------------------------------------------
# Sidebar - global controls
# ---------------------------------------------------------------------------

st.sidebar.title("Controls")
st.sidebar.caption(
    "Pick an image, then tune the CLAHE parameters. "
    "Everything updates live."
)

image_choice = st.sidebar.selectbox(
    "Image source",
    list(SYNTHETIC_IMAGES.keys()) + ["Upload your own"],
    index=0,
)

if image_choice == "Upload your own":
    uploaded = st.sidebar.file_uploader(
        "Choose an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )
    keep_colour = st.sidebar.checkbox("Keep colour (process L channel)", value=True)
    if uploaded is None:
        st.sidebar.info("Upload an image to continue, or pick a synthetic source.")
        st.stop()
    pil_image = Image.open(uploaded)
    if keep_colour:
        pil_image = pil_image.convert("RGB")
        source_image = np.array(pil_image)
    else:
        pil_image = pil_image.convert("L")
        source_image = np.array(pil_image)
else:
    source_image = SYNTHETIC_IMAGES[image_choice]()

st.sidebar.markdown("---")
st.sidebar.subheader("CLAHE parameters")
clip_limit = st.sidebar.slider(
    "Clip limit",
    min_value=1.0,
    max_value=40.0,
    value=2.0,
    step=0.5,
    help="Maximum histogram-bin amplification per tile. "
         "Low values keep noise in check; high values behave more like plain HE.",
)
tile_size = st.sidebar.slider(
    "Tile grid size (NxN)",
    min_value=2,
    max_value=32,
    value=8,
    step=1,
    help="Image is divided into this many tiles per side. "
         "Smaller tiles are more local; larger tiles approach global HE.",
)


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title("Histogram Equalisation & CLAHE - Interactive Simulator")
st.markdown(
    "Topic 7 of the ARI 2129 Learning Pack. "
    "Use the controls on the left to choose an image and tune the CLAHE "
    "parameters. Hover any element for a short explanation."
)

tab_demo, tab_example, tab_failures, tab_theory = st.tabs(
    ["Interactive demo", "Worked example", "Failure cases", "Theory"]
)


# ---------------------------------------------------------------------------
# Tab 1 - Interactive demo
# ---------------------------------------------------------------------------
with tab_demo:
    # Apply the two transformations.
    he_image = histogram_equalisation(source_image)
    clahe_image = clahe(source_image, clip_limit=clip_limit, tile_size=tile_size)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader("Original")
        st.image(source_image, use_container_width=True, clamp=True)
    with col_b:
        st.subheader("Global HE")
        st.image(he_image, use_container_width=True, clamp=True)
    with col_c:
        st.subheader(f"CLAHE (clip={clip_limit}, tile={tile_size}x{tile_size})")
        st.image(clahe_image, use_container_width=True, clamp=True)

    st.markdown("### Histograms")
    fig_hist, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    plot_histogram(axes[0], luminance(source_image), "Original")
    plot_histogram(axes[1], luminance(he_image), "Global HE")
    plot_histogram(axes[2], luminance(clahe_image), "CLAHE")
    fig_hist.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    st.markdown("### Cumulative distribution functions")
    st.caption(
        "Global HE pushes the CDF towards a straight diagonal line - that is "
        "the mathematical definition of a uniform distribution. CLAHE keeps "
        "the CDF closer to the original because each tile is clipped before "
        "redistribution."
    )
    fig_cdf, ax_cdf = plt.subplots(figsize=(10, 4))
    plot_cdf_overlay(
        ax_cdf,
        {
            "Original": luminance(source_image),
            "Global HE": luminance(he_image),
            "CLAHE": luminance(clahe_image),
        },
    )
    # Reference: the target CDF for perfect equalisation is the diagonal.
    ax_cdf.plot([0, 255], [0, 1], "k--", linewidth=1, alpha=0.4, label="Ideal uniform")
    ax_cdf.legend(loc="lower right")
    st.pyplot(fig_cdf)
    plt.close(fig_cdf)


# ---------------------------------------------------------------------------
# Tab 2 - Worked example
# ---------------------------------------------------------------------------
with tab_example:
    st.markdown(
        "### Step-by-step CDF derivation on a tiny image\n"
        "Histogram equalisation is easier to grasp once you compute it by "
        "hand. The 8x8 patch below uses only 8 distinct intensities, so the "
        "arithmetic stays small."
    )

    # A deliberately low-contrast 8x8 toy image with values in {2..9}.
    toy = np.array(
        [
            [4, 4, 4, 4, 5, 5, 5, 5],
            [4, 4, 4, 5, 5, 5, 6, 6],
            [4, 4, 5, 5, 5, 6, 6, 6],
            [4, 5, 5, 5, 6, 6, 6, 7],
            [5, 5, 5, 6, 6, 6, 7, 7],
            [5, 5, 6, 6, 6, 7, 7, 8],
            [5, 6, 6, 6, 7, 7, 8, 8],
            [6, 6, 6, 7, 7, 8, 8, 9],
        ],
        dtype=np.uint8,
    )
    L = 16  # number of allowed levels in this toy example (4-bit image)
    n_pixels = toy.size

    st.markdown("**1. Pixel intensity matrix (8x8, 4-bit so levels are 0..15)**")
    st.dataframe(toy, use_container_width=False)

    # Step 2: histogram
    levels = np.arange(L)
    hist = np.bincount(toy.ravel(), minlength=L)
    st.markdown("**2. Histogram h(r) - how many pixels at each intensity**")
    hist_table = {f"r = {r}": int(hist[r]) for r in range(L) if hist[r] > 0}
    st.write(hist_table)

    # Step 3: probability density
    pdf = hist / n_pixels
    st.markdown("**3. Probability density p(r) = h(r) / N**, with N = " + str(n_pixels))
    pdf_table = {f"r = {r}": f"{pdf[r]:.4f}" for r in range(L) if hist[r] > 0}
    st.write(pdf_table)

    # Step 4: CDF
    cdf = pdf.cumsum()
    st.markdown("**4. CDF(r) = sum of p(k) for k <= r**")
    cdf_table = {f"r = {r}": f"{cdf[r]:.4f}" for r in range(L) if hist[r] > 0}
    st.write(cdf_table)

    # Step 5: equalisation mapping
    mapping = np.round((L - 1) * cdf).astype(np.uint8)
    st.markdown(
        "**5. Mapping T(r) = round((L-1) * CDF(r))**  - with L = "
        + str(L)
        + " levels"
    )
    mapping_table = {
        f"r = {r}": int(mapping[r]) for r in range(L) if hist[r] > 0
    }
    st.write(mapping_table)

    # Step 6: apply the mapping
    equalised = mapping[toy]
    st.markdown("**6. Apply T to every pixel - equalised image**")
    st.dataframe(equalised, use_container_width=False)

    # Side-by-side preview, scaled up
    col_x, col_y = st.columns(2)
    big = lambda arr: cv2.resize(
        (arr.astype(np.float32) * 255 / (L - 1)).astype(np.uint8),
        (256, 256),
        interpolation=cv2.INTER_NEAREST,
    )
    with col_x:
        st.image(big(toy), caption="Before (scaled up)", clamp=True)
    with col_y:
        st.image(big(equalised), caption="After (scaled up)", clamp=True)

    st.info(
        "Notice that the intensities now span almost the entire range and that "
        "intensities present in greater quantity get spread further apart. "
        "This is exactly the behaviour OpenCV's `cv2.equalizeHist` produces "
        "on 8-bit images - just with 256 levels instead of 16."
    )


# ---------------------------------------------------------------------------
# Tab 3 - Failure cases
# ---------------------------------------------------------------------------
with tab_failures:
    st.markdown(
        "### When histogram equalisation goes wrong\n"
        "The simulator's image picker on the left already includes images "
        "that produce textbook failure modes. The mini-experiments below "
        "isolate specific pitfalls."
    )

    # Failure 1: noise amplification on a near-flat image
    st.markdown("#### 1. Noise amplification on a near-flat image")
    st.caption(
        "When an image has very little real structure, global HE has nothing "
        "to enhance and instead stretches the small intensity differences "
        "(i.e. the noise) across the full range."
    )
    flat = make_noisy_flat()
    flat_he = histogram_equalisation(flat)
    flat_clahe_low = clahe(flat, clip_limit=2.0, tile_size=8)
    flat_clahe_high = clahe(flat, clip_limit=40.0, tile_size=8)

    cols = st.columns(4)
    cols[0].image(flat, caption="Original (flat + noise)", use_container_width=True, clamp=True)
    cols[1].image(flat_he, caption="Global HE - noise blown up", use_container_width=True, clamp=True)
    cols[2].image(flat_clahe_low, caption="CLAHE clip=2 - controlled", use_container_width=True, clamp=True)
    cols[3].image(flat_clahe_high, caption="CLAHE clip=40 - approaches HE", use_container_width=True, clamp=True)

    # Failure 2: bimodal image - global HE distorts the perception
    st.markdown("#### 2. Bimodal images - global HE can ruin the separation")
    st.caption(
        "On a clean bimodal image, global HE pushes the two peaks to the "
        "extreme ends of the range, which is sometimes useful but often "
        "destroys mid-tone detail and produces a posterised look."
    )
    bim = make_bimodal()
    bim_he = histogram_equalisation(bim)
    bim_clahe = clahe(bim, clip_limit=2.0, tile_size=8)
    cols = st.columns(3)
    cols[0].image(bim, caption="Original (bimodal)", use_container_width=True, clamp=True)
    cols[1].image(bim_he, caption="Global HE - mid-tones gone", use_container_width=True, clamp=True)
    cols[2].image(bim_clahe, caption="CLAHE - preserves local context", use_container_width=True, clamp=True)

    # Failure 3: tile-boundary artefacts at small tile sizes
    st.markdown("#### 3. CLAHE tile artefacts when the grid is too coarse")
    st.caption(
        "With very small tile grids on a smooth image, the local contrast "
        "varies sharply between tiles. CLAHE uses bilinear interpolation "
        "between tile centres to hide this, but at extreme settings you can "
        "still see tile-shaped patches in the result."
    )
    grad = make_low_contrast()
    grad_small = clahe(grad, clip_limit=40.0, tile_size=2)
    grad_big = clahe(grad, clip_limit=40.0, tile_size=16)
    cols = st.columns(3)
    cols[0].image(grad, caption="Original (smooth gradient)", use_container_width=True, clamp=True)
    cols[1].image(grad_small, caption="CLAHE tile=2 - patchy", use_container_width=True, clamp=True)
    cols[2].image(grad_big, caption="CLAHE tile=16 - smooth", use_container_width=True, clamp=True)


# ---------------------------------------------------------------------------
# Tab 4 - Theory
# ---------------------------------------------------------------------------
with tab_theory:
    st.markdown(
        r"""
### Why histogram equalisation works

For an image with intensities $r \in [0, L-1]$ and probability density
$p_r(r)$, define the transformation

$$
s = T(r) = (L - 1) \int_0^{r} p_r(w)\, dw = (L-1)\, \mathrm{CDF}(r).
$$

A standard probability result shows that the random variable $s$ produced
this way has an approximately uniform distribution on $[0, L-1]$. In a
*uniform* distribution every intensity is equally probable, so the histogram
of $s$ is - in the continuous limit - flat. In practice we work with a
discrete histogram, so the result is only approximately flat, but the
contrast-stretching effect is the same.

### Discrete algorithm

1. Compute the histogram $h(r)$ of the input.
2. Form the probability density $p(r) = h(r) / N$, where $N$ is the pixel count.
3. Compute the CDF $C(r) = \sum_{k=0}^{r} p(k)$.
4. Build the mapping $T(r) = \mathrm{round}\bigl((L-1)\, C(r)\bigr)$.
5. Replace every pixel value $r$ by $T(r)$.

### What CLAHE adds

CLAHE (Zuiderveld, 1994) addresses two weaknesses of global HE:

- **Locality.** Different parts of an image usually want different mappings.
  CLAHE divides the image into a grid of *tiles* and equalises each tile
  independently, then bilinearly interpolates between tile transformations
  so the boundaries are not visible.
- **Noise.** Plain HE can amplify noise in low-variance regions. CLAHE
  *clips* every tile histogram at a user-defined threshold (the **clip
  limit**) and redistributes the clipped pixels uniformly across all bins
  before computing the CDF. This caps how much any single intensity bin
  can be amplified.

### Parameter cheat sheet

| Parameter | Effect of a small value | Effect of a large value |
|---|---|---|
| `clipLimit` | Very gentle enhancement; near-identity on uniform regions | Behaviour approaches plain HE; noise amplification returns |
| `tileGridSize` | Very local; can produce a patchy result on smooth regions | Approaches global HE; loses the locality advantage |

### Practical defaults

Most OpenCV tutorials and academic CLAHE references start at
`clipLimit = 2.0` and `tileGridSize = (8, 8)`. The simulator opens with
exactly these values so the first thing you see matches the convention.
"""
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "ARI 2129 - Principles of Computer Vision for AI | Group project, "
    "Topic 7 (Histogram Equalisation and CLAHE). Built with Streamlit, "
    "OpenCV, NumPy and Matplotlib."
)
