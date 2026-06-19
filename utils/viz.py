import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# PlanetScope SuperDove band layout (1-indexed):
#  1=coastal_blue  2=blue  3=green_i  4=green_ii  5=yellow  6=red  7=rededge  8=nir
#
# Natural-color display: R=band6, G=band4, B=band2  →  0-based indices (5, 3, 1)
RGB_BANDS = (5, 3, 1)


def unnormalize(tensor, norm_factor):
    """
    Reverse the dataset's scalar normalization (x / norm_factor → x * norm_factor).

    Args:
        tensor: numpy array or torch tensor [..., H, W]
        norm_factor: float, from cfg['normalization_factor']

    Returns numpy array with same shape, in original reflectance scale.
    """
    arr = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
    return arr * norm_factor


def to_display_rgb(tensor, band_indices=RGB_BANDS):
    """
    Extract three bands from a [C, H, W] tensor and return a [H, W, 3] float32 array.

    Pulls from the reference image (first 8 channels). Band indices are 0-based.
    Default band_indices=(5,3,1) → red, green_ii, blue natural color.

    Args:
        tensor: numpy array or torch tensor [C, H, W]; C >= 8
        band_indices: tuple of three 0-based channel indices

    Returns [H, W, 3] float32 numpy array (NOT yet contrast-stretched).
    """
    arr = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
    rgb = np.stack([arr[b] for b in band_indices], axis=-1).astype(np.float32)
    return rgb


def percentile_stretch(img, lo=2, hi=98):
    """
    Clip and rescale [H, W, 3] image to [0, 1] using per-channel percentiles.

    Args:
        img: [H, W, 3] float32 numpy array
        lo, hi: percentile bounds (default 2/98 matches the notebook convention)

    Returns [H, W, 3] float32 array clipped to [0, 1].
    """
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        channel = img[..., c]
        p_lo = np.nanpercentile(channel, lo)
        p_hi = np.nanpercentile(channel, hi)
        out[..., c] = np.clip((channel - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)
    return out


def overlay_cam(img, cam, alpha=0.45, colormap="jet"):
    """
    Alpha-blend a spatial attention map onto an RGB image.

    Args:
        img: [H, W, 3] float32 in [0, 1] — output of percentile_stretch
        cam: [H, W] float32 — activation/saliency map, any range (will be normalized)
        alpha: blending weight for the heatmap overlay (0 = image only, 1 = heatmap only)
        colormap: matplotlib colormap name

    Returns [H, W, 3] float32 blended image.
    """
    cam_norm = cam - cam.min()
    denom = cam_norm.max()
    if denom > 1e-8:
        cam_norm = cam_norm / denom

    heatmap = cm.get_cmap(colormap)(cam_norm)[..., :3].astype(np.float32)
    blended = (1 - alpha) * img + alpha * heatmap
    return np.clip(blended, 0, 1)


def show_pair(tensor, norm_factor=None, title=None, ax=None):
    """
    Display the reference and query images side-by-side from a single [18, H, W] tensor.

    Args:
        tensor: [18, H, W] — output of BleachDataset.__getitem__
        norm_factor: if provided, unnormalizes before display
        title: optional suptitle string
        ax: if None, creates a new (1, 2) figure

    Returns (fig, axes).
    """
    arr = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
    if norm_factor is not None:
        arr = arr * norm_factor

    ref = percentile_stretch(to_display_rgb(arr[:8]))
    qry = percentile_stretch(to_display_rgb(arr[8:16]))

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    else:
        fig, axes = ax[0].get_figure(), ax

    axes[0].imshow(ref)
    axes[0].set_title("Reference")
    axes[0].axis("off")

    axes[1].imshow(qry)
    axes[1].set_title("Query")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=10)

    plt.tight_layout()
    return fig, axes
