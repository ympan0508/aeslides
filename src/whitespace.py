"""
Whitespace metric for rendered screenshots.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import warnings

import cv2
import numpy as np


def compute_whitespace_metric(
    image_bytes: bytes,
    *,
    box_ksize_h: int = 201,
    box_ksize_v: int = 151,
    std_clip: float = 50.0,
    gaussian_ksize: Tuple[int, int] = (21, 21),
    binary_threshold: float = 0.05,
    crop_ratios: Tuple[float, float, float, float] = (0.15, 0.1, 0.1, 0.1),
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute whitespace / content-density metrics from raw image bytes.

    Parameters
    ----------
    image_bytes:
        Raw PNG or JPEG bytes.
    box_ksize_h:
        Horizontal kernel size for the local box filter used to compute
        local mean and local variance.
    box_ksize_v:
        Vertical kernel size for the local box filter used to compute
        local mean and local variance.
    std_clip:
        Maximum local standard deviation used for clipping before
        normalization to [0, 1]. Larger values make the metric less
        sensitive; smaller values make it more sensitive.
    gaussian_ksize:
        Gaussian blur kernel size applied before local variance estimation.
        Must be a pair of positive odd integers.
    binary_threshold:
        Threshold in [0, 1]. Pixels with normalized local std-dev above this
        value are considered content; others are considered whitespace.
    crop_ratios:
        Fractions of the image to crop from (top, bottom, left, right) before
        computing the ROI content ratio.
    save_path:
        Optional path for saving a debug visualization. If None, no image is
        written.

    Returns
    -------
    dict
        A dictionary with:
        - content_ratio_full: float
        - content_ratio_crop: float

        For debugging / inspection, this implementation also returns:
        - resolution: (width, height)
        - normalized_frequency_map: np.ndarray of shape (H, W), float32 in [0, 1]
        - binary_mask: np.ndarray of shape (H, W), uint8 in {0, 255}
        - crop_slice: tuple[slice, slice]

    Raises
    ------
    ValueError
        If the input or parameters are invalid, or decoding fails.
    RuntimeError
        If the core computation fails unexpectedly.
    """
    _validate_inputs(
        image_bytes=image_bytes,
        box_ksize_h=box_ksize_h,
        box_ksize_v=box_ksize_v,
        std_clip=std_clip,
        gaussian_ksize=gaussian_ksize,
        binary_threshold=binary_threshold,
        crop_ratios=crop_ratios,
    )

    img = _decode_image_bytes(image_bytes)

    try:
        norm_freq = _compute_local_variance_map(
            img=img,
            box_ksize_h=box_ksize_h,
            box_ksize_v=box_ksize_v,
            gaussian_ksize=gaussian_ksize,
            std_clip=std_clip,
        )
        metrics = _compute_metrics_data(
            norm_freq=norm_freq,
            threshold=binary_threshold,
            crop_ratios=crop_ratios,
        )
    except Exception as exc:
        raise RuntimeError(f"whitespace metric computation failed: {exc}") from exc

    if save_path:
        _save_visualization(
            img=img,
            norm_freq=norm_freq,
            metrics_data=metrics,
            threshold=binary_threshold,
            crop_ratios=crop_ratios,
            std_clip=std_clip,
            box_ksize_h=box_ksize_h,
            box_ksize_v=box_ksize_v,
            gaussian_ksize=gaussian_ksize,
            save_path=save_path,
        )

    return {
        "content_ratio_full": metrics["content_ratio_full"],
        "content_ratio_crop": metrics["content_ratio_crop"],
        "resolution": metrics["resolution"],
        "normalized_frequency_map": norm_freq,
        "binary_mask": metrics["binary_mask"],
        "crop_slice": metrics["crop_slice"],
    }


def _validate_inputs(
    *,
    image_bytes: bytes,
    box_ksize_h: int,
    box_ksize_v: int,
    std_clip: float,
    gaussian_ksize: Tuple[int, int],
    binary_threshold: float,
    crop_ratios: Tuple[float, float, float, float],
) -> None:
    if not image_bytes:
        raise ValueError("image_bytes must be non-empty raw PNG/JPEG bytes")

    if box_ksize_h <= 0 or box_ksize_v <= 0:
        raise ValueError("box_ksize_h and box_ksize_v must be positive integers")

    if std_clip < 0:
        raise ValueError("std_clip must be >= 0")

    if (
        not isinstance(gaussian_ksize, tuple)
        or len(gaussian_ksize) != 2
        or any(not isinstance(x, int) or x <= 0 for x in gaussian_ksize)
    ):
        raise ValueError("gaussian_ksize must be a tuple of two positive integers")

    if gaussian_ksize[0] % 2 == 0 or gaussian_ksize[1] % 2 == 0:
        raise ValueError("gaussian_ksize values must be odd integers")

    if not (0.0 <= binary_threshold <= 1.0):
        raise ValueError("binary_threshold must be in [0, 1]")

    if (
        not isinstance(crop_ratios, tuple)
        or len(crop_ratios) != 4
        or any(not isinstance(x, (int, float)) for x in crop_ratios)
    ):
        raise ValueError("crop_ratios must be a 4-tuple: (top, bottom, left, right)")

    t, b, l, r = crop_ratios
    if any(x < 0 or x >= 1 for x in (t, b, l, r)):
        raise ValueError("each crop ratio must satisfy 0 <= ratio < 1")

    if t + b >= 1 or l + r >= 1:
        raise ValueError("crop_ratios remove the entire image; require top+bottom < 1 and left+right < 1")


def _decode_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """
    Decode raw PNG/JPEG bytes into an OpenCV BGR image.

    Returns
    -------
    np.ndarray
        Shape (H, W, 3), dtype uint8, BGR order.
    """
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode image_bytes into an OpenCV image")
    return img


def _compute_local_variance_map(
    img: np.ndarray,
    box_ksize_h: int,
    box_ksize_v: int,
    gaussian_ksize: Tuple[int, int],
    std_clip: float,
) -> np.ndarray:
    """
    Compute a normalized local standard-deviation map in [0, 1].

    The map acts as a proxy for local visual complexity:
    - low values -> likely whitespace / blank area
    - high values -> likely content
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray_f32 = gray.astype(np.float32)

    kh = box_ksize_h if box_ksize_h % 2 == 1 else box_ksize_h + 1
    kv = box_ksize_v if box_ksize_v % 2 == 1 else box_ksize_v + 1
    ksize = (kh, kv)

    blurred = cv2.GaussianBlur(gray_f32, gaussian_ksize, 0)
    blurred_sq = blurred ** 2

    local_mean = cv2.boxFilter(
        blurred,
        ddepth=-1,
        ksize=ksize,
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    local_mean_sq = cv2.boxFilter(
        blurred_sq,
        ddepth=-1,
        ksize=ksize,
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )

    variance = np.maximum(local_mean_sq - local_mean ** 2, 0.0)
    std_dev = np.sqrt(variance)

    if std_clip <= 1e-8:
        return np.zeros_like(std_dev, dtype=np.float32)

    clipped = np.clip(std_dev, 0.0, std_clip)
    return (clipped / std_clip).astype(np.float32)


def _compute_metrics_data(
    norm_freq: np.ndarray,
    threshold: float,
    crop_ratios: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """
    Threshold the normalized map and compute content ratios.

    Returns
    -------
    dict
        Includes the binary mask, cropped mask, crop slice, ratios, and resolution.
    """
    h, w = norm_freq.shape
    binary_mask = (norm_freq > threshold).astype(np.uint8) * 255

    t, b, l, r = crop_ratios
    s_y, e_y = int(h * t), int(h * (1 - b))
    s_x, e_x = int(w * l), int(w * (1 - r))

    crop_slice = (slice(s_y, e_y), slice(s_x, e_x))
    cropped_binary = binary_mask[crop_slice]

    content_ratio_full = float(np.mean(binary_mask > 0))
    content_ratio_crop = float(np.mean(cropped_binary > 0))

    return {
        "binary_mask": binary_mask,
        "cropped_binary": cropped_binary,
        "crop_slice": crop_slice,
        "content_ratio_full": content_ratio_full,
        "content_ratio_crop": content_ratio_crop,
        "resolution": (w, h),
    }


def _save_visualization(
    img: np.ndarray,
    norm_freq: np.ndarray,
    metrics_data: Dict[str, Any],
    threshold: float,
    crop_ratios: Tuple[float, float, float, float],
    std_clip: float,
    box_ksize_h: int,
    box_ksize_v: int,
    gaussian_ksize: Tuple[int, int],
    save_path: str,
) -> None:
    """
    Save a debug visualization.

    This is optional and never affects the metric result.
    Any visualization failure is downgraded to a warning.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        binary_mask = metrics_data["binary_mask"]
        cropped_binary = metrics_data["cropped_binary"]
        crop_slice = metrics_data["crop_slice"]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cmap = matplotlib.colormaps.get_cmap("RdYlGn")
        mask_rgb = (cmap(norm_freq)[:, :, :3] * 255).astype(np.uint8)

        overlay = cv2.addWeighted(
            cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
            0.3,
            mask_rgb,
            0.7,
            0,
        )
        cropped_overlay = overlay[crop_slice]

        fig = Figure(figsize=(42, 6))
        FigureCanvasAgg(fig)

        gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.6])
        axes = [fig.add_subplot(gs[0, i]) for i in range(7)]

        titles = [
            "Original",
            "Freq Map",
            "Binary (Full)",
            "Overlay (Full)",
            "Binary (Crop)",
            "Overlay (Crop)",
        ]
        datas = [
            img_rgb,
            norm_freq,
            binary_mask,
            overlay,
            cropped_binary,
            cropped_overlay,
        ]
        cmaps = [None, "RdYlGn", "gray", None, "gray", None]

        for i in range(6):
            axes[i].set_title(titles[i], fontsize=14, fontweight="bold")
            im = axes[i].imshow(
                datas[i],
                cmap=cmaps[i],
                vmin=0,
                vmax=1 if i == 1 else None,
            )
            axes[i].axis("off")
            if i == 1:
                fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        axes[6].axis("off")
        info_text = (
            f"Metrics\n\n"
            f"Res: {metrics_data['resolution']}\n"
            f"Thresh: {threshold:.2f}\n"
            f"Crop: {crop_ratios}\n"
            f"Clip: {std_clip}\n"
            f"Box ksize: ({box_ksize_h}, {box_ksize_v})\n"
            f"Gaussian ksize: {gaussian_ksize}\n"
            f"Full Ratio: {metrics_data['content_ratio_full']:.2%}\n"
            f"Crop Ratio: {metrics_data['content_ratio_crop']:.2%}"
        )
        axes[6].text(0, 0.5, info_text, transform=axes[6].transAxes, fontsize=12)

        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Failed to save whitespace visualization: {exc}",
            stacklevel=2,
        )
