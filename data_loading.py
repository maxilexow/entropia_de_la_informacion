import numpy as np
import os
import utils
import pandas as pd

def hilbert_dataframe_from_folder(
    root_dir: str,
    crop_size: int = 512,
    extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff")
):
    series = {}

    for root, _, files in os.walk(root_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(extensions):
                continue

            path = os.path.join(root, fname)
            img = load_grayscale_image(path, crop_size=crop_size)

            ts = utils.image_to_hilbert_timeseries(img)
            key = os.path.relpath(path, root_dir)
            series[key] = ts

    if not series:
        raise RuntimeError("No valid images found")

    return pd.DataFrame(series)

from PIL import Image


def load_grayscale_image(path: str, crop_size: int = 512) -> np.ndarray:
    """
    Load image, convert to grayscale, center-crop.
    """
    img = Image.open(path).convert("L")
    img = np.asarray(img, dtype=np.float64)

    img = center_crop(img, size=crop_size)

    # Sanity: 512 is already 2^9
    assert img.shape == (crop_size, crop_size)

    return img

import numpy as np


def center_crop(img: np.ndarray, size: int = 512) -> np.ndarray:
    """
    Center-crop a square region of given size.

    Parameters
    ----------
    img : np.ndarray
        2D image
    size : int
        Crop size (pixels)

    Returns
    -------
    cropped : np.ndarray
    """
    h, w = img.shape
    if h < size or w < size:
        raise ValueError(
            f"Image too small for {size}x{size} crop: {img.shape}"
        )

    y0 = (h - size) // 2
    x0 = (w - size) // 2

    return img[y0:y0 + size, x0:x0 + size]
