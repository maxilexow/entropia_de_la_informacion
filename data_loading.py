import os
import numpy as np
import pandas as pd
from PIL import Image
import utils

def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """
    Center-crop a square region of given size.
    """
    h, w = img.shape
    if h < size or w < size:
        raise ValueError(
            f"Image too small for {size}x{size} crop: {img.shape}"
        )

    y0 = (h - size) // 2
    x0 = (w - size) // 2

    return img[y0:y0 + size, x0:x0 + size]

def load_grayscale_image_with_crop(path: str, crop_size: int) -> np.ndarray:
    """
    Load image, convert to grayscale, center-crop.
    """
    img = Image.open(path).convert("L")
    img = np.asarray(img, dtype=np.float64)

    img = center_crop(img, crop_size)

    # Enforce square + power of two
    assert img.shape == (crop_size, crop_size)
    assert crop_size & (crop_size - 1) == 0

    return img

def hilbert_dataframe_from_folder(
    root_dir: str,
    crop_size: int = 512,
    extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff"),
):
    """
    Load images from a folder, center-crop, convert to Hilbert
    time series, and return a single DataFrame.

    Columns = image filenames
    Rows    = Hilbert index
    """
    series = {}

    for root, _, files in os.walk(root_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(extensions):
                continue

            path = os.path.join(root, fname)

            img = load_grayscale_image_with_crop(path, crop_size)
            ts = utils.image_to_hilbert_timeseries(img)

            key = os.path.relpath(path, root_dir)
            series[key] = ts

    if not series:
        raise RuntimeError("No valid images found")

    return pd.DataFrame(series)
