"""
Utility functions for file handling, image conversion, and data processing.
"""

import os
import uuid
import numpy as np
import rasterio
from PIL import Image
from typing import Optional, Tuple


UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
PREVIEW_DIR = os.path.join(os.path.dirname(__file__), "previews")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {".tif", ".tiff"}

# Max file size: 100MB
MAX_FILE_SIZE = 100 * 1024 * 1024


def validate_file(filename: str, file_size: int) -> Tuple[bool, str]:
    """Validate uploaded file."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type '{ext}'. Only .tif files are allowed."
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large ({file_size / 1024 / 1024:.1f}MB). Maximum is 100MB."
    return True, "OK"


def save_upload(file_content: bytes, original_filename: str) -> str:
    """Save uploaded file and return the saved path."""
    ext = os.path.splitext(original_filename)[1].lower()
    file_id = str(uuid.uuid4())[:8]
    safe_name = f"{file_id}_{os.path.basename(original_filename)}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(save_path, "wb") as f:
        f.write(file_content)

    return save_path


def tif_to_png_preview(
    tif_path: str, 
    mode: str = "rgb",
    output_size: int = 512
) -> str:
    """
    Convert a .tif file to a PNG preview image.
    
    Args:
        tif_path: Path to the .tif file
        mode: 'rgb' for true-color or 'ndvi' for NDVI visualization
        output_size: Output image size (square)
        
    Returns:
        Path to the generated PNG file
    """
    with rasterio.open(tif_path) as src:
        data = src.read()  # (bands, H, W)

    num_bands = data.shape[0]

    if mode == "ndvi" and num_bands >= 8:
        # NDVI = (NIR - Red) / (NIR + Red)
        # Sentinel-2: Band 4 (Red, index 3), Band 8 (NIR, index 7)
        red = data[3].astype(np.float32)
        nir = data[7].astype(np.float32)

        denominator = nir + red
        ndvi = np.where(denominator > 0, (nir - red) / denominator, 0.0)

        # Colormap for NDVI (-1 to 1) -> (0 to 255) with green gradient
        ndvi_normalized = ((ndvi + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        # Create a green-themed colormap
        img_array = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
        
        # Low NDVI: brown/bare soil
        mask_low = ndvi < 0.2
        img_array[mask_low] = [139, 119, 101]  # Brown
        
        # Medium NDVI: light green
        mask_mid = (ndvi >= 0.2) & (ndvi < 0.5)
        img_array[mask_mid] = [144, 238, 144]  # Light green
        
        # High NDVI: dark green  
        mask_high = (ndvi >= 0.5) & (ndvi < 0.7)
        img_array[mask_high] = [34, 139, 34]  # Forest green
        
        # Very high NDVI: deep green
        mask_vhigh = ndvi >= 0.7
        img_array[mask_vhigh] = [0, 100, 0]  # Dark green

    else:
        # RGB preview
        if num_bands >= 4:
            # Sentinel-2: Band 4 (Red), Band 3 (Green), Band 2 (Blue) - 1-indexed
            r, g, b = data[3], data[2], data[1]
        elif num_bands >= 3:
            r, g, b = data[0], data[1], data[2]
        else:
            # Grayscale
            r = g = b = data[0]

        # Normalize to 0-255
        def normalize_band(band):
            band = band.astype(np.float32)
            p2, p98 = np.percentile(band, (2, 98))
            if p98 - p2 > 0:
                band = (band - p2) / (p98 - p2) * 255
            else:
                band = band * 0
            return np.clip(band, 0, 255).astype(np.uint8)

        img_array = np.stack([
            normalize_band(r),
            normalize_band(g),
            normalize_band(b)
        ], axis=-1)

    # Create PIL image and resize
    img = Image.fromarray(img_array)
    img = img.resize((output_size, output_size), Image.Resampling.LANCZOS)

    # Save preview
    file_id = str(uuid.uuid4())[:8]
    preview_name = f"{file_id}_preview_{mode}.png"
    preview_path = os.path.join(PREVIEW_DIR, preview_name)
    img.save(preview_path, "PNG")

    return preview_path


def cleanup_old_files(directory: str, max_age_seconds: int = 3600):
    """Remove files older than max_age_seconds from a directory."""
    import time
    now = time.time()
    for f in os.listdir(directory):
        fpath = os.path.join(directory, f)
        if os.path.isfile(fpath):
            age = now - os.path.getmtime(fpath)
            if age > max_age_seconds:
                os.remove(fpath)
