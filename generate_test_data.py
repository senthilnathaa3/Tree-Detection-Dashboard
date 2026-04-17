"""
Generate a synthetic test .tif file for testing the TreeSat dashboard.
Creates a 15-band GeoTIFF simulating Sentinel-2 + Sentinel-1 stacked data.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os

def generate_test_tif(output_path="test_data/sample_s2_s1_200m.tif", size=64):
    """Generate a synthetic 15-band .tif file."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 15 channels: 13 Sentinel-2 bands + 2 Sentinel-1 bands (VV, VH)
    num_bands = 15
    
    # Create synthetic data
    np.random.seed(42)
    data = np.zeros((num_bands, size, size), dtype=np.float32)
    
    # Simulate Sentinel-2 bands with realistic-ish spectral signatures
    # Band 1: Coastal aerosol
    data[0] = np.random.uniform(0.02, 0.06, (size, size))
    # Band 2: Blue
    data[1] = np.random.uniform(0.03, 0.08, (size, size))
    # Band 3: Green - higher for vegetation
    data[2] = np.random.uniform(0.05, 0.12, (size, size))
    # Band 4: Red - lower for vegetation
    data[3] = np.random.uniform(0.02, 0.08, (size, size))
    # Band 5: Red Edge 1
    data[4] = np.random.uniform(0.05, 0.15, (size, size))
    # Band 6: Red Edge 2
    data[5] = np.random.uniform(0.1, 0.3, (size, size))
    # Band 7: Red Edge 3
    data[6] = np.random.uniform(0.15, 0.35, (size, size))
    # Band 8: NIR - high for vegetation
    data[7] = np.random.uniform(0.2, 0.5, (size, size))
    # Band 8a: Narrow NIR
    data[8] = np.random.uniform(0.2, 0.45, (size, size))
    # Band 9: Water vapor
    data[9] = np.random.uniform(0.15, 0.35, (size, size))
    # Band 10: SWIR - Cirrus
    data[10] = np.random.uniform(0.001, 0.01, (size, size))
    # Band 11: SWIR 1
    data[11] = np.random.uniform(0.08, 0.25, (size, size))
    # Band 12: SWIR 2
    data[12] = np.random.uniform(0.05, 0.2, (size, size))
    
    # Sentinel-1 bands
    # Band 13: VV backscatter (dB range: -25 to 0, normalized)
    data[13] = np.random.uniform(0.1, 0.5, (size, size))
    # Band 14: VH backscatter  
    data[14] = np.random.uniform(0.05, 0.3, (size, size))
    
    # Add a spatial pattern (circular forest patch in center)
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
    dist = np.sqrt(x**2 + y**2) / (size // 2)
    forest_mask = dist < 0.7
    
    # Enhance vegetation signal in forest area
    data[7][forest_mask] *= 1.5  # Higher NIR
    data[3][forest_mask] *= 0.6  # Lower Red
    data[2][forest_mask] *= 1.3  # Higher Green
    
    # Create GeoTIFF with CRS
    transform = from_bounds(10.0, 48.0, 10.002, 48.002, size, size)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=size,
        width=size,
        count=num_bands,
        dtype='float32',
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        for i in range(num_bands):
            dst.write(data[i], i + 1)
    
    print(f"✅ Generated test .tif: {output_path}")
    print(f"   Shape: ({num_bands}, {size}, {size})")
    print(f"   Bands: 13 S2 + 2 S1 = 15 total")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    return output_path


if __name__ == "__main__":
    generate_test_tif()
