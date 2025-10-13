"""
Data Utilities
Các hàm tiện ích để load và visualize data
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_geotiff(filepath):
    """Load GeoTIFF file"""
    with rasterio.open(filepath) as src:
        data = src.read()
        profile = src.profile
    return data, profile

def visualize_rgb(data, bands=[3, 2, 1], title="RGB Composite"):
    """Visualize RGB composite"""
    rgb = np.stack([data[b] for b in bands], axis=-1)
    # Normalize to 0-1
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def calculate_ndvi(nir, red):
    """Calculate NDVI index"""
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi
