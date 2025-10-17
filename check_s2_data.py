import rasterio
import numpy as np

src = rasterio.open('data/raw/sentinel2/S2_2024_01_30.tif')

print("=== SENTINEL-2 RAW DATA CHECK ===")
print(f"Bands: {src.count}")
print(f"Shape: {src.shape}")
print(f"CRS: {src.crs}")
print(f"NoData value: {src.nodata}")
print()

for i in range(1, src.count+1):
    data = src.read(i)
    valid = ~np.isnan(data)
    if src.nodata is not None:
        valid = valid & (data != src.nodata)

    print(f"Band {i}:")
    print(f"  Valid pixels: {valid.sum()}/{data.size} ({valid.sum()/data.size*100:.1f}%)")
    if valid.sum() > 0:
        valid_data = data[valid]
        print(f"  Range: [{valid_data.min():.2f}, {valid_data.max():.2f}]")
        print(f"  Mean: {valid_data.mean():.2f}")
    print()
