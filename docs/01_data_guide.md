# Data Guide

## Cấu trúc Data

### 1. Raw Data (`data/raw/`)
**KHÔNG BAO GIỜ sửa đổi thư mục này!**

#### Sentinel-2 (Optical)
- T1: `S2_2024_01_30.tif` (Before)
- T2: `S2_2025_02_28.tif` (After)
- Bands: B4, B8, B11, B12
- Derived indices: NDVI, NBR, NDMI

#### Sentinel-1 (SAR)
- T1: `S1_2024_02_04_matched.tif`
- T2: `S1_2025_02_22_matched.tif`
- Bands: VH polarization, Ratio (VV-VH)

#### Ground Truth
- 1285 điểm (shapefile + CSV)
- Label: 0 (no change), 1 (forest loss)

### 2. Processed Data (`data/processed/`)

#### Phase 1: S2 only (14 channels)
- Before: 7 channels (4 bands + 3 indices)
- After: 7 channels (4 bands + 3 indices)

#### Phase 2: S2 + S1 (18 channels)
- Before: 9 channels (S2: 7 + S1: 2)
- After: 9 channels (S2: 7 + S1: 2)

### 3. Samples (`data/samples/`)
Training patches (256x256 pixels):
- Train: ~1028 samples (80%)
- Val: ~128 samples (10%)
- Test: ~129 samples (10%)
