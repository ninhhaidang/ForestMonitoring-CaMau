# 🌳 Forest Change Detection - Ca Mau Mangrove

**Phát hiện mất rừng ngập mặn Cà Mau sử dụng Deep Learning với dữ liệu đa nguồn vệ tinh**

**Sinh viên**: Ninh Hải Đăng (MSSV: 21021411)
**Trường**: Đại học Công nghệ - ĐHQGHN
**Năm học**: 2025-2026

---

## 📋 PROJECT STATUS

**Current Status:** Clean Slate - Ready for New Approach

Previous PyTorch approach has been archived to `_archive_pytorch_approach/`.

---

## 📁 DATA STRUCTURE

### Input Data (Preserved)

```
data/raw/
├── sentinel1/
│   ├── S1_2024_02_04_matched_S2_2024_01_30.tif  (490MB)
│   └── S1_2025_02_22_matched_S2_2025_02_28.tif  (489MB)
├── sentinel2/
│   ├── S2_2024_01_30.tif                        (1.5GB)
│   └── S2_2025_02_28.tif                        (1.5GB)
└── ground_truth/
    └── Training_Points_CSV.csv                  (1,285 points)
```

### Data Specifications

**Sentinel-1 (SAR):**
- 2 time points: 2024-02-04, 2025-02-22
- 2 bands each: VH, VH/VV Ratio
- Not affected by clouds

**Sentinel-2 (Optical):**
- 2 time points: 2024-01-30, 2025-02-28
- 7 bands each: B4, B8, B11, B12, NDVI, NBR, NDMI
- High spectral resolution

**Ground Truth:**
- 1,285 labeled points
- 650 points: No change (label = 0)
- 635 points: Deforestation (label = 1)
- Format: `id,label,x,y` (coordinates in image CRS)

---

## 🎯 PROJECT GOAL

**Objective:** Detect and map mangrove deforestation in Ca Mau region (2024-2025)

**Approach:** TBD (To Be Determined)

**Expected Output:**
1. `probability_map.tif` - Probability of deforestation (0.0-1.0)
2. `binary_map.tif` - Binary classification (0/1)
3. `visualization.png` - Colored map (Green=No change, Red=Deforestation)

---

## 📚 ARCHIVED APPROACHES

### PyTorch Approach (Archived)

**Location:** `_archive_pytorch_approach/`

**What was implemented:**
- 3 models: UNet-MobileNetV2, UNet-EfficientNet-B0, FPN-EfficientNet-B0
- Live training monitoring with tqdm
- Jupyter notebooks for training and inference
- Batch size optimization for 12GB VRAM

**Why archived:**
- Ready for new approach/framework
- All code preserved in archive for reference

---

## 🔧 ENVIRONMENT

### Python Environment

```bash
# Using conda
conda env create -f environment.yml
conda activate dang

# Or using pip
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 1.13+ (CUDA 11.7+)
- GPU: NVIDIA with 12GB+ VRAM recommended
- RAM: 32GB
- Disk: ~10GB free space

---

## 📊 NEXT STEPS

1. **Decide on new approach:**
   - Different framework? (TensorFlow, JAX, etc.)
   - Different architecture? (Transformers, etc.)
   - Different methodology? (Classical ML, etc.)

2. **Set up new pipeline**

3. **Train and evaluate**

4. **Generate final maps**

---

## 📝 NOTES

- All original data is preserved in `data/raw/`
- Previous work is safely archived in `_archive_pytorch_approach/`
- Environment files (environment.yml, requirements.txt) are kept for reference
- .gitignore is configured to exclude large files

---

## 📄 LICENSE

MIT License - See LICENSE file

---

## 📧 CONTACT

**Sinh viên:** Ninh Hải Đăng
**MSSV:** 21021411
**Email:** ninhhaidangg@gmail.com
**Trường:** Đại học Công nghệ - ĐHQGHN

---

**Last Updated:** 2025-10-22
**Status:** ✨ Clean slate, ready for new approach
