# CA MAU FOREST CHANGE DETECTION - PROJECT STATUS
**Author:** Ninh Hai Dang (21021411)
**Date:** 2025-10-17
**Status:** Ready to Train 3 Models

---

## 🎯 OBJECTIVE

Train and compare 3 state-of-the-art change detection models:
1. **BAN** (Bi-temporal Adapter Network) - TGRS 2024
2. **TinyCDv2** - Lightweight model (2024-2025)
3. **Changer** - Feature Interaction Network (TGRS 2023)

---

## ✅ COMPLETED

### 1. Environment Setup
- ✅ Python 3.8.20, PyTorch 1.13.1+cu117, CUDA 11.7
- ✅ Open-CD 1.1.0 installed (freshly cloned from GitHub)
- ✅ Custom rasterio TIFF loader registered

### 2. Data Preparation
- ✅ **1,285 samples** total
  - Train: 1,028 (80%)
  - Val: 128 (10%)
  - Test: 129 (10%)
- ✅ **9 channels per timestep**:
  - S2: B4, B8, B11, B12, NDVI, NBR, NDMI
  - S1: VH, Ratio
- ✅ Format: 256×256 patches, float32, normalized [0,1]

### 3. Model Configs
- ✅ `configs/ban_camau.py` - BAN config (batch=4, 25K iters ≈100 epochs)
- ✅ `configs/tinycdv2_camau.py` - TinyCDv2 config (batch=8, 12.8K iters ≈100 epochs)
- ✅ `configs/changer_camau.py` - Changer config (batch=6, 17.1K iters ≈100 epochs)

### 4. Project Cleanup
- ✅ Removed all ablation studies and baselines
- ✅ Removed SimpleSiameseUNet experiments
- ✅ Clean project structure

---

## 📂 PROJECT STRUCTURE

```
D:\HaiDang\25-26_HKI_DATN_21021411_DangNH\
├── data/
│   └── processed/              # 1,285 samples (9-channel TIFF)
│       ├── train/              # 1,028 samples
│       ├── val/                # 128 samples
│       └── test/               # 129 samples
│
├── configs/                    # Model configurations
│   ├── ban_camau.py           # BAN config
│   ├── tinycdv2_camau.py      # TinyCDv2 config
│   └── changer_camau.py       # Changer config
│
├── src/
│   ├── custom_transforms.py   # Rasterio loader for 9-ch TIFF
│   └── data_utils.py          # Data preprocessing utilities
│
├── experiments/                # Training outputs
│   ├── ban/                   # BAN experiments
│   ├── tinycdv2/              # TinyCDv2 experiments
│   └── changer/               # Changer experiments
│
├── results/                    # Final results
│
├── open-cd/                    # Open-CD framework
│
├── train_camau.py             # Training script
├── README.md                  # Project documentation
└── PROJECT_STATUS.md          # This file
```

---

## 🚀 NEXT STEPS - TRAINING 3 MODELS

### Step 1: Train TinyCDv2 (Fastest - ~4-6 hours)
```bash
python train_camau.py configs/tinycdv2_camau.py --work-dir experiments/tinycdv2
```

**Expected:**
- Training time: ~4-6 hours
- Batch size: 8
- Iterations: 12,800 (~100 epochs)
- Validation interval: Every 1,280 iters (~10 epochs)

### Step 2: Train BAN (~8-12 hours)
```bash
python train_camau.py configs/ban_camau.py --work-dir experiments/ban
```

**Expected:**
- Training time: ~8-12 hours
- Batch size: 4
- Iterations: 25,000 (~100 epochs)
- Validation interval: Every 2,500 iters (~10 epochs)

### Step 3: Train Changer (~6-10 hours)
```bash
python train_camau.py configs/changer_camau.py --work-dir experiments/changer
```

**Expected:**
- Training time: ~6-10 hours
- Batch size: 6
- Iterations: 17,100 (~100 epochs)
- Validation interval: Every 1,710 iters (~10 epochs)

---

## 📊 EXPECTED RESULTS

| Model | Parameters | Speed | Expected F1 | Expected IoU |
|-------|-----------|-------|-------------|--------------|
| **BAN** | ~8M | Slow (~2s/tile) | 0.90-0.92 | 0.82-0.85 |
| **TinyCDv2** | ~1.5M | Fast (~0.5s/tile) | 0.87-0.89 | 0.77-0.80 |
| **Changer** | ~10M | Slow (~2.5s/tile) | 0.89-0.91 | 0.80-0.83 |

---

## ⚠️ IMPORTANT NOTES

1. **Custom Transforms**: All configs use `MultiImgLoadRasterioFromFile` for 9-channel TIFF loading
2. **PhotoMetricDistortion**: Removed from all configs (not compatible with >3 channels)
3. **Training Order**: Recommend training TinyCDv2 first (fastest) to verify pipeline works
4. **GPU Memory**: RTX A4000 16GB is sufficient for all 3 models
5. **Checkpoints**: Best models saved based on validation mIoU

---

## 🎓 FOR THESIS

### Data
- Multi-sensor fusion: Sentinel-1 SAR + Sentinel-2 Optical
- Study area: Ca Mau mangrove forest
- Time period: Jan 2024 → Feb 2025

### Contributions
1. Comparison of 3 SOTA models on 9-channel multi-spectral+SAR data
2. Custom data pipeline for real-world satellite imagery
3. Practical recommendations for operational deployment

### Timeline
- **This week**: Train all 3 models
- **Next week**: Evaluate, compare, and analyze results
- **Following week**: Write thesis and prepare presentation

---

## 📞 CONTACT

**Ninh Hải Đăng (21021411)**
Email: ninhhaidangg@gmail.com
Status: Ready to train 3 Open-CD models
