# CA MAU FOREST CHANGE DETECTION
**Author:** Ninh Hai Dang (21021411)
**Institution:** University of Engineering and Technology, VNU
**Year:** 2024-2025

Automatic mangrove forest change detection using Deep Learning with comparison of 3 state-of-the-art models and multi-sensor satellite imagery (Sentinel-2 + Sentinel-1).

---

## 🎯 PROJECT OBJECTIVE

Compare performance of **3 state-of-the-art change detection models**:
1. **BAN** (Bi-temporal Adapter Network) - IEEE TGRS 2024
2. **TinyCDv2** - Ultra-lightweight efficient model (2024-2025)
3. **Changer** (Feature Interaction Network) - IEEE TGRS 2023

**Approach:** Multi-sensor fusion (Sentinel-1 SAR + Sentinel-2 Optical) for robust change detection.

---

## 📊 DATASET

### Ground Truth
- **1,285 samples** total
  - Train: 1,028 (80%)
  - Val: 128 (10%)
  - Test: 129 (10%)
- **Study area:** Ca Mau mangrove forest, Vietnam
- **Time period:** January 2024 → February 2025

### Satellite Data

**Sentinel-2 (Optical):**
- 4 bands: B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2)
- 3 indices: NDVI, NBR, NDMI
- Resolution: 10-20m

**Sentinel-1 (SAR):**
- 2 features: VH polarization, Ratio (VV-VH)
- Resolution: 10m

**Total Input:** 18 channels (9 per timestep × 2 timesteps)

---

## 🧠 MODEL ARCHITECTURES

| Model | Type | Parameters | Input Size | Speed | Expected F1 |
|-------|------|-----------|-----------|-------|-------------|
| **BAN** | Transformer | ~8M | 512×512 | ~2s/tile | 0.90-0.92 |
| **TinyCDv2** | CNN (Lightweight) | ~1.5M | 256×256 | ~0.5s/tile | 0.87-0.89 |
| **Changer** | CNN+FI | ~10M | 256×256 | ~2.5s/tile | 0.89-0.91 |

### Why These 3 Models?

After analyzing 18 models in Open-CD framework, selected based on:
- ✅ State-of-the-art (2023-2025)
- ✅ Diverse approaches (Transformer vs CNN vs Hybrid)
- ✅ Suitable for limited data (1,200 samples)
- ✅ Multi-sensor fusion capability

---

## 💻 ENVIRONMENT

### Hardware
```
CPU: Intel Xeon E5-2678 v3 (12 cores @ 2.5GHz)
RAM: 32GB DDR3 ECC
GPU: NVIDIA RTX A4000 16GB VRAM
Storage: 4TB HDD
OS: Windows 11 Pro
```

### Software
```
Python: 3.8.20
PyTorch: 1.13.1+cu117
CUDA: 11.7
Open-CD: 1.1.0
MMCV: 2.1.0
MMSegmentation: 1.2.2
Rasterio: 1.3.11
```

### Installation
```bash
# Create conda environment
conda env create -f environment.yml
conda activate dang

# Install Open-CD
git clone https://github.com/likyoo/open-cd.git
cd open-cd && pip install -v -e . && cd ..

# Verify installation
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 📁 PROJECT STRUCTURE

```
Ca_Mau_Forest_Change_Detection/
├── data/
│   ├── raw/                    # Original satellite imagery (6GB)
│   │   ├── sentinel2/          # S2: 2 files
│   │   ├── sentinel1/          # S1: 2 files
│   │   └── ground_truth/       # 1,285 points
│   └── processed/              # Training patches (18 channels)
│       ├── train/              # 1,028 samples
│       ├── val/                # 128 samples
│       └── test/               # 129 samples
│
├── configs/                    # Model configurations
│   ├── ban_camau.py           # BAN config
│   ├── tinycdv2_camau.py      # TinyCDv2 config
│   └── changer_camau.py       # Changer config
│
├── src/                        # Source code
│   ├── data_utils.py          # Data preprocessing
│   ├── custom_transforms.py   # Custom TIFF loader
│   └── simple_model.py        # Model architectures
│
├── experiments/                # Training outputs
│   ├── ban/                   # BAN experiments
│   ├── tinycdv2/              # TinyCDv2 experiments
│   └── changer/               # Changer experiments
│
├── results/                    # Evaluation results
│
├── open-cd/                    # Open-CD framework
│
├── train_camau.py             # Training script
├── PROJECT_STATUS.md          # Current project status
├── ENVIRONMENT_CHECK.md       # Environment validation
└── README.md                  # This file
```

---

## 🚀 USAGE

### 1. Data Preprocessing
Data has been preprocessed into 256×256 patches with 9 channels per timestep.

### 2. Training

**Train TinyCDv2 (Recommended first - fastest):**
```bash
python train_camau.py configs/tinycdv2_camau.py --work-dir experiments/tinycdv2
```

**Train BAN:**
```bash
python train_camau.py configs/ban_camau.py --work-dir experiments/ban
```

**Train Changer:**
```bash
python train_camau.py configs/changer_camau.py --work-dir experiments/changer
```

### 3. Monitoring
```bash
# View training logs
tail -f experiments/[model]/[timestamp]/[timestamp].log

# Check GPU usage (Windows)
# Task Manager → Performance → GPU
```

### 4. Evaluation
```bash
# Evaluate on test set
python open-cd/tools/test.py \
    configs/[model]_camau.py \
    experiments/[model]/[timestamp]/best_checkpoint.pth
```

---

## 🔬 KEY INNOVATIONS

1. **Multi-sensor Fusion:** Combining Sentinel-1 SAR + Sentinel-2 Optical for robust detection
2. **9-channel Input:** Custom data pipeline handling B4, B8, B11, B12, NDVI, NBR, NDMI, VH, Ratio
3. **Custom TIFF Loader:** Rasterio-based loader for >4 channel images (OpenCV limitation)
4. **Model Comparison:** Systematic evaluation of 3 SOTA architectures
5. **Real-world Application:** Operational mangrove forest monitoring in Ca Mau

---

## 📊 EXPECTED RESULTS

### Quantitative Metrics
- **Overall Accuracy:** 87-92%
- **F1 Score:** 0.87-0.92
- **IoU:** 0.77-0.85
- **Precision:** 0.85-0.90
- **Recall:** 0.85-0.90

### Model Comparison
Will compare 3 models on:
- Accuracy metrics (F1, IoU, Precision, Recall)
- Inference speed
- Model size
- Robustness to cloud/shadow
- Multi-sensor fusion effectiveness

---

## 🎓 THESIS CONTRIBUTIONS

1. **Comprehensive Comparison:** First systematic comparison of BAN, TinyCDv2, and Changer on mangrove forest
2. **Multi-sensor Dataset:** Novel 9-channel dataset combining S1+S2 for Vietnam mangrove
3. **Practical Application:** Operational deployment recommendations for mangrove monitoring
4. **Reproducible Research:** Clean code, detailed documentation, open-source

---

## ⚠️ TECHNICAL NOTES

### Custom Data Pipeline
- **Challenge:** OpenCV TIFF decoder only supports ≤4 channels
- **Solution:** Custom `MultiImgLoadRasterioFromFile` transform using Rasterio
- **Location:** `src/custom_transforms.py`

### PhotoMetric Augmentation
- **Removed:** Not compatible with >3 channel images
- **Kept:** Geometric augmentations (rotation, flip)

### Pretrained Weights
- **BAN:** CLIP ViT-B/16 + MiT-B0
- **TinyCDv2:** EfficientNet-B4 (ImageNet)
- **Changer:** MiT-B0 (SegFormer)

---

## 📈 TIMELINE

- **Week 1:** Environment setup + Data preprocessing ✅
- **Week 2:** Model training (BAN, TinyCDv2, Changer) ⏳
- **Week 3:** Evaluation + Comparison + Analysis
- **Week 4:** Thesis writing + Presentation preparation

**Current Status:** Ready to train 3 models

---

## 📚 REFERENCES

### Papers
1. **BAN:** "Bi-temporal Adapter Network for Remote Sensing Change Detection", IEEE TGRS 2024
2. **TinyCDv2:** "Tiny Change Detection v2" (Under Review, 2024-2025)
3. **Changer:** "Changer: Feature Interaction is What You Need for Change Detection", IEEE TGRS 2023
4. **Open-CD:** Li et al., "Open-CD: A Comprehensive Toolbox for Change Detection", 2024

### Resources
- **Open-CD:** https://github.com/likyoo/open-cd
- **Sentinel Data:** https://scihub.copernicus.eu/

---

## 📞 CONTACT

**Ninh Hải Đăng**
Student ID: 21021411
Email: ninhhaidangg@gmail.com
GitHub: ninhhaidang

**Project Status:** Environment validated, ready for training
**Last Updated:** 2025-10-17

---

## 📄 LICENSE

This project is for academic purposes as part of a Bachelor's thesis at University of Engineering and Technology, VNU.

---

## 🙏 ACKNOWLEDGMENTS

- **Open-CD Team** for the comprehensive change detection framework
- **PyTorch Team** for the excellent deep learning framework
- **Rasterio Contributors** for geospatial data handling
- **NVIDIA** for CUDA support enabling GPU training
