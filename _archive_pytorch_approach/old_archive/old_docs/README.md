# Ca Mau Forest Change Detection - Multi-Model Comparison

**Author:** Ninh Hải Đăng (21021411)
**Institution:** VNU University of Engineering and Technology
**Year:** 2025-2026

Automatic mangrove forest change detection using Deep Learning with comparison of 3 state-of-the-art models and multi-sensor satellite imagery (Sentinel-2 + Sentinel-1).

---

## 📋 OVERVIEW

This thesis compares **3 deep learning models** for forest change detection:

| Model | Type | Params | Batch | Time | Test mIoU | Status |
|-------|------|--------|-------|------|-----------|--------|
| **SNUNet-CD** | CNN | 4-8M | 16 | 1h 43min | **79.50%** | ✅ Complete |
| **Changer** | Transformer | 8-10M | 12 | ~45 min | - | ⏳ Pending |
| **BAN** | Transformer | 90M | 8 | ~60 min | - | ⏳ Pending |

**Dataset:** 1,285 samples (1,028 train / 128 val / 129 test) from Ca Mau, Vietnam
**Input:** 9 channels per time step (Sentinel-2: B4,B8,B11,B12,NDVI,NBR,NDMI + Sentinel-1: VH,Ratio)

---

## 📁 PROJECT STRUCTURE

```
25-26_HKI_DATN_21021411_DangNH/
│
├── configs/                    # Model configs
│   ├── snunet_camau.py
│   ├── changer_camau.py
│   └── ban_camau.py
│
├── data/processed/             # Preprocessed dataset
│   ├── train/                 # 1,028 samples
│   │   ├── A/                # Time 1 (9-ch TIFFs)
│   │   ├── B/                # Time 2 (9-ch TIFFs)
│   │   └── label/            # Change masks (PNGs)
│   ├── val/                  # 128 samples
│   └── test/                 # 129 samples
│
├── experiments/               # Training outputs
│   ├── snunet/
│   ├── changer/
│   └── ban/
│
├── notebooks/                 # 🎯 TRAINING NOTEBOOKS
│   ├── train_snunet.ipynb    # ⭐ Train SNUNet-CD
│   ├── train_changer.ipynb   # ⭐ Train Changer
│   ├── train_ban.ipynb       # ⭐ Train BAN
│   └── compare_models.ipynb  # ⭐ Compare results
│
├── results/                   # Final results
│
├── src/                       # Source code
│   ├── custom_transforms.py  # 9-channel TIFF loader
│   ├── data_utils.py
│   ├── evaluation_utils.py
│   └── training_utils.py
│
├── open-cd/                   # Open-CD framework
│
├── train_camau.py             # CLI training script
├── MODEL_ANALYSIS.md          # Model selection analysis
├── environment.yml
├── requirements.txt
└── README.md
```

---

## 🚀 QUICK START

### 1. Environment Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate dang

# Verify GPU
nvidia-smi
```

### 2. Training (Choose One Method)

#### ⭐ Method A: Jupyter Notebook (Recommended - Easy!)

```bash
# Start Jupyter
jupyter notebook

# Open one of these notebooks:
# - notebooks/train_snunet.ipynb   (Fastest, ~35 min)
# - notebooks/train_changer.ipynb  (Balanced, ~45 min)
# - notebooks/train_ban.ipynb      (Best accuracy, ~60 min)
```

**Notebook Features:**
- ✅ Start training with 1 click
- ✅ **Auto-refreshing progress bar** (updates every 5s)
- ✅ Real-time metrics (Loss, Accuracy, ETA)
- ✅ Easy stop/resume
- ✅ No terminal needed!

**How to use:**
1. Run cells 1-4 (Setup + Start Training)
2. Run cell 5 (Progress Bar) - **Just once, it auto-refreshes!**
3. Watch the progress bar update automatically:
   ```
   ══════════════════════════════════════════════
     🚀 SNUNet-CD TRAINING
   ══════════════════════════════════════════════

     [████████████░░░░░░░░░░░░] 42.5%

     Progress: 1,827 / 4,300 iterations
     ETA: 0:18:30
     Loss: 0.3245 | Acc: 92.35%

     🕐 Last update: 2025-10-17 15:05:30

   ══════════════════════════════════════════════
   ```
4. Press "Kernel → Interrupt" to stop monitoring

#### Method B: Command Line

```bash
python train_camau.py configs/snunet_camau.py
python train_camau.py configs/changer_camau.py
python train_camau.py configs/ban_camau.py
```

### 3. Compare Results

After training all 3 models:

```bash
# Open comparison notebook
jupyter notebook notebooks/compare_models.ipynb
```

This generates:
- Training curves comparison
- Performance summary table
- Bar charts (Loss, Accuracy, Speed, Memory)
- CSV export

---

## 🎓 WHY THESE 3 MODELS?

After analyzing Open-CD models for **9-channel compatibility**:

### ✅ SNUNet-CD (Lightweight CNN)
- **Architecture:** Dense Siamese UNet + ECAM attention
- **Why:** Conv2d natively supports any number of channels
- **Advantage:** Fastest, no pretrained needed, good for production

### ✅ Changer (Medium Transformer)
- **Architecture:** MixVisionTransformer with Interaction modules
- **Why:** Patch embedding (Conv2d-based) is flexible
- **Advantage:** Balanced accuracy/speed, explicit bi-temporal interaction

### ✅ BAN (Heavy Transformer)
- **Architecture:** CLIP ViT-B/16 + MiT-B0 with adapters
- **Why:** ViT patch embedding supports any channels
- **Advantage:** Best accuracy, strong pretrained features

### ❌ TinyCDv2 (Rejected)
- **Problem:** Hardcoded for 3 channels in MixingMaskAttentionBlock
- **Cannot use** for 9-channel input without major rewrite

---

## 💻 SYSTEM REQUIREMENTS

### Hardware
- **GPU:** NVIDIA RTX A4000 (16GB) or equivalent
- **RAM:** 32GB recommended
- **Storage:** ~10GB for data + experiments

### Software
- **Python:** 3.8+
- **PyTorch:** 1.13+ with CUDA 11.7+
- **OS:** Windows 11 / Linux

---

## 📊 TRAINING CONFIGS

### SNUNet-CD (Fastest)
```python
batch_size = 24        # Largest batch (lightweight model)
max_iters = 4,300      # ~100 epochs
optimizer = AdamW(lr=0.001, weight_decay=0.01)
GPU memory: ~14-15GB
Training time: ~35-40 minutes
```

### Changer (Balanced)
```python
batch_size = 12        # Medium batch
max_iters = 8,600      # ~100 epochs
optimizer = AdamW(lr=0.0001, weight_decay=0.01)
GPU memory: ~10-12GB
Training time: ~45-50 minutes
```

### BAN (Most Accurate)
```python
batch_size = 8         # Smallest batch (heavy model)
max_iters = 12,800     # ~100 epochs
optimizer = AdamW(lr=0.0001, weight_decay=0.0001)
GPU memory: ~12-14GB
Training time: ~60-65 minutes
```

---

## 🔬 KEY FEATURES

### Multi-Spectral 9-Channel Input
- **Sentinel-2 Optical:** B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2)
- **Vegetation Indices:** NDVI, NBR, NDMI
- **Sentinel-1 SAR:** VH polarization, VH/VV Ratio

### Custom Data Pipeline
- **Rasterio-based TIFF loader** for >4 channel images
- Multi-temporal pairing (Time 1 vs Time 2)
- Data augmentation: Random rotation, flip (no PhotoMetric for 9-ch)
- Normalization: Min-max to [0, 1]

### Model Diversity
- **Architecture:** CNN vs Transformer (Medium vs Heavy)
- **Size:** 4M → 10M → 90M parameters
- **Pretrained:** None / ImageNet / CLIP + ImageNet

---

## 🛠️ TROUBLESHOOTING

### GPU Out of Memory
Reduce batch size in config:
```python
train_dataloader = dict(batch_size=16, ...)  # Lower
```

### Training Too Slow
Increase if GPU has memory:
```python
train_dataloader = dict(batch_size=32, num_workers=8, ...)
```

### Stop Training
**In Jupyter:** Press "Kernel → Interrupt" in monitoring cell
**In CLI:** Press `Ctrl+C`
**In Task Manager:** End `python.exe` process

---

## 📖 DOCUMENTATION

- **MODEL_ANALYSIS.md:** Why these 3 models? Detailed analysis
- **configs/*.py:** Model configs with inline comments
- **notebooks/*.ipynb:** Interactive training with auto-refresh progress
- **src/*.py:** Source code with docstrings

---

## 📈 ACTUAL RESULTS

### SNUNet-CD (Completed ✅)

**Training:** 1h 43min (6,400 iterations / 100 epochs)

| Metric | Validation | Test Set |
|--------|------------|----------|
| **mIoU** | 74.99% | **79.50%** |
| **F1-Score** | 85.67% | **88.56%** |
| **Precision** | 85.79% | **88.86%** |
| **Recall** | 86.57% | **88.39%** |
| **Overall Accuracy** | 86.11% | **88.71%** |

**Per-Class Performance (Test Set):**
| Class | IoU | F1-Score | Precision | Recall |
|-------|-----|----------|-----------|--------|
| Unchanged | 81.56% | 89.84% | 87.75% | 92.04% |
| Changed | 77.44% | 87.28% | 89.97% | 84.75% |

**Details:** See `SNUNET_RESULTS.md`

---

### Changer (Pending)
*To be trained*

### BAN (Pending)
*To be trained*

---

## 📈 EXPECTED RESULTS (Reference)

| Model | Loss | Accuracy | Precision | Recall | F1-Score |
|-------|------|----------|-----------|--------|----------|
| SNUNet-CD | 0.15-0.20 | 94-95% | 0.85 | 0.87 | 0.86-0.88 |
| Changer | 0.12-0.18 | 95-96% | 0.87 | 0.89 | 0.88-0.90 |
| BAN | 0.10-0.15 | 96-97% | 0.89 | 0.90 | 0.89-0.91 |

**Trade-offs:**
- **SNUNet-CD:** ✅ Fastest (35-40 min), good for real-time applications, **79.50% mIoU achieved**
- **Changer:** Best balance between accuracy and speed
- **BAN:** Highest accuracy, best for research

---

## 📝 CITATION

```bibtex
@mastersthesis{dang2025camau,
  title={Multi-Model Comparison for Forest Change Detection in Ca Mau Using Deep Learning},
  author={Ninh Hải Đăng},
  school={VNU University of Engineering and Technology},
  year={2025}
}
```

---

## 📧 CONTACT

**Student:** Ninh Hải Đăng (21021411)
**Email:** ninhhaidangg@gmail.com

---

## 🙏 ACKNOWLEDGMENTS

- **Open-CD Framework:** https://github.com/likyoo/open-cd
- **MMSegmentation:** https://github.com/open-mmlab/mmsegmentation
- **Sentinel Hub:** Satellite imagery access

---

## 🔄 VERSION HISTORY

### 2025-10-17 (Current)
- ✅ Finalized 3-model comparison (SNUNet-CD, Changer, BAN)
- ✅ Optimized configs for RTX A4000 16GB
- ✅ Created interactive notebooks with **auto-refresh progress bars**
- ✅ Implemented custom 9-channel TIFF data loader
- ✅ Clean project structure
- ✅ **Completed SNUNet-CD training** (1h 43min, 79.50% test mIoU)
- ✅ **Evaluated SNUNet-CD on test set** (88.56% F1-score)
- ✅ Created comprehensive results report (`SNUNET_RESULTS.md`)

---

**Status:** ✅ SNUNet-CD completed | Changer & BAN pending
**Framework:** Open-CD (MMSegmentation)
**GPU:** NVIDIA RTX A4000 16GB
**Last Updated:** 2025-10-17
