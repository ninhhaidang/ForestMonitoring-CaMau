# 📓 Jupyter Notebooks

Thư mục này chứa các Jupyter notebooks để khám phá dữ liệu, tạo dataset, huấn luyện models và trực quan hóa kết quả.

---

## 📋 Danh Sách Notebooks

### ✅ `00_module_usage_example.ipynb`
**Trạng thái:** Hoàn thành
**Mục đích:** Hướng dẫn sử dụng các modules từ `src/`

**Nội dung:**
- Import và setup modules
- Ví dụ sử dụng `utils.py`: load_tiff, check_tiff_metadata, get_tiff_stats
- Ví dụ sử dụng `preprocessing.py`: normalize_band, handle_nan, extract_patch
- Ví dụ sử dụng `visualization.py`: plot_band, plot_statistics
- Ví dụ sử dụng `models.py`: get_model, count_parameters
- Test forward pass với 3 models

**Thời gian:** 2-3 phút
**Outputs:** Không có (chỉ tutorial)

---

### ✅ `01_data_exploration.ipynb`
**Trạng thái:** Hoàn thành
**Mục đích:** Kiểm tra metadata và khám phá dữ liệu Sentinel-1/2

**Nội dung:**
- Import libraries và setup paths
- Check metadata của 4 TIFF files (CRS, resolution, bounds, dtype)
- Phân tích statistics (min, max, mean, std, NaN%)
- Visualizations:
  - NaN percentage comparison
  - Band mean value comparison
  - Vegetation indices 2024 vs 2025 (NDVI, NBR, NDMI)
  - Sample band images
- Summary report

**Thời gian:** 3-5 phút
**Outputs:**
- `data/metadata_summary.csv`
- `figures/band_nan_comparison.png`
- `figures/band_mean_comparison.png`
- `figures/indices_2024_vs_2025.png`
- `figures/sample_band_images.png`

**Key Finding:** Vegetation indices giảm 46-66% từ 2024 sang 2025

---

### ✅ `02_create_patches_dataset.ipynb`
**Trạng thái:** Hoàn thành
**Mục đích:** Tạo patches dataset 128×128×18 từ 4 TIFF files

**Nội dung:**
- Import và check data availability
- Load ground truth CSV (1,285 points)
- Call `create_patches_dataset()` từ `src.preprocessing`
  - Load 4 TIFF files (~4GB)
  - Stack thành 18 channels
  - Extract patches tại ground truth locations
  - Handle NaN values
  - Normalize bands (mixed strategy)
  - Split train/val/test (70/15/15)
- Verify created patches
- Visualize sample patches (key bands)
- Check for NaN values in final patches
- Summary

**Thời gian:** 10-15 phút
**Outputs:**
- `data/patches/train/*.npy` (~900 files)
- `data/patches/val/*.npy` (~190 files)
- `data/patches/test/*.npy` (~195 files)
- `data/patches/dataset_summary.txt`

**⚠️ Note:** Sử dụng `tqdm` để theo dõi tiến độ

---

### ✅ `03_train_models.ipynb`
**Trạng thái:** Hoàn thành
**Mục đích:** Huấn luyện 3 shallow CNN models

**Nội dung:**
- Setup và imports
- Configuration (batch size, learning rate, epochs, etc.)
- Check patches availability
- Create dataloaders với augmentation
- Define training function với:
  - `tqdm` progress bars
  - BCELoss
  - Adam optimizer
  - ReduceLROnPlateau scheduler
  - Early stopping
  - Model checkpointing
- Train Model 1: Spatial Context CNN (~30K params)
- Train Model 2: Multi-Scale CNN (~80K params)
- Train Model 3: Shallow U-Net (~120K params)
- Save training history
- Plot training curves (loss, accuracy)
- Compare best results

**Thời gian:** 30-60 phút per model (RTX A4000)
**Outputs:**
- `checkpoints/spatial_cnn_best.pth`
- `checkpoints/multiscale_cnn_best.pth`
- `checkpoints/shallow_unet_best.pth`
- `logs/training_history_all_models.csv`
- `logs/{model_name}_history.csv` (individual)
- `logs/models_comparison.csv`
- `figures/training_curves/training_curves_all_models.png`

**⚠️ Features:**
- Real-time training progress với `tqdm`
- Automatic early stopping
- Learning rate scheduling
- Best model checkpointing

---

### ✅ `04_evaluate_and_visualize_results.ipynb`
**Trạng thái:** Hoàn thành
**Mục đích:** Đánh giá models trên test set và visualize kết quả

**Nội dung:**
- Setup và imports
- Load test dataset
- Load trained model checkpoints
- Evaluate models on test set với `tqdm`
- Calculate metrics:
  - Accuracy, Precision, Recall, F1-Score
  - AUC-ROC
  - Confusion matrices
- Generate visualizations:
  - Confusion matrices (3 models side-by-side)
  - ROC curves comparison
  - Sample predictions (RGB + probability maps + NDVI change)
  - Model agreement analysis
- Detailed classification reports
- Summary and recommendations

**Thời gian:** 5-10 phút
**Outputs:**
- `outputs/test_metrics.csv`
- `figures/confusion_matrices/confusion_matrices_all_models.png`
- `figures/roc_curves_all_models.png`
- `figures/sample_predictions/sample_predictions_comparison.png`
- `figures/model_agreement_analysis.png`

**⚠️ Features:**
- Comprehensive metrics comparison
- Visual comparison of predictions
- Model agreement analysis
- Best model recommendation

---

## 🚀 Workflow

### Bước 0: Tutorial (Optional)
```bash
conda activate dang
jupyter lab
# Mở: 00_module_usage_example.ipynb
```

### Bước 1: Khám Phá Dữ Liệu (Optional)
```bash
# Mở: 01_data_exploration.ipynb
# Chạy all cells
```

### Bước 2: Tạo Patches (Required)
```bash
# Mở: 02_create_patches_dataset.ipynb
# Chạy all cells
# ⏱️ Đợi 10-15 phút
```

### Bước 3: Huấn Luyện Models (Required)
```bash
# Mở: 03_train_models.ipynb
# Chạy all cells
# ⏱️ Đợi 1-3 giờ (tùy GPU)
```

### Bước 4: Đánh Giá Kết Quả (Required)
```bash
# Mở: 04_evaluate_and_visualize_results.ipynb
# Chạy all cells
# ⏱️ ~5-10 phút
```

---

## 📦 Dependencies

Tất cả dependencies đã được cài đặt trong môi trường `dang`:

**Core:**
- `torch==1.13.1+cu117` - Deep learning
- `rasterio==1.3.11` - GeoTIFF I/O
- `numpy==1.24.4` - Numerical computing

**Visualization:**
- `matplotlib` - Plotting
- `seaborn` - Statistical plots
- `tqdm` - Progress bars

**Utilities:**
- `pandas` - Data manipulation
- `scikit-learn` - Metrics & splitting
- `opencv-python` - Image processing

**Environment:**
- `jupyterlab==4.2.5` - Notebook interface

---

## 💡 Tips

### Chạy Notebook nhanh
```bash
# Chạy notebook từ command line (không mở browser)
jupyter nbconvert --execute --to notebook your_notebook.ipynb
```

### Progress Bars (tqdm)
Tất cả notebooks đã tích hợp `tqdm.auto`:
```python
from tqdm.auto import tqdm

for item in tqdm(iterable, desc="Processing", unit="item"):
    # Your code here
    pass
```

### GPU Memory Management
```python
import torch

# Clear GPU cache
torch.cuda.empty_cache()

# Check GPU memory
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### Reload Modules
Khi đang phát triển modules trong `src/`:
```python
import importlib
from src import utils

# Sau khi sửa utils.py
importlib.reload(utils)
```

### Save/Load Model Checkpoints
```python
# Save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## ⚠️ Lưu Ý

### 1. Memory
- Notebooks có thể dùng nhiều RAM khi load TIFF files (~4GB)
- Khuyến nghị: ≥16GB RAM
- GPU: NVIDIA với ≥8GB VRAM cho training

### 2. Paths
Sử dụng relative paths từ thư mục notebooks:
```python
from pathlib import Path

# Đúng
data_dir = Path('../data/patches')

# Sai
data_dir = 'D:/HaiDang/.../data/patches'  # Hard-coded path
```

### 3. Kernel
Đảm bảo chọn đúng kernel `dang` trong JupyterLab:
- Kernel → Change Kernel → **dang**

### 4. Training Time
- Spatial CNN: ~30-45 phút
- Multi-Scale CNN: ~45-60 phút
- Shallow U-Net: ~60-90 phút
- Total: ~2-3 giờ

### 5. Disk Space
- Patches: ~500MB
- Checkpoints: ~10MB per model
- Figures: ~50MB
- Total: ~600-700MB

---

## 📊 Expected Directory Structure

Sau khi chạy TẤT CẢ notebooks:

```
ca-mau-deforestation/
├── data/
│   ├── metadata_summary.csv                    ← 01
│   └── patches/
│       ├── train/ (900 .npy files)             ← 02
│       ├── val/ (190 .npy files)               ← 02
│       ├── test/ (195 .npy files)              ← 02
│       └── dataset_summary.txt                 ← 02
│
├── checkpoints/
│   ├── spatial_cnn_best.pth                    ← 03
│   ├── multiscale_cnn_best.pth                 ← 03
│   └── shallow_unet_best.pth                   ← 03
│
├── logs/
│   ├── training_history_all_models.csv         ← 03
│   ├── spatial_cnn_history.csv                 ← 03
│   ├── multiscale_cnn_history.csv              ← 03
│   ├── shallow_unet_history.csv                ← 03
│   └── models_comparison.csv                   ← 03
│
├── outputs/
│   └── test_metrics.csv                        ← 04
│
└── figures/
    ├── band_nan_comparison.png                 ← 01
    ├── band_mean_comparison.png                ← 01
    ├── indices_2024_vs_2025.png                ← 01
    ├── sample_band_images.png                  ← 01
    ├── training_curves/
    │   └── training_curves_all_models.png      ← 03
    ├── confusion_matrices/
    │   └── confusion_matrices_all_models.png   ← 04
    ├── roc_curves_all_models.png               ← 04
    ├── sample_predictions/
    │   └── sample_predictions_comparison.png   ← 04
    └── model_agreement_analysis.png            ← 04
```

---

## 🎯 Summary

| Notebook | Trạng thái | Thời gian | Output chính |
|----------|------------|-----------|--------------|
| 00 | ✅ | 2-3 phút | Tutorial |
| 01 | ✅ | 3-5 phút | Metadata & figures |
| 02 | ✅ | 10-15 phút | ~1,285 patches |
| 03 | ✅ | 1-3 giờ | 3 trained models |
| 04 | ✅ | 5-10 phút | Evaluation & viz |

**Total time:** ~2-4 giờ (phụ thuộc GPU)

---

## 🔥 Quick Commands

```bash
# Kích hoạt environment
conda activate dang

# Start JupyterLab
jupyter lab

# Run notebook from command line
jupyter nbconvert --execute --to notebook --inplace notebooks/01_data_exploration.ipynb

# Clear all outputs
jupyter nbconvert --clear-output notebooks/*.ipynb

# Export to HTML
jupyter nbconvert --to html notebooks/04_evaluate_and_visualize_results.ipynb
```

---

**Last updated:** 2025-10-22
**Author:** Ninh Hải Đăng
