# 🧠 Deep Learning Pipeline - Quick Start Guide

Hướng dẫn nhanh để chạy pipeline CNN phát hiện mất rừng với spatial context.

---

## 📊 Tổng quan

**Pipeline này làm gì?**
- Sử dụng **2D CNN** với patches 3x3 (thay vì single pixel như Random Forest)
- Khai thác **spatial context** → giảm nhiễu "lấm tấm"
- **Spatial-aware splitting** → tránh data leakage
- Kết quả: Classification map **mượt mà hơn** Random Forest

**Khác biệt chính với Random Forest:**

| Aspect | Random Forest | CNN (Deep Learning) |
|--------|--------------|---------------------|
| Input | Single pixel (27 features) | Patch 3x3 (27 features × 9 pixels) |
| Spatial context | ❌ No | ✅ Yes (3x3 neighborhood) |
| Training time | ~5-10 min | ~15-20 min |
| Result smoothness | ⚠️ Có noise lấm tấm | ✅ Mượt hơn |
| Accuracy | Good baseline | Similar or better |
| GPU | Not needed | Recommended |

---

## 🚀 Cách chạy

### Option 1: Chạy với settings mặc định

```bash
cd src
python main_dl.py
```

**Settings mặc định:**
- Patch size: 3x3
- Epochs: 50 (có early stopping)
- Batch size: 32
- Learning rate: 0.001
- Device: CUDA (tự động fallback CPU nếu không có GPU)

### Option 2: Custom settings

```bash
# Chạy với 100 epochs
python main_dl.py --epochs 100

# Chạy với batch size lớn hơn (nếu có GPU memory)
python main_dl.py --batch-size 64

# Force CPU (nếu GPU gặp vấn đề)
python main_dl.py --device cpu

# Kết hợp
python main_dl.py --epochs 100 --batch-size 64 --device cuda
```

---

## 📁 Output Files

Sau khi chạy xong, check folder `results/`:

```
results/
├── rasters/
│   ├── cnn_classification.tif      # Binary map (0=No loss, 1=Deforestation)
│   └── cnn_probability.tif         # Probability map (0.0-1.0)
│
├── models/
│   └── cnn_model.pth               # Trained CNN model
│
└── data/
    ├── cnn_training_patches.npz    # Training patches (có thể load lại)
    ├── cnn_evaluation_metrics.json # Accuracy, Precision, Recall, F1, AUC
    └── cnn_training_history.json   # Training curves (loss/accuracy per epoch)
```

---

## 🔍 Hiểu Pipeline

### Pipeline 8 bước:

```
1. Load Data
   └─ Sentinel-1, Sentinel-2, Ground Truth, Boundary

2. Feature Extraction
   └─ 27 features (same as Random Forest)

3. Spatial-Aware Splitting ⭐ (NEW!)
   └─ Cluster nearby points → Split by cluster
   └─ Ensure NO overlap between train/val/test

4. Extract Patches ⭐ (NEW!)
   └─ Extract 3x3 patches at ground truth locations
   └─ Normalize patches (z-score)

5. Train CNN
   └─ 2D CNN with 2 conv layers + FC layers
   └─ Early stopping, learning rate scheduling
   └─ ~50K parameters

6. Evaluate
   └─ Test set metrics: Accuracy, Precision, Recall, F1, AUC

7. Predict Full Raster
   └─ Sliding window over entire area
   └─ Generate classification + probability maps

8. Save Results
   └─ GeoTIFF rasters with metadata
```

---

## ⚙️ Configuration

Nếu muốn thay đổi settings chi tiết hơn, edit file `src/common/config.py`:

```python
DL_CONFIG = {
    # Model architecture
    'model_type': 'standard',       # 'standard' hoặc 'deeper'
    'patch_size': 3,                 # 3x3 patches (30m x 30m)
    'dropout_rate': 0.5,             # Dropout để tránh overfitting

    # Training parameters
    'epochs': 50,                    # Max epochs (có early stopping)
    'batch_size': 32,                # Batch size
    'learning_rate': 0.001,          # Initial learning rate
    'weight_decay': 1e-4,            # L2 regularization
    'early_stopping_patience': 10,   # Stop nếu val loss không giảm sau 10 epochs

    # Spatial splitting
    'cluster_distance': 50.0,        # Cluster points within 50m
    'train_size': 0.70,              # 70% train
    'val_size': 0.15,                # 15% validation
    'test_size': 0.15,               # 15% test

    # Normalization
    'normalize_method': 'standardize', # 'standardize' or 'minmax'

    # Device
    'device': 'cuda',                # 'cuda' or 'cpu'

    # Prediction
    'pred_batch_size': 1000,         # Batch size cho full raster prediction
    'pred_stride': 1,                # Stride=1 → dense prediction
}
```

---

## 🎯 Spatial-Aware Splitting (Tránh Data Leakage)

### ⚠️ Vấn đề

Nếu có 2 training points gần nhau (<30m):
```
Point A: Patch bao phủ pixels (98-102, 198-202)
Point B: Patch bao phủ pixels (100-104, 200-204)
         → OVERLAP!
```

Nếu Point A ở train set, Point B ở test set:
- Model đã "nhìn thấy" vùng của Point B khi training (qua patch A)
- Test accuracy sẽ bị thổi phồng (không đúng)

### ✅ Giải pháp của chúng ta

```python
1. Cluster các points gần nhau (distance < 50m)
2. Split theo CLUSTER (không phải individual points)
3. Tất cả points trong 1 cluster → cùng ở train hoặc cùng ở test
4. Verify: đảm bảo khoảng cách giữa train/test >= 50m
```

**Kết quả:**
- ✅ NO data leakage
- ✅ Test accuracy phản ánh khả năng generalization thật
- ✅ An toàn với patch size 3x3 (30m)

---

## 📊 Kết quả mong đợi

Dựa trên nghiên cứu tương tự với small dataset:

| Metric | Expected Range | Note |
|--------|----------------|------|
| **Accuracy** | 85-92% | Similar to RF |
| **Precision** | 82-90% | Slightly better than RF |
| **Recall** | 80-88% | Depends on class balance |
| **F1-Score** | 82-89% | Balanced metric |
| **ROC-AUC** | 88-94% | Good discrimination |

**So sánh với Random Forest:**
- Accuracy: Tương đương hoặc hơi cao hơn một chút
- Smoothness: **Rõ rệt tốt hơn** (ít noise lấm tấm)
- Training time: Chậm hơn (~2-3x)

---

## 💡 Tips & Tricks

### 1. **Nếu bị Out of Memory (GPU)**

```python
# Edit config.py
DL_CONFIG['batch_size'] = 16  # Giảm từ 32 xuống 16
```

Hoặc:
```bash
python main_dl.py --batch-size 16 --device cpu
```

### 2. **Nếu bị Overfitting (Val accuracy giảm)**

```python
# Tăng regularization
DL_CONFIG['dropout_rate'] = 0.6      # Từ 0.5 lên 0.6
DL_CONFIG['weight_decay'] = 1e-3     # Từ 1e-4 lên 1e-3
```

Hoặc giảm số epochs:
```bash
python main_dl.py --epochs 30
```

### 3. **Nếu muốn train lâu hơn**

```bash
python main_dl.py --epochs 100
```

Early stopping sẽ tự động dừng nếu không improve.

### 4. **Nếu không có GPU**

```bash
python main_dl.py --device cpu
```

Training sẽ chậm hơn (~30-40 min) nhưng vẫn chạy được.

---

## 🔬 Phân tích kết quả

### 1. **Xem training history**

```python
import json

with open('results/data/cnn_training_history.json', 'r') as f:
    history = json.load(f)

# Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.savefig('results/plots/training_curves.png', dpi=300)
plt.show()
```

### 2. **Load và visualize kết quả**

```python
import rasterio
import matplotlib.pyplot as plt

# Load classification map
with rasterio.open('results/rasters/cnn_classification.tif') as src:
    cnn_classification = src.read(1)

# Load probability map
with rasterio.open('results/rasters/cnn_probability.tif') as src:
    cnn_probability = src.read(1)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(cnn_classification, cmap='RdYlGn')
axes[0].set_title('CNN Classification (Binary)')
axes[0].axis('off')

im = axes[1].imshow(cnn_probability, cmap='RdYlGn_r', vmin=0, vmax=1)
axes[1].set_title('CNN Probability (0-1)')
axes[1].axis('off')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig('results/plots/cnn_results.png', dpi=300)
plt.show()
```

### 3. **So sánh với Random Forest**

```python
# Load RF results
with rasterio.open('results/rasters/rf_classification.tif') as src:
    rf_classification = src.read(1)

# Compare
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(rf_classification, cmap='RdYlGn')
axes[0].set_title('Random Forest')
axes[0].axis('off')

axes[1].imshow(cnn_classification, cmap='RdYlGn')
axes[1].set_title('CNN')
axes[1].axis('off')

# Difference
diff = cnn_classification.astype(int) - rf_classification.astype(int)
axes[2].imshow(diff, cmap='bwr', vmin=-1, vmax=1)
axes[2].set_title('Difference (CNN - RF)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('results/plots/rf_vs_cnn.png', dpi=300)
plt.show()
```

---

## 🐛 Troubleshooting

### Lỗi: "ModuleNotFoundError: No module named 'torch'"

```bash
# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Lỗi: "CUDA out of memory"

```bash
# Giảm batch size
python main_dl.py --batch-size 16

# Hoặc dùng CPU
python main_dl.py --device cpu
```

### Lỗi: "RuntimeError: expected scalar type Float but found Double"

→ Đây là bug trong code. Đã handle sẵn bằng `.float()` conversions.

### Model không học (loss không giảm)

1. Check data normalization: patches phải được normalize
2. Check learning rate: có thể quá cao hoặc quá thấp
3. Check labels: đúng format (0/1) chưa?

---

## 📚 Tài liệu tham khảo

- **Deep Learning Module README**: `src/deep_learning/README.md`
- **Main Pipeline**: `src/main_dl.py`
- **Configuration**: `src/common/config.py` (DL_CONFIG section)
- **Compare with RF**: `src/main.py` (Random Forest pipeline)

---

## ✅ Checklist

Trước khi chạy, đảm bảo:

- [ ] Đã có dữ liệu trong `data/raw/` (Sentinel-1, Sentinel-2, Ground Truth, Boundary)
- [ ] Đã install PyTorch (`pip install torch`)
- [ ] Đã check GPU availability (hoặc sẵn sàng dùng CPU)
- [ ] Đã chạy Random Forest trước (để so sánh)

**Sẵn sàng?**
```bash
cd src
python main_dl.py
```

**Thời gian chạy:** ~15-25 phút (GPU) hoặc ~30-45 phút (CPU)

Good luck! 🚀
