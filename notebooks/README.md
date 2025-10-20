# 📓 Jupyter Notebooks - Forest Change Detection

## 🚀 Quick Start

Chạy lần lượt 3 notebooks sau:

1. **`1_train_models.ipynb`** - Train models
2. **`2_inference_wholescene.ipynb`** - Whole scene inference
3. **`3_create_maps.ipynb`** - Generate final maps

---

## 📋 Chi tiết từng Notebook

### 1️⃣ `1_train_models.ipynb` - Training

**Chức năng:**
- Load dữ liệu từ 4 ảnh TIFF + CSV
- Tạo train/val/test split (80/10/10)
- Train 3 models: UNet-EfficientNet, UNet-MobileNet, FPN-EfficientNet
- Real-time visualization (loss, accuracy, F1)
- Save best checkpoints

**Thời gian:** ~30-60 phút/model (GPU)

**Output:**
```
models/
├── unet_efficientnet/
│   ├── best_model.pth
│   ├── training_history.png
│   └── sample_predictions.png
├── unet_mobilenet/
└── fpn_efficientnet/
```

---

### 2️⃣ `2_inference_wholescene.ipynb` - Inference

**Chức năng:**
- Load trained model
- Sliding window inference (256×256 với overlap 32px)
- Merge predictions → Probability map
- Analyze distribution
- Preview results

**Thời gian:** ~10-30 phút (tùy kích thước ảnh)

**Output:**
```
results/whole_scene/
├── probability_map.npy          # Numpy array
├── preview_maps.png             # Preview
├── probability_analysis.png     # Analysis
└── zoomed_regions.png           # Zoomed samples
```

---

### 3️⃣ `3_create_maps.ipynb` - Final Maps

**Chức năng:**
- Load probability map
- Apply threshold (0.5) → Binary map
- Colorize (Green=No change, Red=Deforestation)
- Save GeoTIFF + PNG outputs
- Statistics summary

**Thời gian:** ~5 phút

**Output:**
```
results/whole_scene/
├── probability_map.tif          # GeoTIFF (Float32, 0.0-1.0)
├── binary_map.tif               # GeoTIFF (UInt8, 0/1)
├── visualization.png            # RGB PNG (150 DPI)
├── visualization_highres.png    # RGB PNG (300 DPI)
└── statistics_summary.png       # Stats & charts
```

---

## ⚙️ Configuration

### Important Settings trong Notebook 1:

```python
# Model to train
MODEL_TO_TRAIN = 'unet_efficientnet'  # or 'unet_mobilenet', 'fpn_efficientnet'

# Training params
BATCH_SIZE = 16
NUM_WORKERS = 0  # IMPORTANT: Keep 0 for Windows/Jupyter
PATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-4
```

### ⚠️ Windows/Jupyter Note:

**QUAN TRỌNG:** Giữ `NUM_WORKERS = 0` khi chạy trong Jupyter trên Windows để tránh lỗi multiprocessing pickling.

- ✅ `NUM_WORKERS = 0` - Safe cho Windows/Jupyter
- ❌ `NUM_WORKERS > 0` - Sẽ bị lỗi `TypeError: cannot be converted to a Python object for pickling`

Nếu chạy như Python script trên Linux, có thể tăng lên `NUM_WORKERS = 4` để tăng tốc.

---

## 🎯 Expected Results

### Model Performance (Test set):
- **Accuracy**: 85-90%
- **F1-Score**: 0.85-0.90
- **IoU**: 0.75-0.85

### Final Maps:
- **Probability map**: Xác suất mất rừng [0.0-1.0] cho mỗi pixel
- **Binary map**: Phân loại rõ ràng (0=No change, 1=Deforestation)
- **Visualization**: Bản đồ màu (Xanh/Đỏ) dễ hiểu

---

## 🔧 Troubleshooting

### Lỗi: `TypeError: self._hds cannot be converted to a Python object for pickling`

**Nguyên nhân:** `NUM_WORKERS > 0` trên Windows/Jupyter

**Giải pháp:**
```python
NUM_WORKERS = 0  # Set this in notebook cell 2
```

### Lỗi: `CUDA out of memory`

**Giải pháp:** Giảm batch size
```python
BATCH_SIZE = 8  # Hoặc 4 nếu vẫn lỗi
```

### Lỗi: `FileNotFoundError` cho CSV/TIFF

**Kiểm tra:**
```python
# Cell trong notebook
print(f"CSV exists: {CSV_PATH.exists()}")
print(f"S1 T1 exists: {S1_T1_PATH.exists()}")
print(f"S2 T1 exists: {S2_T1_PATH.exists()}")
```

---

## 📊 Workflow Summary

```
┌─────────────────────────────────────────┐
│  1. 1_train_models.ipynb                │
│  - Train 3 models                       │
│  - Save checkpoints                     │
│  Output: models/*/best_model.pth        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  2. 2_inference_wholescene.ipynb        │
│  - Load best model                      │
│  - Sliding window inference             │
│  Output: probability_map.npy            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  3. 3_create_maps.ipynb                 │
│  - Load probability map                 │
│  - Create final outputs                 │
│  Output: 3 GeoTIFFs + PNGs              │
└─────────────────────────────────────────┘
```

---

## 💡 Tips

1. **Chạy cell từ trên xuống** - Không skip cells
2. **Check GPU usage** - Mở Task Manager → Performance → GPU
3. **Monitor training** - Quan sát loss curves để phát hiện overfitting
4. **Save checkpoints** - Models tự động save mỗi 10 epochs
5. **Early stopping** - Training tự dừng nếu không improve sau 10 epochs

---

## 🎨 Customization

### Thay đổi model:

```python
# Trong cell 2 của notebook 1
MODEL_TO_TRAIN = 'unet_mobilenet'  # Lightest, fastest
MODEL_TO_TRAIN = 'unet_efficientnet'  # Balanced (recommended)
MODEL_TO_TRAIN = 'fpn_efficientnet'  # Highest accuracy
```

### Thay đổi threshold:

```python
# Trong cell 2 của notebook 3
THRESHOLD = 0.5  # Default
THRESHOLD = 0.3  # More sensitive (more deforestation detected)
THRESHOLD = 0.7  # More conservative (less deforestation detected)
```

### Train all 3 models:

Uncomment cell cuối cùng trong notebook 1 để train cả 3 models tuần tự.

---

Happy coding! 🎉
