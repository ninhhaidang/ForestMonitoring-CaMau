# 🚀 FOREST CHANGE DETECTION - PYTORCH PIPELINE

**Pipeline mới với Jupyter Notebooks - Dễ theo dõi và visualize**

---

## 📁 CẤU TRÚC THƯ MỤC

```
project/
├── 📂 data/
│   ├── raw/                    # ← ĐẶT DATA CỦA BẠN Ở ĐÂY
│   │   ├── S1_T1.tif          # Sentinel-1 Time 1 (2 bands: VH, Ratio)
│   │   ├── S1_T2.tif          # Sentinel-1 Time 2 (2 bands: VH, Ratio)
│   │   ├── S2_T1.tif          # Sentinel-2 Time 1 (7 bands)
│   │   ├── S2_T2.tif          # Sentinel-2 Time 2 (7 bands)
│   │   └── training_points.csv # 1285 points (id, label, x, y)
│   │
│   └── patches/                # Tự động tạo
│       ├── train/             # 80% = 1028 samples
│       ├── val/               # 10% = 128 samples
│       └── test/              # 10% = 129 samples
│
├── 📂 notebooks/               # ⭐ NOTEBOOKS CHÍNH
│   ├── 1_extract_patches.ipynb      # Bước 1: Extract patches từ CSV
│   ├── 2_train_models.ipynb         # Bước 2: Train 3 models
│   ├── 3_inference_wholescene.ipynb # Bước 3: Sliding window inference
│   └── 4_create_final_maps.ipynb    # Bước 4: Tạo 3 outputs
│
├── 📂 models/                  # Saved models
│   ├── unet_efficientnet/
│   ├── unet_mobilenet/
│   └── fpn_efficientnet/
│
├── 📂 results/                 # Outputs
│   ├── whole_scene/
│   │   ├── probability_map.tif      # Xác suất [0.0-1.0]
│   │   ├── binary_map.tif           # Nhị phân (0/1)
│   │   └── visualization.png        # Màu sắc (xanh/đỏ)
│   └── model_comparison/
│
└── 📂 src/                     # Helper modules
    ├── dataset.py             # PyTorch Dataset
    ├── models.py              # Model definitions
    └── utils.py               # Helper functions
```

---

## 🔧 SETUP

### 1. Cài đặt thư viện

```bash
# Cài segmentation_models_pytorch
pip install segmentation-models-pytorch

# Các thư viện khác
pip install albumentations rasterio pandas matplotlib tqdm
pip install torch torchvision  # Nếu chưa có
pip install jupyter ipywidgets  # Cho notebooks
```

### 2. Chuẩn bị data

**Bạn cần:**
- ✅ 4 ảnh TIFF lớn (whole scene):
  - S1_T1.tif, S1_T2.tif (Sentinel-1)
  - S2_T1.tif, S2_T2.tif (Sentinel-2)
- ✅ 1 file CSV: `training_points.csv`
  ```csv
  id,label,x,y
  1,0,105.123,8.456
  2,1,105.234,8.567
  ...
  ```

**Đặt vào:** `data/raw/`

---

## 📊 WORKFLOW (4 NOTEBOOKS)

### **Notebook 1: Extract Patches** 📦
```bash
jupyter notebook notebooks/1_extract_patches.ipynb
```

**Features:**
- ✅ Load 4 ảnh TIFF + CSV
- ✅ Extract patches 256×256 từ tọa độ (x, y)
- ✅ Progress bar real-time
- ✅ Visualize samples
- ✅ Auto split train/val/test (80/10/10)

**Output:** `data/patches/{train,val,test}/`

---

### **Notebook 2: Train 3 Models** 🎯
```bash
jupyter notebook notebooks/2_train_models.ipynb
```

**3 Models:**
1. **UNet + EfficientNet-B0** (Cân bằng - 5M params)
2. **UNet + MobileNetV2** (Nhẹ nhất - 2M params)
3. **FPN + EfficientNet-B0** (Accuracy cao - 6M params)

**Features:**
- ✅ Train cả 3 models hoặc chọn 1 model
- ✅ Real-time loss/accuracy plots
- ✅ Progress bar cho mỗi epoch
- ✅ Visualize predictions during training
- ✅ Auto save best model
- ✅ Early stopping

**Output:** `models/{model_name}/best.pth`

---

### **Notebook 3: Inference Whole Scene** 🗺️
```bash
jupyter notebook notebooks/3_inference_wholescene.ipynb
```

**Features:**
- ✅ Load best model
- ✅ Sliding window 256×256 trên toàn bộ ảnh
- ✅ Progress bar real-time
- ✅ Merge predictions → Probability map
- ✅ Visualize progress

**Output:** Probability map (numpy array hoặc partial results)

---

### **Notebook 4: Create Final Maps** 🎨
```bash
jupyter notebook notebooks/4_create_final_maps.ipynb
```

**Features:**
- ✅ Load probability map
- ✅ Apply threshold → Binary map
- ✅ Colorize → Visualization
- ✅ Save 3 outputs:
  - `probability_map.tif` (float32, 0.0-1.0)
  - `binary_map.tif` (uint8, 0/1)
  - `visualization.png` (RGB, xanh/đỏ)
- ✅ Statistics & histogram

**Output:** `results/whole_scene/`

---

## 📈 FEATURES CỦA NOTEBOOKS

### Real-time Visualization:
- 📊 Loss/Accuracy curves
- 📸 Sample predictions
- 🎯 Confusion matrix
- 📉 Learning rate schedule
- ⏱️ Progress bars (tqdm)

### Interactive:
- 🔧 Adjust parameters
- 🎨 Visualize any layer
- 📊 Compare models
- 💾 Export results

### Auto-save:
- 💾 Checkpoints
- 📊 Training history
- 🖼️ Plots

---

## 🎯 QUICK START

```bash
# 1. Đặt data vào data/raw/

# 2. Mở Jupyter
jupyter notebook

# 3. Chạy lần lượt 4 notebooks:
#    → 1_extract_patches.ipynb
#    → 2_train_models.ipynb
#    → 3_inference_wholescene.ipynb
#    → 4_create_final_maps.ipynb

# 4. Kết quả ở results/whole_scene/
```

---

## ⚙️ CONFIGURATION

Các tham số có thể điều chỉnh trong notebooks:

```python
# Training config
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
IMG_SIZE = 256

# Model config
MODEL_NAME = 'unet_efficientnet'  # hoặc 'unet_mobilenet', 'fpn_efficientnet'
ENCODER_WEIGHTS = 'imagenet'

# Inference config
TILE_SIZE = 256
OVERLAP = 32  # Overlap để smooth edges
THRESHOLD = 0.5  # Ngưỡng cho binary map
```

---

## 📊 EXPECTED OUTPUT

```
results/whole_scene/
├── probability_map.tif     # Float32, values in [0.0, 1.0]
├── binary_map.tif          # UInt8, values in {0, 1}
└── visualization.png       # RGB image (Green=0, Red=1)
```

**Statistics:**
- Tổng pixels: X
- Không mất rừng: Y (Z%)
- Mất rừng: W (V%)

---

Sẵn sàng chưa? Tôi sẽ tạo 4 notebooks ngay!
