# CẤU TRÚC THƯ MỤC MỚI - FOREST CHANGE DETECTION

## 📁 CLEAN STRUCTURE

```
25-26_HKI_DATN_21021411_DangNH/
│
├── 📂 data/
│   ├── raw/                    # ← Data gốc (BẠN SẼ ĐẶT Ở ĐÂY)
│   │   ├── S1_T1.tif          # Sentinel-1 Time 1 (VH, Ratio)
│   │   ├── S1_T2.tif          # Sentinel-1 Time 2 (VH, Ratio)
│   │   ├── S2_T1.tif          # Sentinel-2 Time 1 (7 bands)
│   │   ├── S2_T2.tif          # Sentinel-2 Time 2 (7 bands)
│   │   └── training_points.csv # 1285 điểm (id, label, x, y)
│   │
│   └── patches/                # ← Patches extracted (sẽ tự động tạo)
│       ├── train/             # 80% = ~1028 samples
│       ├── val/               # 10% = ~128 samples
│       └── test/              # 10% = ~129 samples
│
├── 📂 models/                  # ← Saved models
│   ├── unet_efficientnet/
│   ├── unet_mobilenet/
│   └── fpn_efficientnet/
│
├── 📂 results/                 # ← Outputs
│   ├── whole_scene/           # Bản đồ toàn bộ vùng
│   │   ├── probability_map.tif
│   │   ├── binary_map.tif
│   │   └── visualization.png
│   ├── model_comparison/      # So sánh 3 models
│   └── training_logs/         # Training history
│
├── 📂 src/                     # ← Source code
│   ├── 1_extract_patches.py   # Extract từ 4 ảnh + CSV
│   ├── 2_train.py             # Train 3 models
│   ├── 3_inference.py         # Sliding window inference
│   ├── 4_create_maps.py       # Tạo outputs
│   ├── dataset.py             # Custom PyTorch Dataset
│   └── utils.py               # Helper functions
│
├── 📂 _archive/                # ← Thư mục archive (files cũ)
│   ├── old_experiments/
│   ├── old_scripts/
│   └── old_predictions/
│
├── environment.yml
├── requirements.txt
└── README.md
```

---

## 🔄 WORKFLOW MỚI

### Step 1: Chuẩn bị data
```bash
# Bạn đặt 4 ảnh TIFF + CSV vào data/raw/
python src/1_extract_patches.py
# → Tạo patches trong data/patches/
```

### Step 2: Train models
```bash
python src/2_train.py --model all
# → Train cả 3 models: UNet-Eff, UNet-Mobile, FPN-Eff
# → Save checkpoints vào models/
```

### Step 3: Inference whole scene
```bash
python src/3_inference.py --model unet_efficientnet
# → Sliding window inference
# → Merge predictions
```

### Step 4: Create final maps
```bash
python src/4_create_maps.py
# → probability_map.tif (0.0 - 1.0)
# → binary_map.tif (0/1)
# → visualization.png (xanh/đỏ)
```

---

## 📦 THƯ VIỆN CẦN CÀI

```bash
pip install segmentation-models-pytorch
pip install albumentations
pip install rasterio
pip install pandas
pip install matplotlib
pip install tqdm
pip install pytorch-lightning  # Optional, for easier training
```

---

## ✅ ACTION ITEMS

1. **Di chuyển files cũ vào _archive/**
2. **Tạo cấu trúc mới**
3. **Bạn chuẩn bị:**
   - 4 ảnh TIFF lớn
   - 1 file CSV (1285 points)
   - Đặt vào `data/raw/`
4. **Tôi viết code pipeline**

---

Bạn đồng ý cấu trúc này không? Tôi sẽ bắt đầu dọn dẹp và tạo lại!
