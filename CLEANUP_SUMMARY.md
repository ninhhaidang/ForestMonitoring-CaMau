# 🧹 PROJECT CLEANUP SUMMARY

**Date**: 2025-10-18
**Action**: Dọn dẹp project, giữ lại data + files quan trọng

---

## ✅ GIỮ LẠI (KEPT)

### 📊 Data Gốc
```
data/raw/
├── sentinel1/
│   ├── S1_2024_02_04_matched_S2_2024_01_30.tif (490MB)
│   └── S1_2025_02_22_matched_S2_2025_02_28.tif (489MB)
├── sentinel2/
│   ├── S2_2024_01_30.tif (1.5GB)
│   └── S2_2025_02_28.tif (1.5GB)
└── ground_truth/
    ├── Training_Points_CSV.csv (1,285 điểm)
    └── Training_Points__SHP.shp (+ các files .dbf, .prj, etc.)
```

### 📄 Files Quan Trọng
- `environment.yml` - Conda environment
- `requirements.txt` - Python dependencies
- `LICENSE` - MIT License
- `.gitignore` - Git ignore rules
- `.claude/` - Claude Code settings

---

## 📦 ĐÃ ARCHIVE (MOVED TO _archive/)

### _archive/old_project/
- `processed/` - Patches cũ (1,285 samples đã cắt)
- `experiments/` - SNUNet training results
- `predictions/` - Visualizations cũ
- `evaluation_results/` - Evaluation outputs
- `configs/` - Open-CD configs
- `notebooks/` - Old notebooks
- `results/` - Old results
- `src/` - Old source code

### _archive/old_scripts/
- `train_camau.py`
- `evaluate.py`
- `monitor_progress.py`
- `inference_and_save.py`
- `create_maps_from_predictions.py`
- `create_overlay_maps.py`
- `visualize_predictions.py`
- `rename_predictions.py`
- Và các scripts khác...

### _archive/old_docs/
- `README.md` (cũ)
- `MODEL_ANALYSIS.md`
- `SNUNET_RESULTS.md`

---

## 🆕 CẤU TRÚC MỚI (CLEAN)

```
project/
├── data/
│   └── raw/              # ✅ Data gốc (không thay đổi)
│
├── notebooks/            # 🆕 Jupyter notebooks mới
│   ├── 1_train_models.ipynb
│   ├── 2_inference_wholescene.ipynb
│   └── 3_create_maps.ipynb
│
├── src/                  # 🆕 Source code mới
│   ├── dataset.py
│   ├── models.py
│   └── utils.py
│
├── models/               # 🆕 Saved models
│   ├── unet_efficientnet/
│   ├── unet_mobilenet/
│   └── fpn_efficientnet/
│
├── results/              # 🆕 Outputs
│   ├── whole_scene/
│   │   ├── probability_map.tif
│   │   ├── binary_map.tif
│   │   └── visualization.png
│   └── model_comparison/
│
├── _archive/             # 📦 Backup files cũ
│
├── environment.yml
├── requirements.txt
├── LICENSE
├── .gitignore
├── README.md             # 🆕 README mới
├── PROJECT_RESTRUCTURE.md
└── SETUP_NEW_PIPELINE.md
```

---

## ⚠️ LƯU Ý

### 1. Folder `open-cd/`
- **Không thể move** do permission issue
- Bạn có thể tự xóa nếu không cần: `rm -rf open-cd/`
- Hoặc giữ lại (không ảnh hưởng pipeline mới)

### 2. Folder `data/patches/`
- Đã tạo folders trống: `train/`, `val/`, `test/`
- Sẽ tự động populate khi chạy notebooks

### 3. Backup trong `_archive/`
- Tất cả files cũ đều được backup
- Không mất bất kỳ data/code nào
- Có thể restore bất cứ lúc nào

---

## 🚀 TIẾP THEO

### Bước 1: Cài đặt thư viện mới
```bash
pip install segmentation-models-pytorch
```

### Bước 2: Mở Jupyter
```bash
jupyter notebook
```

### Bước 3: Chạy notebooks
1. `notebooks/1_train_models.ipynb`
2. `notebooks/2_inference_wholescene.ipynb`
3. `notebooks/3_create_maps.ipynb`

---

## 📊 DISK SPACE SAVED

| Item | Size | Status |
|------|------|--------|
| Data gốc (raw) | ~5.5GB | ✅ Kept |
| Patches cũ (processed) | ~400MB | 📦 Archived |
| Experiments | ~100MB | 📦 Archived |
| Predictions | ~50MB | 📦 Archived |
| Scripts/Docs | ~5MB | 📦 Archived |

**Total archived**: ~555MB
**Data safe**: ✅ All original data intact

---

Bạn đã sẵn sàng để bắt đầu pipeline mới!
