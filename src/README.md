# SOURCE CODE STRUCTURE

## 📁 Cấu trúc thư mục

```
src/
├── common/                          # Shared modules for all models
│   ├── __init__.py
│   ├── config.py                    # Global configuration
│   ├── data_loader.py               # Load S1, S2, GT, boundary
│   ├── feature_extraction.py        # Feature extraction (27 features)
│   ├── evaluation.py                # Model evaluation metrics
│   ├── visualization.py             # Plotting và visualization
│   └── utils.py                     # Utility functions
│
├── random_forest/                   # Random Forest model
│   ├── __init__.py
│   ├── train.py                     # Training pipeline
│   └── predict.py                   # Full raster prediction
│
├── _deprecated/                     # Old files (backup)
└── main.py                          # Main entry point
```

---

## 🔄 Migration từ old structure

### Old structure (Flat):
```
src/
├── config.py
├── step1_2_setup_and_load_data.py
├── common/feature_extraction.py
├── step4_extract_training_data.py
├── step5_train_random_forest.py
├── step6_model_evaluation.py
├── step7_predict_full_raster.py
├── step8_visualization.py
└── main.py
```

### New structure (Model-centric):
```
src/
├── common/
│   ├── config.py                    # từ config.py
│   ├── data_loader.py               # từ step1_2_*.py
│   ├── feature_extraction.py        # từ step3_*.py
│   ├── evaluation.py                # từ step6_*.py
│   ├── visualization.py             # từ step9_*.py
│   └── utils.py                     # NEW
│
├── random_forest/
│   ├── train.py                     # từ step4_* + step5_*
│   └── predict.py                   # từ step7_*
│
└── main.py                          # Refactored main.py
```

---

## 📦 Modules chi tiết

### **common/** - Shared modules

#### `config.py`
- Global configuration cho toàn bộ project
- Paths, parameters, feature names
- Output files configuration

#### `data_loader.py`
- **Class:** `DataLoader`
- **Chức năng:** Load Sentinel-1, Sentinel-2, ground truth, boundary
- **Methods:**
  - `load_sentinel2()` - Load S2 before/after
  - `load_sentinel1()` - Load S1 before/after
  - `load_ground_truth()` - Load GT CSV
  - `load_boundary()` - Load boundary shapefile

#### `feature_extraction.py`
- **Class:** `FeatureExtraction`
- **Chức năng:** Tạo 27 features từ S1/S2
- **Methods:**
  - `extract_features()` - Create all features
  - `get_feature_summary()` - Get statistics

#### `evaluation.py`
- **Class:** `ModelEvaluator`
- **Chức năng:** Evaluate model performance
- **Methods:**
  - `evaluate_validation()` - Validate metrics
  - `evaluate_test()` - Test metrics
  - `cross_validate()` - K-Fold CV
  - `calculate_feature_importance()` - Feature rankings

#### `visualization.py`
- **Class:** `Visualizer`
- **Chức năng:** Create plots và visualizations
- **Methods:**
  - `create_all_visualizations()` - All plots
  - `plot_confusion_matrices()` - Confusion matrices
  - `plot_roc_curve()` - ROC curve
  - `plot_feature_importance()` - Feature importance

---

### **random_forest/** - Random Forest model

#### `train.py`
- **Class:** `TrainingDataExtractor`
  - Extract features at GT points
  - Split train/val/test
  - Data quality checks

- **Class:** `RandomForestTrainer`
  - Train RF model (100 trees)
  - Save trained model
  - Get feature importance

#### `predict.py`
- **Class:** `RasterPredictor`
  - Predict on full raster (batch processing)
  - Generate binary + probability maps
  - Save GeoTIFF files

---

## 🚀 Cách sử dụng

### Option 1: Chạy full pipeline (NEW)

```bash
cd src
python main.py
```

### Option 2: Import modules trong Python/Notebook

```python
# Import common modules
from common.data_loader import DataLoader
from common.feature_extraction import FeatureExtraction
from common.evaluation import ModelEvaluator

# Import RF modules
from random_forest.train import RandomForestTrainer, TrainingDataExtractor
from random_forest.predict import RasterPredictor

# Use them
loader = DataLoader()
s2_before, s2_after = loader.load_sentinel2()
```

---

## ✅ Lợi ích của new structure

1. **Scalability** - Dễ thêm models mới (CNN, U-Net)
2. **Code reuse** - Shared modules dùng chung cho nhiều models
3. **Organization** - Clear separation of concerns
4. **Maintainability** - Easier to find and update code
5. **Professional** - Industry standard structure

---

## 🔮 Future: Thêm Deep Learning models

Khi thêm CNN/U-Net, structure sẽ như:

```
src/
├── common/                    # Shared (no changes)
│   ├── data_loader.py
│   ├── feature_extraction.py
│   ├── evaluation.py
│   └── visualization.py
│
├── random_forest/             # RF (no changes)
│   ├── train.py
│   └── predict.py
│
├── cnn/                       # NEW: CNN model
│   ├── __init__.py
│   ├── model.py               # CNN architecture
│   ├── dataset.py             # PyTorch Dataset
│   ├── train.py               # Training loop
│   └── predict.py             # Inference
│
├── unet/                      # NEW: U-Net model
│   ├── __init__.py
│   ├── model.py               # U-Net architecture
│   ├── train.py               # Training loop
│   └── predict.py             # Inference
│
└── main.py                    # Entry point for all models
```

**Command để chọn model:**
```bash
python main.py --model random_forest
python main.py --model cnn
python main.py --model unet
```

---

## 📝 Notes

- **OLD files** (step*.py, old main.py, old config.py) đã được chuyển vào `_deprecated/` folder
- **NEW structure** hiện đang được sử dụng với main.py mới
- Notebook cần update imports để dùng new structure
- Old files vẫn có thể truy cập tại `src/_deprecated/` nếu cần tham khảo

---

**Author:** Ninh Hải Đăng
**Date:** 07/01/2025
**Version:** 2.0 (Model-centric architecture)
