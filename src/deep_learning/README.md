# Deep Learning Module

Patch-based 2D CNN for deforestation detection with spatial context.

## 📁 Module Structure

```
deep_learning/
├── __init__.py              # Module initialization
├── patch_extractor.py       # Extract 3x3 patches from feature stack
├── spatial_split.py         # Spatial-aware data splitting (avoid leakage)
├── cnn_model.py             # 2D CNN architecture
├── train.py                 # Training pipeline
├── predict.py               # Full raster prediction
└── README.md                # This file
```

## 🎯 Key Features

### 1. **Spatial Context**
- Uses **3x3 patches** (30m x 30m) instead of single pixels
- CNN learns from spatial patterns in the neighborhood
- Reduces "salt-and-pepper" noise in classification

### 2. **Spatial-Aware Splitting**
- **Group-aware splitting** to avoid data leakage
- Clusters nearby points (<50m) and keeps them in same split
- Ensures train/val/test sets are spatially separated

### 3. **Lightweight CNN Architecture**
- Only **~50K parameters** (suitable for small dataset)
- 2 conv layers + global pooling + FC layers
- Heavy regularization: Dropout + BatchNorm + Weight Decay

### 4. **Training Features**
- **Early stopping** to prevent overfitting
- **Learning rate scheduling** (ReduceLROnPlateau)
- **Class weighting** for imbalanced data
- **GPU support** (CUDA) for faster training

## 🚀 Usage

### Quick Start

```bash
# Run full CNN pipeline
cd src
python main_dl.py

# Custom settings
python main_dl.py --epochs 100 --batch-size 64 --device cuda
```

### Step-by-Step Usage

```python
# 1. Load data (same as Random Forest)
from common.data_loader import DataLoader
from common.feature_extraction import FeatureExtraction

loader = DataLoader()
s2_before, s2_after = loader.load_sentinel2()
s1_before, s1_after = loader.load_sentinel1()
ground_truth = loader.load_ground_truth()

# Extract features
extractor = FeatureExtraction()
feature_stack, valid_mask = extractor.extract_features(
    s2_before, s2_after, s1_before, s1_after
)

# 2. Spatial-aware splitting
from deep_learning.spatial_split import SpatialSplitter

splitter = SpatialSplitter(cluster_distance=50.0)
train_idx, val_idx, test_idx, metadata = splitter.spatial_split(
    ground_truth, stratify_by_class=True, verify=True
)

# 3. Extract patches
from deep_learning.patch_extractor import PatchExtractor

patch_extractor = PatchExtractor(patch_size=3)
patches, labels, valid_indices = patch_extractor.extract_patches_at_points(
    feature_stack, ground_truth, transform, valid_mask
)
patch_extractor.normalize_patches(method='standardize')

# 4. Create and train model
from deep_learning.cnn_model import create_model
from deep_learning.train import CNNTrainer

model = create_model(model_type='standard', patch_size=3, n_features=27)
trainer = CNNTrainer(model, device='cuda', learning_rate=0.001)

history = trainer.fit(
    X_train, y_train,
    X_val, y_val,
    epochs=50,
    batch_size=32,
    early_stopping_patience=10
)

# 5. Evaluate
test_metrics = trainer.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

# 6. Predict full raster
from deep_learning.predict import RasterPredictor

predictor = RasterPredictor(model, device='cuda', patch_size=3)
classification_map, probability_map = predictor.predict_raster(
    feature_stack, valid_mask, stride=1, normalize=True
)

# Save results
predictor.save_rasters(
    'results/rasters/cnn_classification.tif',
    'results/rasters/cnn_probability.tif',
    reference_metadata
)
```

## ⚙️ Configuration

Edit `src/common/config.py` → `DL_CONFIG` section:

```python
DL_CONFIG = {
    # Model architecture
    'model_type': 'standard',       # or 'deeper'
    'patch_size': 3,                 # 3x3 patches
    'dropout_rate': 0.5,             # Regularization

    # Training
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'early_stopping_patience': 10,

    # Spatial splitting
    'cluster_distance': 50.0,        # meters
    'train_size': 0.70,
    'val_size': 0.15,
    'test_size': 0.15,

    # Device
    'device': 'cuda',                # or 'cpu'
}
```

## 📊 Model Architecture

### Standard Model (Default)

```
Input: (batch, 3, 3, 27)
    ↓
Conv2D(64, 3x3) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Conv2D(32, 3x3) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Global Average Pooling → (batch, 32)
    ↓
Dense(64) + BatchNorm + ReLU + Dropout(0.5)
    ↓
Dense(2) → Logits
    ↓
Softmax → Probabilities
```

**Parameters:** ~50K (suitable for small dataset)

### Deeper Model (Alternative)

3 conv layers instead of 2, ~80K parameters. Use with caution on small datasets.

## 🔍 Spatial Leakage Prevention

### The Problem
If training points are close together (<50m), their 3x3 patches overlap:

```
Point A: patch covers pixels (98-102, 198-202)
Point B: patch covers pixels (100-104, 200-204)
         → OVERLAP!
```

If A is in train and B is in test → model already "saw" test data during training → inflated accuracy!

### Our Solution: Cluster-based Splitting

```python
1. Cluster nearby points (distance < 50m)
2. Split by CLUSTERS, not individual points
3. All points in a cluster go to same split
4. Verify spatial separation between splits
```

Result: **NO overlap** between train/val/test patches ✅

## 📈 Expected Performance

Based on similar studies with small datasets:

| Metric | Expected Range |
|--------|----------------|
| **Accuracy** | 85-92% |
| **Precision** | 82-90% |
| **Recall** | 80-88% |
| **F1-Score** | 82-89% |

**Comparison with Random Forest:**
- **Accuracy**: Similar or slightly better
- **Smoothness**: Significantly better (less noise)
- **Training time**: Slower (~15-20 min vs 5-10 min)
- **Interpretability**: Lower

## 💡 Tips for Best Results

### 1. **Prevent Overfitting**
- Use dropout (0.5 recommended)
- Early stopping is crucial
- Don't train too long (50 epochs usually enough)

### 2. **Learning Rate**
- Start with 0.001
- Scheduler will reduce automatically if validation loss plateaus

### 3. **Batch Size**
- 32 is good for small datasets
- Larger (64) if you have GPU memory
- Smaller (16) if overfitting

### 4. **Class Imbalance**
- Enable `use_class_weights: True` in config
- Automatically computed from training data

### 5. **GPU vs CPU**
- GPU: ~15-20 min training
- CPU: ~30-40 min training
- Prediction time similar (both fast)

## 🐛 Troubleshooting

### Out of Memory (GPU)
```python
# Reduce batch size
DL_CONFIG['batch_size'] = 16  # or 8
```

### Overfitting (Val accuracy drops)
```python
# Increase regularization
DL_CONFIG['dropout_rate'] = 0.6
DL_CONFIG['weight_decay'] = 1e-3

# Reduce model capacity
DL_CONFIG['model_type'] = 'standard'  # not 'deeper'
```

### Underfitting (Low accuracy)
```python
# Train longer
DL_CONFIG['epochs'] = 100

# Larger model
DL_CONFIG['model_type'] = 'deeper'

# Check data quality
# - Are patches normalized?
# - Are labels correct?
```

## 📦 Output Files

After running `python main_dl.py`:

```
results/
├── rasters/
│   ├── cnn_classification.tif     # Binary map
│   └── cnn_probability.tif        # Probability map
├── models/
│   └── cnn_model.pth              # Trained model checkpoint
└── data/
    ├── cnn_training_patches.npz   # Saved patches
    ├── cnn_evaluation_metrics.json
    └── cnn_training_history.json  # Loss/accuracy curves
```

## 🔗 Related Files

- `src/main_dl.py` - Main pipeline entry point
- `src/common/config.py` - Configuration (DL_CONFIG section)
- `src/random_forest/` - Compare with Random Forest baseline

## 📚 References

- Architecture inspired by: https://arxiv.org/abs/1411.4038
- Spatial splitting: https://doi.org/10.1016/j.isprsjprs.2020.01.001
- Remote sensing with small data: https://doi.org/10.3390/rs13040629
