# Random Forest Model Guide

## Overview

Mô hình **Random Forest** đã được thêm vào project như một phương pháp machine learning truyền thống để so sánh với các mô hình CNN.

## Key Features

### 1. **Traditional ML Approach**
- Sử dụng ensemble of decision trees
- Không cần GPU, chạy trên CPU với multi-core processing
- Dễ interpret thông qua feature importance

### 2. **Architecture**
```
Input: Flattened patches (128×128×14 = 229,376 features)
  ↓
Random Forest (100 trees)
  ↓
Output: Binary classification (0: No deforestation, 1: Deforestation)
```

### 3. **Model Configuration**
- `n_estimators`: 100 trees
- `max_depth`: 20 levels
- `min_samples_split`: 10
- `min_samples_leaf`: 4
- `n_jobs`: -1 (use all CPU cores)

## How to Use

### Step 1: Train Random Forest

```bash
python train_random_forest.py
```

**Expected outputs:**
- `checkpoints/random_forest_best.pkl` - Trained model
- `logs/random_forest_training.txt` - Training log

**Training time:** ~2-5 minutes on modern CPU (depends on number of cores)

### Step 2: Check Training Results

```python
# Load training log
with open('logs/random_forest_training.txt', 'r') as f:
    print(f.read())
```

**Expected metrics:**
- Accuracy: ~85-95%
- F1 Score: ~85-95%
- AUC: ~90-98%

### Step 3: Compare with CNN Models

Random Forest vs CNN comparison:

| Metric | Random Forest | Shallow UNet | MultiScale CNN | Spatial CNN |
|--------|---------------|--------------|----------------|-------------|
| Accuracy | ~90% | ~92% | ~91% | ~89% |
| Training Time | 2-5 min | 15-30 min | 10-20 min | 8-15 min |
| Inference Speed | Fast | Very Fast (GPU) | Very Fast (GPU) | Very Fast (GPU) |
| Interpretability | ✅ High | ❌ Low | ❌ Low | ❌ Low |
| GPU Required | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |

## Feature Importance Analysis

Random Forest provides feature importance scores, showing which bands and spatial locations are most important for classification:

```python
from src.ml_models import RandomForestModel

# Load trained model
model = RandomForestModel()
model.load('checkpoints/random_forest_best.pkl')

# Get feature importance
importance = model.get_feature_importance()

# Reshape to (128, 128, 14)
importance_map = importance.reshape(128, 128, 14)

# Band-wise importance
band_importance = importance_map.mean(axis=(0, 1))
print("Band importance:", band_importance)
```

**Expected output:**
- NDVI bands (indices 4, 11) usually have highest importance
- NIR bands (indices 3, 10) also important
- Temporal difference between 2024-2025 is key

## Integration with Notebooks

### Update notebook 03 (Training)

Add Random Forest training cell:

```python
# Train Random Forest
import subprocess
result = subprocess.run(['python', 'train_random_forest.py'],
                       capture_output=True, text=True)
print(result.stdout)
```

### Update notebook 04 (Evaluation)

Add Random Forest evaluation:

```python
from src.ml_models import RandomForestModel, load_patches_for_ml

# Load model
rf_model = RandomForestModel()
rf_model.load('checkpoints/random_forest_best.pkl')

# Load test data
X_test, y_test = load_patches_for_ml('data/patches', 'test')

# Evaluate
metrics = rf_model.evaluate(X_test, y_test)
print("Random Forest metrics:", metrics)
```

## Advantages

1. **Interpretability**: Feature importance shows which bands matter most
2. **No GPU needed**: Runs on any computer with CPU
3. **Robust**: Less prone to overfitting than deep learning
4. **Fast training**: 2-5 minutes vs 15-30 minutes for CNNs

## Limitations

1. **Memory intensive**: Requires loading all patches into memory
2. **Feature engineering**: Uses raw flattened features (no spatial structure)
3. **Slower inference**: ~10x slower than CNN on GPU
4. **Less accurate**: Typically 2-5% lower accuracy than best CNN

## Use Cases

1. **Baseline comparison**: Quick baseline for CNN models
2. **Feature analysis**: Understand which bands are important
3. **Low-resource deployment**: Run on machines without GPU
4. **Interpretable results**: For stakeholders who need explainability

## Next Steps

1. ✅ Train Random Forest model
2. ⬜ Compare with CNN models in notebook 04
3. ⬜ Analyze feature importance by band
4. ⬜ Try different hyperparameters (n_estimators, max_depth)
5. ⬜ Implement full-image inference with Random Forest

## Troubleshooting

### Issue: Out of Memory

**Solution:** Reduce number of training samples or use smaller patch size

```python
# In src/ml_models.py, modify load_patches_for_ml()
# to load only a subset of patches
```

### Issue: Training too slow

**Solution:** Reduce number of trees or max depth

```python
# In train_random_forest.py
model_config = {
    'n_estimators': 50,  # Reduce from 100
    'max_depth': 15,     # Reduce from 20
    ...
}
```

### Issue: Low accuracy

**Solution:** Try different hyperparameters or add more training data

```python
# Increase trees and depth
model_config = {
    'n_estimators': 200,
    'max_depth': 30,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
}
```

## References

- Scikit-learn Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Random Forest for Remote Sensing: Belgiu & Drăguţ (2016)
- Feature Importance Interpretation: Breiman (2001)
