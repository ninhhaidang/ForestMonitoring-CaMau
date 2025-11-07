# 🚀 Getting Started - Deep Learning Pipeline

**Quick checklist để chạy Deep Learning pipeline lần đầu tiên**

---

## ✅ Pre-flight Checklist

### 1. Kiểm tra dữ liệu

```bash
# Check data files
ls data/raw/sentinel-1/
ls data/raw/sentinel-2/
ls data/raw/ground_truth/
ls data/raw/boundary/
```

**Expected files:**
- ✅ `S1_2024_02_04_matched_S2_2024_01_30.tif`
- ✅ `S1_2025_02_22_matched_S2_2025_02_28.tif`
- ✅ `S2_2024_01_30.tif`
- ✅ `S2_2025_02_28.tif`
- ✅ `Training_Points_CSV.csv`
- ✅ `forest_boundary.shp`

---

### 2. Kiểm tra Python environment

```bash
# Check Python version (should be 3.8-3.11)
python --version

# Check if packages are installed
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import numpy, pandas, sklearn, rasterio; print('Core packages: OK')"
```

**Expected output:**
- ✅ Python 3.8+
- ✅ PyTorch 2.0+
- ✅ CUDA available: True (hoặc False nếu dùng CPU)
- ✅ Core packages: OK

---

### 3. (Optional) Test modules trước

```bash
cd src
python test_dl_modules.py
```

**Expected:** All tests pass ✅

---

## 🎯 Run Pipeline

### Option 1: Default settings (Recommended)

```bash
cd src
python main_dl.py
```

**Settings:**
- Patch size: 3×3
- Epochs: 50 (early stopping enabled)
- Batch size: 32
- Device: CUDA (auto-fallback to CPU)

**Expected time:**
- With GPU: ~15-20 minutes
- With CPU: ~30-40 minutes

---

### Option 2: Custom settings

```bash
# More epochs
python main_dl.py --epochs 100

# Larger batch size (if GPU has enough memory)
python main_dl.py --batch-size 64

# Force CPU
python main_dl.py --device cpu

# Combine
python main_dl.py --epochs 100 --batch-size 64 --device cuda
```

---

## 📊 Monitor Progress

Trong quá trình chạy, bạn sẽ thấy output như:

```
======================================================================
CNN DEFORESTATION DETECTION PIPELINE
Patch-based 2D CNN with Spatial Context
======================================================================

======================================================================
STEP 1-2: LOADING DATA
======================================================================
Loading Sentinel-2 images...
  ✓ S2 Before: (7, 2048, 2048)
  ✓ S2 After: (7, 2048, 2048)
...

======================================================================
STEP 3: FEATURE EXTRACTION
======================================================================
Engineering Sentinel-2 features...
  ✓ Total S2 features: 21
...

======================================================================
STEP 4: SPATIAL-AWARE DATA SPLITTING
======================================================================
Clustering results:
  Total points: 1285
  Number of clusters: 245
...

======================================================================
STEP 5: EXTRACT PATCHES
======================================================================
Valid patches extracted: 1247
Success rate: 97.04%
...

======================================================================
STEP 6: TRAIN CNN MODEL
======================================================================
Epoch   1/50 | Train Loss: 0.6523 | Train Acc:  62.45% | Val Loss: 0.6201 | Val Acc:  67.83%
Epoch   2/50 | Train Loss: 0.5834 | Train Acc:  69.12% | Val Loss: 0.5456 | Val Acc:  73.91%
  → New best model! Val Loss: 0.5456, Val Acc: 73.91%
...
```

---

## ✅ Check Results

Sau khi chạy xong:

### 1. Check output files

```bash
ls results/rasters/
ls results/models/
ls results/data/
```

**Expected files:**
```
results/
├── rasters/
│   ├── cnn_classification.tif      ✓
│   └── cnn_probability.tif         ✓
├── models/
│   └── cnn_model.pth               ✓
└── data/
    ├── cnn_training_patches.npz    ✓
    ├── cnn_evaluation_metrics.json ✓
    └── cnn_training_history.json   ✓
```

---

### 2. Check metrics

```bash
# View metrics
cat results/data/cnn_evaluation_metrics.json
```

**Expected output:**
```json
{
  "accuracy": 0.8756,
  "precision": 0.8523,
  "recall": 0.8912,
  "f1_score": 0.8713,
  "roc_auc": 0.9234
}
```

---

### 3. Load and visualize (Python)

```python
import rasterio
import matplotlib.pyplot as plt
import json

# Load results
with rasterio.open('results/rasters/cnn_classification.tif') as src:
    classification = src.read(1)

with rasterio.open('results/rasters/cnn_probability.tif') as src:
    probability = src.read(1)

with open('results/data/cnn_evaluation_metrics.json', 'r') as f:
    metrics = json.load(f)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(classification, cmap='RdYlGn')
axes[0].set_title('CNN Classification')
axes[0].axis('off')

im = axes[1].imshow(probability, cmap='RdYlGn_r', vmin=0, vmax=1)
axes[1].set_title('CNN Probability')
axes[1].axis('off')
plt.colorbar(im, ax=axes[1])

plt.suptitle(f"Accuracy: {metrics['accuracy']:.2%} | F1: {metrics['f1_score']:.2%}")
plt.tight_layout()
plt.savefig('results/cnn_visualization.png', dpi=300)
plt.show()

print(f"Metrics:")
for k, v in metrics.items():
    print(f"  {k:12s}: {v:.4f} ({v*100:.2f}%)")
```

---

## 🔄 Compare with Random Forest

### 1. Run Random Forest first (if not done yet)

```bash
cd src
python main.py
```

### 2. Compare results

```python
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Load both results
with rasterio.open('results/rasters/rf_classification.tif') as src:
    rf_map = src.read(1)

with rasterio.open('results/rasters/cnn_classification.tif') as src:
    cnn_map = src.read(1)

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(rf_map, cmap='RdYlGn')
axes[0].set_title('Random Forest')
axes[0].axis('off')

axes[1].imshow(cnn_map, cmap='RdYlGn')
axes[1].set_title('CNN')
axes[1].axis('off')

# Difference map
diff = cnn_map.astype(int) - rf_map.astype(int)
im = axes[2].imshow(diff, cmap='bwr', vmin=-1, vmax=1)
axes[2].set_title('Difference (CNN - RF)')
axes[2].axis('off')
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig('results/rf_vs_cnn_comparison.png', dpi=300)
plt.show()

# Statistics
agreement = (rf_map == cnn_map).sum() / rf_map.size
print(f"Agreement between RF and CNN: {agreement:.2%}")
print(f"RF detects deforestation: {(rf_map == 1).sum()} pixels")
print(f"CNN detects deforestation: {(cnn_map == 1).sum()} pixels")
```

---

## 🐛 Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Issue 2: "CUDA out of memory"

**Solution:**
```bash
# Reduce batch size
python main_dl.py --batch-size 16

# Or use CPU
python main_dl.py --device cpu
```

---

### Issue 3: Model not learning (loss stays constant)

**Possible causes:**
1. Data not normalized properly
2. Learning rate too high/low
3. Labels incorrect

**Debug:**
```bash
# Run test script first
python test_dl_modules.py

# Check if patches are normalized
python -c "
import numpy as np
data = np.load('results/data/cnn_training_patches.npz')
print('X_train mean:', data['X_train'].mean())
print('X_train std:', data['X_train'].std())
# Should be close to 0 and 1
"
```

---

### Issue 4: Training too slow

**Solutions:**
- ✅ Use GPU: `python main_dl.py --device cuda`
- ✅ Increase batch size: `python main_dl.py --batch-size 64`
- ✅ Reduce epochs if early stopping works well

---

## 📚 Next Steps

After successfully running the pipeline:

1. **Analyze results**
   - Compare metrics with Random Forest
   - Visualize classification maps
   - Identify areas of disagreement

2. **Fine-tune if needed**
   - Adjust hyperparameters in `src/common/config.py`
   - Try different patch sizes
   - Experiment with data augmentation

3. **Document findings**
   - Record best accuracy achieved
   - Note which areas are harder to classify
   - Compare smoothness with RF

4. **Write thesis chapter**
   - Methodology: Spatial-aware CNN approach
   - Results: Quantitative metrics + visualizations
   - Discussion: CNN vs RF trade-offs

---

## 📖 Additional Resources

- **Deep Learning Guide:** [DEEP_LEARNING_GUIDE.md](DEEP_LEARNING_GUIDE.md)
- **Module Documentation:** [src/deep_learning/README.md](src/deep_learning/README.md)
- **Implementation Summary:** [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Main README:** [README.md](README.md)

---

## ✅ Success Criteria

Your pipeline is working correctly if:

- ✅ All 8 steps complete without errors
- ✅ Test accuracy > 80%
- ✅ Classification map looks reasonable (not all 0s or all 1s)
- ✅ Probability map has varied values (0.0-1.0)
- ✅ Output files are created and readable
- ✅ Results are similar to or better than Random Forest

---

**Ready to start?**

```bash
cd src
python main_dl.py
```

**Good luck! 🚀**
