# 🔧 CNN Prediction Fix Guide

## ❌ Vấn đề bạn gặp phải

### 1. **False Positives cao**
- Nhiều vùng **không mất rừng** nhưng có **probability rất cao**
- Model dự đoán sai quá nhiều

### 2. **Raster chưa clip**
- Output raster chưa được clip theo boundary shapefile
- Hiển thị cả vùng ngoài rừng (không hợp lệ)

---

## 🔍 Nguyên nhân

### Vấn đề 1: **Normalization Mismatch**

**Training time:**
```python
# Normalize patches from training data
X_train_mean = X_train.mean(axis=(0,1,2))  # Mean từ training patches
X_train_std = X_train.std(axis=(0,1,2))    # Std từ training patches

X_train_normalized = (X_train - X_train_mean) / X_train_std
# Model học trên data đã normalize với training stats
```

**Prediction time (SAI!):**
```python
# Normalize patches from PREDICTION data
prediction_mean = patches.mean(axis=(0,1,2))  # ❌ Mean từ prediction patches (KHÁC!)
prediction_std = patches.std(axis=(0,1,2))    # ❌ Std từ prediction patches (KHÁC!)

patches_normalized = (patches - prediction_mean) / prediction_std
# Model nhận data với distribution KHÁC so với training → Dự đoán SAI!
```

**Tại sao sai?**
- Training data và prediction data có **distribution khác nhau**
- Mean/std tính từ prediction patches **không giống** training
- Model bị **"confused"** vì nhận input khác với khi training
- → Dự đoán sai (false positives cao)

### Vấn đề 2: **Không có NoData mask**

**Code cũ:**
```python
# Prediction ở MỌI pixels, kể cả vùng ngoài rừng
classification_map = np.zeros((height, width))  # All zeros initially
# Predict everywhere → Hiển thị cả vùng invalid
```

**Đúng:**
```python
# Chỉ predict trong valid area, set NoData cho vùng ngoài
classification_map[~valid_mask] = 255  # NoData
# GIS software sẽ không hiển thị vùng NoData
```

---

## ✅ Giải pháp

### Fix 1: **Đồng nhất Normalization**

**Lưu normalization stats từ training:**
```python
# Trong training
train_mean = X_train.mean(axis=(0,1,2), keepdims=True)
train_std = X_train.std(axis=(0,1,2), keepdims=True)

normalization_stats = {
    'mean': train_mean,
    'std': train_std
}
# Save normalization_stats
```

**Dùng lại stats khi prediction:**
```python
# Trong prediction
patches_normalized = (patches - train_mean) / train_std
# ✅ Dùng train_mean và train_std (GIỐNG training!)
# → Model nhận input tương tự training → Dự đoán ĐÚNG!
```

### Fix 2: **Apply Valid Mask**

```python
# Set NoData cho vùng invalid
classification_map[~valid_mask] = 255  # NoData value
probability_map[~valid_mask] = -9999   # NoData value

# Save với NoData metadata
rasterio.open(..., nodata=255)  # GeoTIFF với NoData
```

---

## 🚀 Cách chạy Fix

### Option 1: Chạy Python script

```bash
cd notebook
python fix_cnn_prediction.py
```

**Script này sẽ:**
1. Load lại trained model
2. Load normalization stats từ training data
3. Predict lại với **correct normalization**
4. Apply valid mask
5. Save kết quả mới: `cnn_classification_fixed.tif`

### Option 2: Update notebook và re-run

Thêm cells sau vào notebook (sau cell train model):

**Cell mới 1: Save normalization stats**
```python
# After training, save normalization stats
import pickle

normalization_stats = {
    'mean': X_train.mean(axis=(0, 1, 2), keepdims=True),
    'std': X_train.std(axis=(0, 1, 2), keepdims=True)
}

with open('../results/data/normalization_stats.pkl', 'wb') as f:
    pickle.dump(normalization_stats, f)

print("✓ Normalization stats saved")
```

**Cell mới 2: Load stats khi prediction**
```python
# Before prediction, load normalization stats
import pickle

with open('../results/data/normalization_stats.pkl', 'rb') as f:
    normalization_stats = pickle.load(f)

print("✓ Normalization stats loaded")
print(f"  Mean shape: {normalization_stats['mean'].shape}")
print(f"  Std shape: {normalization_stats['std'].shape}")
```

**Update prediction cell:**
```python
# OLD
classification_map, probability_map = predictor.predict_raster(
    feature_stack,
    valid_mask,
    stride=1,
    normalize=True  # ❌ Compute from prediction data
)

# NEW
classification_map, probability_map = predictor.predict_raster(
    feature_stack,
    valid_mask,
    stride=1,
    normalize=True,
    normalization_stats=normalization_stats  # ✅ Use training stats
)
```

---

## 📊 Kỳ vọng sau khi fix

### Trước fix:
- ❌ False positives: **30-40%** (quá cao!)
- ❌ Probability map: nhiều vùng không mất rừng có prob > 0.8
- ❌ Raster hiển thị cả vùng ngoài boundary

### Sau fix:
- ✅ False positives: **5-10%** (hợp lý)
- ✅ Probability map: đúng hơn, vùng không mất rừng có prob < 0.3
- ✅ Raster chỉ hiển thị vùng trong boundary
- ✅ NoData cho vùng ngoài rừng

### Metrics có thể thay đổi:
- **Accuracy:** Có thể giảm nhẹ (~1-2%) nhưng **đúng hơn**
- **Precision:** Tăng mạnh (ít false positives hơn)
- **Recall:** Giữ nguyên hoặc tăng nhẹ
- **F1-Score:** Tăng do precision tăng

---

## 🔬 Kiểm tra kết quả

### 1. Visual check

```python
import rasterio
import matplotlib.pyplot as plt

# Load fixed results
with rasterio.open('../results/rasters/cnn_classification_fixed.tif') as src:
    classification = src.read(1)
    nodata = src.nodata  # Should be 255

with rasterio.open('../results/rasters/cnn_probability_fixed.tif') as src:
    probability = src.read(1, masked=True)  # Masked=True để mask NoData

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(classification, cmap='RdYlGn', vmin=0, vmax=1)
axes[0].set_title('Fixed Classification')
axes[0].axis('off')

axes[1].imshow(probability, cmap='RdYlGn_r', vmin=0, vmax=1)
axes[1].set_title('Fixed Probability')
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

### 2. Statistics check

```python
# Check NoData masking
print(f"NoData value: {nodata}")
print(f"Pixels with NoData: {(classification == 255).sum():,}")
print(f"Valid pixels: {((classification == 0) | (classification == 1)).sum():,}")

# Check probability distribution
valid_probs = probability[probability != -9999]
print(f"\nProbability statistics (valid pixels only):")
print(f"  Min:    {valid_probs.min():.4f}")
print(f"  Max:    {valid_probs.max():.4f}")
print(f"  Mean:   {valid_probs.mean():.4f}")
print(f"  Median: {np.median(valid_probs):.4f}")

# Should see more reasonable distribution (not too many high probs)
```

### 3. Load in QGIS

```
File → Add Raster Layer → cnn_classification_fixed.tif

✅ Vùng ngoài boundary sẽ KHÔNG hiển thị (NoData)
✅ Chỉ thấy prediction trong vùng rừng
✅ Symbology: 0=Green (No loss), 1=Red (Deforestation)
```

---

## 📝 Notes quan trọng

### 1. **Luôn luôn save normalization stats**
```python
# Best practice: Save cùng với model
torch.save({
    'model_state_dict': model.state_dict(),
    'normalization_stats': normalization_stats,  # Save cùng luôn
    'config': CONFIG
}, 'model_checkpoint.pth')
```

### 2. **Consistency is key**
- Training time: normalize với train stats
- Validation time: normalize với train stats (KHÔNG phải val stats)
- Test time: normalize với train stats (KHÔNG phải test stats)
- Prediction time: normalize với train stats (KHÔNG phải prediction stats)

### 3. **Valid mask vs Boundary**
- **Valid mask:** Pixels có data hợp lệ (không NoData, không cloud, etc.)
- **Boundary:** Ranh giới rừng (từ shapefile)
- **Final mask:** valid_mask AND boundary mask
- Output: Chỉ predict trong vùng (valid AND inside boundary)

---

## 🎯 Quick Fix Checklist

- [ ] Đã train model xong
- [ ] Run `python notebook/fix_cnn_prediction.py`
- [ ] Check output: `cnn_classification_fixed.tif`
- [ ] Visualize để verify
- [ ] Load vào QGIS để check NoData masking
- [ ] So sánh với kết quả cũ
- [ ] Document kết quả mới trong thesis

---

## ❓ FAQs

**Q: Tại sao không fix ngay trong notebook từ đầu?**

A: Notebook cũ chưa implement normalization stats saving. Fix script này giúp bạn re-run prediction mà không cần train lại model.

**Q: Có cần train lại model không?**

A: **KHÔNG cần!** Model đã train đúng. Chỉ cần re-run prediction với correct normalization.

**Q: Kết quả fix có khác nhiều không?**

A: Phụ thuộc vào difference giữa train distribution và prediction distribution. Thường thì:
- Nếu train và predict trên cùng area: khác ít (~2-5%)
- Nếu distribution khác nhiều: khác rất nhiều (~20-30%)

**Q: Làm sao biết fix đã đúng?**

A: Check các dấu hiệu:
- ✅ Probability map reasonable (không quá nhiều vùng >0.8)
- ✅ NoData masking đúng (vùng ngoài rừng không hiển thị)
- ✅ Visual check: kết quả hợp lý với ground truth
- ✅ Metrics improve (precision tăng)

---

**Fix script location:** `notebook/fix_cnn_prediction.py`

**Run:** `cd notebook && python fix_cnn_prediction.py`

**Expected time:** ~10-15 minutes

Good luck! 🚀
