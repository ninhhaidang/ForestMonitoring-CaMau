# 🚀 Quick Reference: 3 Deep Learning Models

## TL;DR - Chọn Model Nào?

```
🏃 Cần NHANH?          → Spatial Context CNN
🎯 Cần CÂN BẰNG?       → Multi-Scale CNN (KHUYẾN NGHỊ)
🎨 Cần CHẤT LƯỢNG CAO? → Shallow U-Net
🌿 Cần ALIGN VỚI NDVI? → Multi-Scale CNN (NDVI-Weighted)
```

---

## 📊 Bảng So Sánh Nhanh

| Model | Params | Speed | Accuracy | Receptive Field | GPU RAM | Use Case |
|-------|--------|-------|----------|----------------|---------|----------|
| **Spatial CNN** | 13K | ⚡⚡⚡ | ⭐⭐ | 50m | 50MB | Edge devices, baseline |
| **Multi-Scale** | 90K | ⚡⚡ | ⭐⭐⭐ | 90m | 150MB | **PRODUCTION** |
| **U-Net** | 476K | ⚡ | ⭐⭐⭐⭐ | 610m | 400MB | Research, best quality |
| **MultiScale+NDVI** | 100K | ⚡⚡ | ⭐⭐⭐? | 90m | 160MB | NDVI-focused detection |

---

## 🏗️ Kiến Trúc 1 Câu

### 1. Spatial Context CNN
```
Input → Conv3×3 → Conv3×3 → Conv1×1 → Output
```
**Giống như:** Random Forest + spatial smoothing

### 2. Multi-Scale CNN
```
Input ──┬→ Conv3×3 ──┐
        └→ Conv5×5 ──┴→ Concat → More layers → Output
```
**Giống như:** Nhìn cả chi tiết (3×3) VÀ bối cảnh (5×5)

### 3. Shallow U-Net
```
Input → Encoder ↓ → Bottleneck → Decoder ↑ → Output
         │                            ↑
         └──────── Skip Connect ──────┘
```
**Giống như:** Zoom out để nhìn tổng thể, zoom in lại với details

### 4. Multi-Scale + NDVI
```
Input → Channel Attention (emphasize NDVI)
     → Multi-Scale branches
     → NDVI Difference Branch (explicit NDVI change)
     → Fusion → Output
```
**Giống như:** Multi-Scale + "chú ý đặc biệt vào NDVI change"

---

## ⚡ Performance Numbers

### Inference Speed (Full image ~137M pixels)
```
Spatial CNN:      16s  ████░░░░░░
Multi-Scale:      40s  ██████████
U-Net:           115s  ████████████████████████████
```

### Memory Usage (Batch=64)
```
Spatial CNN:     200MB  ████░░░░░░
Multi-Scale:     400MB  ████████░░
U-Net:          1000MB  ████████████████████
```

---

## 🎓 Khi Nào Dùng Gì?

### Scenario 1: Đồ án/Thesis
```
✅ Train tất cả 4 models
✅ So sánh metrics
✅ Chọn best model dựa trên:
   - Accuracy
   - Smoothness (ít noise)
   - Speed (nếu cần deploy)

🎯 Khuyến nghị: U-Net cho best quality
```

### Scenario 2: Production Deployment
```
✅ Multi-Scale CNN
   - Balance tốt nhất
   - Đủ nhanh cho real-time (~40s/image)
   - Accuracy cao (89-90%)

❌ Không dùng U-Net (quá chậm)
❌ Không dùng Spatial CNN (accuracy thấp)
```

### Scenario 3: Mobile/Edge Devices
```
✅ Spatial CNN (only choice)
   - 13K params → 50KB model size
   - Chạy được trên Raspberry Pi
   - 16s for full image

❌ Các models khác quá nặng
```

### Scenario 4: Research/Analysis
```
✅ U-Net
   - Best quality maps
   - Smooth, connected regions
   - Dễ interpret cho analysts

✅ Multi-Scale + NDVI
   - Nếu NDVI change là key indicator
   - Muốn align với physical process
```

---

## 🧮 Chi Phí Training

### Time (epochs=100, batch=64, RTX A4000)
```
Spatial CNN:    ~15 minutes
Multi-Scale:    ~25 minutes
U-Net:          ~45 minutes
MultiScale+NDVI: ~30 minutes
```

### Storage
```
Spatial CNN:    50KB  (.pth file)
Multi-Scale:   350KB
U-Net:         1.9MB
MultiScale+NDVI: 400KB
```

---

## 💡 Pro Tips

### Tip 1: Start Simple
```
1. Train Spatial CNN first (15 min)
2. Nếu accuracy OK → Done!
3. Nếu không → Try Multi-Scale
4. Nếu vẫn không OK → U-Net
```

### Tip 2: Ensemble
```
# Combine 3 models
final_pred = (spatial_pred + multiscale_pred + unet_pred) / 3

# Often better than single model!
# Tốn thời gian nhưng accuracy cao hơn
```

### Tip 3: Check NDVI First
```
# Trước khi train CNN, check xem:
ndvi_change = NDVI_2025 - NDVI_2024

# Nếu NDVI change correlates tốt với deforestation
# → Dùng NDVI-weighted model!
```

---

## 📚 Chi Tiết Đầy Đủ

Xem [MODELS_ARCHITECTURE.md](MODELS_ARCHITECTURE.md) để hiểu chi tiết:
- Cách mỗi layer hoạt động
- Receptive field calculations
- Example code
- Memory breakdown
- Training tips

---

## ❓ FAQ

**Q: Model nào best?**
A: Không có "best" universal. U-Net best quality, Multi-Scale best balance.

**Q: Tôi chỉ có 500 training samples, dùng gì?**
A: Spatial CNN hoặc Multi-Scale. U-Net có thể overfit.

**Q: Làm sao giảm overfitting?**
A:
- Thêm augmentation
- Giảm model size (dùng Spatial CNN)
- Thêm dropout
- Early stopping

**Q: Model nào dễ interpret nhất?**
A: Spatial CNN (gần với linear model). U-Net khó interpret nhất.

**Q: Tôi muốn deploy lên web, dùng gì?**
A: Multi-Scale CNN. Convert sang ONNX cho fast inference.

---

## 🔗 Links

- [Full Architecture Explanation](MODELS_ARCHITECTURE.md)
- [Training Notebooks](../notebooks/)
- [Model Source Code](../src/models.py)
- [NDVI-Weighted Models](../src/models_ndvi_weighted.py)
