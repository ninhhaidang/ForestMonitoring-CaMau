# 🎯 Training All 3 Models - Complete Guide

## 📋 Overview

Bạn giờ có **4 notebooks riêng** để train và compare 3 models:

1. **`1a_train_unet_mobilenet.ipynb`** - UNet-MobileNetV2 (Fastest)
2. **`1b_train_unet_efficientnet.ipynb`** - UNet-EfficientNet-B0 (Balanced) ⭐
3. **`1c_train_fpn_efficientnet.ipynb`** - FPN-EfficientNet-B0 (Most Accurate)
4. **`1d_compare_all_models.ipynb`** - Compare all 3 models

---

## 🎯 Model Specifications

### Optimized for 12GB VRAM Usage

| Model | Batch Size | VRAM Usage | Parameters | Speed | Accuracy |
|-------|-----------|------------|------------|-------|----------|
| **UNet-MobileNetV2** | 64 | ~3GB | ~2M | ⚡⚡⚡⚡ | ⭐⭐⭐ |
| **UNet-EfficientNet** ⭐ | 48 | ~10-12GB | ~5M | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| **FPN-EfficientNet** | 32 | ~12GB | ~6M | ⚡⚡ | ⭐⭐⭐⭐⭐ |

⭐ = Recommended (best balance)

---

## 🚀 Training Workflow

### Option 1: Train All 3 Models Sequentially

**Recommended approach:**

```bash
jupyter notebook
```

Then run notebooks in order:
1. `1a_train_unet_mobilenet.ipynb` → ~8-10 giờ
2. `1b_train_unet_efficientnet.ipynb` → ~10-12 giờ
3. `1c_train_fpn_efficientnet.ipynb` → ~12-14 giờ
4. `1d_compare_all_models.ipynb` → ~5 phút

**Total time:** ~30-36 giờ (có thể để chạy xuyên đêm)

### Option 2: Train Parallel (Multiple GPUs)

Nếu có nhiều GPUs, mở 3 Jupyter sessions riêng:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 jupyter notebook 1a_train_unet_mobilenet.ipynb

# Terminal 2
CUDA_VISIBLE_DEVICES=1 jupyter notebook 1b_train_unet_efficientnet.ipynb

# Terminal 3 (nếu có GPU thứ 3)
CUDA_VISIBLE_DEVICES=2 jupyter notebook 1c_train_fpn_efficientnet.ipynb
```

### Option 3: Train Only Best Model

Nếu chỉ muốn train 1 model tốt nhất:

```bash
# Recommended
jupyter notebook notebooks/1b_train_unet_efficientnet.ipynb
```

---

## 📊 Batch Size Optimization

### Tại sao batch size khác nhau?

```python
# UNet-MobileNetV2 (lightweight)
BATCH_SIZE = 64  # Nhẹ → batch size lớn → train nhanh hơn

# UNet-EfficientNet (medium)
BATCH_SIZE = 48  # Vừa → batch size vừa → sử dụng ~10-12GB VRAM

# FPN-EfficientNet (heavy)
BATCH_SIZE = 32  # Nặng → batch size nhỏ → tránh OOM, sử dụng ~12GB VRAM
```

### Nếu GPU của bạn khác 16GB:

**8GB VRAM:**
```python
# Giảm batch size:
BATCH_SIZE = 32  # UNet-MobileNetV2
BATCH_SIZE = 16  # UNet-EfficientNet
BATCH_SIZE = 8   # FPN-EfficientNet
```

**24GB+ VRAM:**
```python
# Tăng batch size để train nhanh hơn:
BATCH_SIZE = 128  # UNet-MobileNetV2
BATCH_SIZE = 64   # UNet-EfficientNet
BATCH_SIZE = 48   # FPN-EfficientNet
```

---

## 📁 Output Structure

Sau khi train xong, bạn sẽ có:

```
models/
├── unet_mobilenet/
│   ├── best_model.pth              # Best checkpoint
│   ├── checkpoint_epoch_10.pth     # Checkpoint @ epoch 10
│   ├── checkpoint_epoch_20.pth     # Checkpoint @ epoch 20
│   └── training_history.png        # Training curves
│
├── unet_efficientnet/
│   ├── best_model.pth
│   ├── checkpoint_epoch_10.pth
│   └── training_history.png
│
└── fpn_efficientnet/
    ├── best_model.pth
    ├── checkpoint_epoch_10.pth
    └── training_history.png

results/
├── model_comparison.csv           # Comparison table
└── model_comparison.png           # Comparison charts
```

---

## 📈 Expected Results

### Performance Range (on test set):

| Model | Accuracy | F1 Score | Training Time |
|-------|----------|----------|---------------|
| UNet-MobileNetV2 | 83-87% | 0.82-0.86 | ~8-10 giờ |
| UNet-EfficientNet | 87-91% | 0.86-0.90 | ~10-12 giờ |
| FPN-EfficientNet | 89-93% | 0.88-0.92 | ~12-14 giờ |

**Note:** Actual results depend on data quality và random seed.

---

## 🎨 Live Monitoring Features

Mỗi notebook có:

### 1. Progress Bars
```
Overall Progress: 20%|████      | 10/50 epochs
Epoch 10 [Train]: 100%|██████████| 22/22 [00:30<00:00]
  └─ loss: 0.3214  acc: 87.32%
Epoch 10 [Val]  : 100%|██████████| 3/3 [00:02<00:00]
```

### 2. Live Plots
- Loss curve (train vs val)
- Accuracy curve (train vs val)
- F1 Score curve
- Learning Rate schedule

### 3. Auto-save
- Best model checkpoint
- Training history plots
- Epoch-wise checkpoints (every 10 epochs)

---

## 🔧 Customization

### Training Config

Trong mỗi notebook cell 2:

```python
# Adjust these:
BATCH_SIZE = 48       # Change based on your GPU
EPOCHS = 50           # Increase/decrease
LEARNING_RATE = 1e-4  # Try 1e-3 for faster convergence
EARLY_STOPPING_PATIENCE = 10  # Patience for early stopping
```

### Quick Test Run

Để test nhanh (5-10 phút):

```python
EPOCHS = 5
EARLY_STOPPING_PATIENCE = 3
```

### Production Training

Để train kỹ hơn (24+ giờ):

```python
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
```

---

## 💡 Tips

### 1. Monitor GPU Usage

```python
# Trong cell đầu tiên của notebook
import torch
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Sau mỗi epoch, check:
print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### 2. Save Intermediate Results

Models tự động save mỗi 10 epochs:
- `checkpoint_epoch_10.pth`
- `checkpoint_epoch_20.pth`
- etc.

Nếu training bị interrupt, có thể resume từ checkpoint cuối.

### 3. Compare Before Choosing

Chạy notebook `1d_compare_all_models.ipynb` sau khi train xong để xem model nào tốt nhất cho data của bạn.

### 4. Use Best Model for Inference

Sau khi compare, dùng model tốt nhất trong notebook 2 (inference):

```python
# Trong notebook 2_inference_wholescene.ipynb
MODEL_NAME = 'unet_efficientnet'  # Hoặc model nào tốt nhất
```

---

## 🎯 Training Schedule

### Kế hoạch training 3 models:

**Day 1 (Evening):**
- 6 PM: Start `1a_train_unet_mobilenet.ipynb`
- 2 AM next day: Finish

**Day 2 (Evening):**
- 6 PM: Start `1b_train_unet_efficientnet.ipynb`
- 4 AM next day: Finish

**Day 3 (Evening):**
- 6 PM: Start `1c_train_fpn_efficientnet.ipynb`
- 6 AM next day: Finish

**Day 4:**
- Run `1d_compare_all_models.ipynb` (5 phút)
- Choose best model
- Run inference với best model

---

## 🔍 Troubleshooting

### OOM (Out of Memory)

```python
# Giảm batch size trong cell 2
BATCH_SIZE = 16  # Thay vì 48
```

### Training Too Slow

```python
# Tăng batch size (nếu GPU cho phép)
BATCH_SIZE = 64

# Hoặc giảm epochs
EPOCHS = 30
```

### Model Not Improving

- Check learning rate (có thể quá cao hoặc quá thấp)
- Check data quality
- Try different optimizer:
  ```python
  optimizer = get_optimizer(model, 'adam', lr=1e-3)  # Thay vì adamw
  ```

---

## ✅ Checklist

Before starting training:

- [ ] GPU có đủ VRAM (12GB+ recommended)
- [ ] Data đã đặt trong `data/raw/`
- [ ] CSV file có đủ 1,285 points
- [ ] 4 ảnh TIFF tồn tại và readable
- [ ] Đã cài `segmentation-models-pytorch`
- [ ] Có đủ disk space (~5GB cho models + logs)

During training:

- [ ] Monitor GPU usage
- [ ] Check live plots mỗi 5-10 epochs
- [ ] Note down best validation accuracy
- [ ] Save training logs/screenshots

After training:

- [ ] Run comparison notebook
- [ ] Compare all 3 models
- [ ] Choose best model
- [ ] Use best model for inference

---

## 🎉 Summary

Bạn giờ có **complete pipeline** để train và compare 3 models:

✅ **3 training notebooks** - Mỗi model có notebook riêng với batch size tối ưu
✅ **1 comparison notebook** - So sánh kết quả tự động
✅ **12GB VRAM optimized** - Sử dụng đủ GPU memory
✅ **Live monitoring** - Progress bars + real-time plots
✅ **Auto-save** - Không mất công training nếu interrupt

Chúc bạn training thành công! 🚀
