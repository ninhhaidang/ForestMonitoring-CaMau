# 📊 Live Monitoring Guide - Enhanced Training Visualization

## 🎯 New Features

Tôi đã thêm **live monitoring với tqdm** và **real-time visualization** vào notebooks!

### ✨ Features mới:

1. **📊 Real-time Progress Bars** (tqdm)
   - Progress bar cho mỗi training epoch
   - Progress bar cho validation
   - Overall progress bar cho toàn bộ training
   - Hiển thị metrics real-time (loss, accuracy)

2. **📈 Live Plotting**
   - Plots tự động update sau mỗi epoch
   - 4 plots: Loss, Accuracy, F1 Score, Learning Rate
   - Đánh dấu best epoch trên chart
   - Auto-save plot cuối training

3. **⏱️ Time Estimation**
   - Ước tính thời gian training
   - Dựa trên GPU type (RTX 3090, 4090, A100, etc.)
   - Hiển thị trước khi bắt đầu train

4. **🎨 Better Visualization**
   - Visualize samples với progress bar
   - Formatted epoch summaries
   - Color-coded status messages

---

## 🚀 Usage

### 1. Training với Live Monitoring:

```python
from src.notebook_utils import NotebookTrainer

# Create trainer (thay vì Trainer thông thường)
trainer = NotebookTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    save_dir='models',
    model_name='unet_efficientnet'
)

# Train với live plots
history = trainer.train(
    epochs=50,
    early_stopping_patience=10,
    plot_every=1  # Update plot every 1 epoch
)
```

### 2. Visualize Samples:

```python
from src.notebook_utils import visualize_batch_with_progress

# Visualize 8 random samples với progress bar
visualize_batch_with_progress(train_loader.dataset, num_samples=8)
```

### 3. Training Schedule Estimate:

```python
from src.notebook_utils import print_training_schedule

print_training_schedule(
    epochs=50,
    batch_size=16,
    total_samples=1028,
    gpu_name='NVIDIA RTX A4000'
)
```

---

## 📊 What You'll See During Training

### Progress Bars:

```
Overall Progress: 20%|████      | 10/50 epochs
Epoch 10 [Train]: 100%|██████████| 65/65 [00:45<00:00]
  └─ loss: 0.3214  acc: 87.32%

Epoch 10 [Val]  : 100%|██████████| 8/8 [00:03<00:00]

================================================================================
📊 Epoch 10 Summary
================================================================================
  Train → Loss: 0.3214  |  Acc: 87.32%
  Val   → Loss: 0.2891  |  Acc: 89.15%  |  F1: 0.8876
  LR: 0.000095
================================================================================
✅ New best model! Val Acc: 89.15% (saved)
```

### Live Plots:

Bạn sẽ thấy 4 plots tự động update:

1. **Loss Curve** - Train vs Val loss
2. **Accuracy Curve** - Train vs Val accuracy (với dấu sao ⭐ tại best epoch)
3. **F1 Score Curve** - Validation F1 score
4. **Learning Rate Schedule** - LR changes over time

---

## 🎨 Visual Examples

### Training Progress:

```
🚀 TRAINING STARTED - UNET_EFFICIENTNET
================================================================================
Device: cuda
Total Epochs: 50
Early Stopping Patience: 10
Save Directory: models/unet_efficientnet
================================================================================

Overall Progress:   0%|          | 0/50 epochs [00:00<?, ?it/s]
Epoch 1 [Train]: 100%|██████████| 65/65 [00:47<00:00, 1.38it/s, loss=0.6234, acc=65.43%]
Epoch 1 [Val]  : 100%|██████████| 8/8 [00:03<00:00, 2.15it/s]

================================================================================
📊 Epoch 1 Summary
================================================================================
  Train → Loss: 0.6234  |  Acc: 65.43%
  Val   → Loss: 0.5891  |  Acc: 68.21%  |  F1: 0.6543
  LR: 0.000100
================================================================================
✅ New best model! Val Acc: 68.21% (saved)
```

### Time Estimation:

```
================================================================================
📅 TRAINING SCHEDULE ESTIMATE
================================================================================
Total epochs: 50
Batches per epoch: 65
Total batches: 3,250

Estimated time:
  Per epoch: ~13.0 minutes
  Total: ~10.8 hours (650 minutes)

GPU: NVIDIA RTX A4000
================================================================================
```

---

## 🔧 Customization

### Adjust Plot Update Frequency:

```python
# Update plot every 2 epochs (faster for long training)
history = trainer.train(
    epochs=100,
    early_stopping_patience=15,
    plot_every=2
)
```

### Change Visualization Settings:

```python
# Visualize more samples
visualize_batch_with_progress(dataset, num_samples=16)

# Or fewer samples
visualize_batch_with_progress(dataset, num_samples=4)
```

---

## 📋 Comparison: Old vs New

### Old Trainer (src/utils.py):
```python
from src.utils import Trainer

trainer = Trainer(...)
history = trainer.train(epochs=50)
# → Chỉ có text output
# → Không có live plots
# → Phải plot manually sau khi train xong
```

### New NotebookTrainer (src/notebook_utils.py):
```python
from src.notebook_utils import NotebookTrainer

trainer = NotebookTrainer(...)
history = trainer.train(epochs=50)
# ✅ Progress bars cho mọi operations
# ✅ Live plots update mỗi epoch
# ✅ Auto-save plots
# ✅ Better formatted output
# ✅ Time estimates
```

---

## 💡 Tips

### 1. For Long Training Sessions:

```python
# Update plots less frequently to save time
trainer.train(epochs=100, plot_every=5)
```

### 2. Monitor GPU Usage:

```python
# Check GPU before training
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 3. Adjust for Slower GPUs:

```python
# Reduce batch size if OOM
BATCH_SIZE = 8  # Instead of 16

# Or train for fewer epochs first
EPOCHS = 20  # Quick test run
```

---

## 🎯 Benefits

### Real-time Monitoring:
- ✅ Xem ngay khi model đang học tốt hay không
- ✅ Phát hiện overfitting sớm
- ✅ Dừng training nếu không improve

### Better UX:
- ✅ Progress bars cho mọi operations
- ✅ Không phải đợi đến cuối mới biết kết quả
- ✅ Dễ debug và adjust hyperparameters

### Time Saving:
- ✅ Biết trước training mất bao lâu
- ✅ Auto-save best model
- ✅ Early stopping tự động

---

## 📚 Files Changed

### New Files:
- `src/notebook_utils.py` - NotebookTrainer + visualization utilities

### Updated Files:
- `notebooks/1_train_models.ipynb` - Sử dụng NotebookTrainer
  - Cell 3: Import NotebookTrainer
  - Cell 9: Add training schedule
  - Cell 11: Enhanced sample visualization
  - Cell 15: Use NotebookTrainer with live plots

---

## 🔍 Example Output

Khi bạn chạy notebook, bạn sẽ thấy:

```
🔄 Creating dataloaders...

📊 Data splits:
  Train: 1028 samples (80.0%)
  Val:   128 samples (10.0%)
  Test:  129 samples (10.0%)

✅ DataLoaders created!
  Train: 1028 samples (65 batches)
  Val:   128 samples (8 batches)
  Test:  129 samples (9 batches)

================================================================================
📅 TRAINING SCHEDULE ESTIMATE
================================================================================
Total epochs: 50
Batches per epoch: 65
Total batches: 3,250

Estimated time:
  Per epoch: ~13.0 minutes
  Total: ~10.8 hours (650 minutes)

GPU: NVIDIA RTX A4000
================================================================================

Loading 8 random samples...
Loading samples: 100%|██████████| 8/8 [00:02<00:00, 3.21it/s]
✅ Displayed 8 samples

[... 8 images displayed ...]

🎬 Training will start with LIVE visualization!
   - Progress bars for each epoch
   - Real-time plots (updated every epoch)
   - Automatic best model saving
   - Early stopping monitoring

[... Live training with progress bars and plots ...]
```

---

## 🎉 Summary

Bây giờ bạn có **real-time monitoring** hoàn chỉnh cho training process:

- 📊 **tqdm progress bars** - Track từng batch, epoch
- 📈 **Live plots** - Xem metrics update real-time
- ⏱️ **Time estimates** - Biết trước training mất bao lâu
- 🎨 **Better visualization** - Đẹp và dễ hiểu hơn

Chạy notebook và xem magic xảy ra! ✨
