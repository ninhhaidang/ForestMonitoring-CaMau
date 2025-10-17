# Training Guide for Ca Mau Forest Change Detection

**Author:** Ninh Hai Dang (21021411)
**Date:** 2025-10-17

## Overview

Đã tạo xong 3 config files cho training:
- `ban_camau.py` - BAN model (ViT-B/16 + MiT-B0)
- `tinycdv2_camau.py` - TinyCDv2 model (EfficientNet-B4)
- `changer_camau.py` - Changer model (MiT-B0 with interaction modules)

## Configuration Details

### Common Settings
- **Input channels**: 9 channels per time step (18 total)
  - Sentinel-2: B4, B8, B11, B12, NDVI, NBR, NDMI
  - Sentinel-1: VH, VV/VH ratio
- **Image size**: 256×256 pixels
- **Data format**: `.tif` for images, `.png` for labels
- **Training epochs**: ~100 epochs
- **Validation interval**: Every ~10 epochs

### Model-Specific Settings

| Model | Batch Size | Max Iterations | Learning Rate | Expected GPU Memory |
|-------|-----------|---------------|---------------|-------------------|
| BAN | 4 | 25,000 | 0.0001 | ~8-10 GB |
| TinyCDv2 | 8 | 12,800 | 0.00357 | ~4-6 GB |
| Changer | 6 | 17,100 | 0.0001 | ~6-8 GB |

## Prerequisites

1. **Open-CD installed**:
   ```bash
   cd open-cd
   pip install -v -e .
   ```

2. **Dataset ready** in `data/processed/`:
   ```
   data/processed/
   ├── train/ (1,028 samples)
   ├── val/ (128 samples)
   └── test/ (129 samples)
   ```

3. **Download pretrained weights** (optional but recommended):
   - BAN needs CLIP ViT-B/16: Download to `pretrain/clip_vit-base-patch16-224_3rdparty-d08f8887.pth`
   - Changer uses online pretrained MiT-B0 (auto-download)
   - TinyCDv2 uses ImageNet pretrained EfficientNet-B4 (auto-download)

## Training Commands

### 1. Test Training (Single GPU)

**Start with TinyCDv2** (lightest model, fastest training):

```bash
python open-cd/tools/train.py configs/tinycdv2_camau.py
```

Expected output:
- Logs saved to `experiments/tinycdv2/`
- Checkpoints every 1,280 iterations (~10 epochs)
- Best model saved as `best_mIoU_*.pth`

### 2. Train BAN

```bash
python open-cd/tools/train.py configs/ban_camau.py
```

**Note**: BAN requires more GPU memory. If OOM (Out of Memory):
- Reduce `batch_size` in config from 4 to 2
- Adjust `max_iters` accordingly

### 3. Train Changer

```bash
python open-cd/tools/train.py configs/changer_camau.py
```

### 4. Resume Training (if interrupted)

```bash
python open-cd/tools/train.py configs/tinycdv2_camau.py \
    --resume experiments/tinycdv2/iter_*.pth
```

## Multi-GPU Training (Optional)

If you have multiple GPUs:

```bash
# 2 GPUs
bash open-cd/tools/dist_train.sh configs/tinycdv2_camau.py 2

# 4 GPUs
bash open-cd/tools/dist_train.sh configs/ban_camau.py 4
```

**Note**: Adjust batch size accordingly when using multiple GPUs.

## Monitoring Training

### TensorBoard

Open-CD automatically logs to TensorBoard:

```bash
tensorboard --logdir experiments/
```

Then open http://localhost:6006 in browser.

### Watch Logs

```bash
# TinyCDv2
tail -f experiments/tinycdv2/*/vis_data/*.log

# BAN
tail -f experiments/ban/*/vis_data/*.log

# Changer
tail -f experiments/changer/*/vis_data/*.log
```

## Evaluation

After training, evaluate on test set:

```bash
# TinyCDv2
python open-cd/tools/test.py \
    configs/tinycdv2_camau.py \
    experiments/tinycdv2/best_mIoU_*.pth

# BAN
python open-cd/tools/test.py \
    configs/ban_camau.py \
    experiments/ban/best_mIoU_*.pth

# Changer
python open-cd/tools/test.py \
    configs/changer_camau.py \
    experiments/changer/best_mIoU_*.pth
```

## Troubleshooting

### Issue 1: `KeyError: 'MultiImgLoadImageFromFile'`

**Solution**: Import Open-CD modules properly:
```python
import opencd.datasets
import opencd.datasets.transforms
```

### Issue 2: Out of Memory (OOM)

**Solutions**:
1. Reduce batch size in config file
2. Use smaller model (TinyCDv2 > Changer > BAN)
3. Use mixed precision training (already enabled in BAN config)

### Issue 3: `FileNotFoundError` for pretrained weights

**Solution**:
- For BAN: Download CLIP weights manually and place in `pretrain/` folder
- For others: Weights will auto-download (requires internet connection)

### Issue 4: Data loading errors

**Verify dataset**:
```bash
python -c "from pathlib import Path; print(list(Path('data/processed/train/A').glob('*.tif'))[:5])"
```

Should show 5 `.tif` files.

## Expected Training Time

On a single RTX 3080 (10GB):
- **TinyCDv2**: ~2-3 hours for 100 epochs
- **Changer**: ~3-4 hours for 100 epochs
- **BAN**: ~4-6 hours for 100 epochs

## Next Steps

1. **Train all 3 models**
2. **Compare results** on validation set
3. **Ensemble predictions** (Phase 4)
4. **Final evaluation** on test set
5. **Write thesis chapter** on results

## Questions?

If you encounter issues:
1. Check logs in `experiments/{model}/*/vis_data/*.log`
2. Verify GPU memory with `nvidia-smi`
3. Ensure dataset integrity with verification script
4. Check Open-CD installation: `python -c "import opencd; print(opencd.__version__)"`

Good luck with training! 🚀
