# Training Guide

## Quick Start

### 1. Activate Environment
```bash
conda activate dang
```

### 2. Phase 1: Train with Sentinel-2 Only

#### Step 1: Prepare data
```bash
jupyter notebook notebooks/02_preprocessing/2.1_prepare_phase1.ipynb
```

#### Step 2: Train
```bash
python open-cd/tools/train.py configs/phase1_snunet_s2only.py
```

#### Step 3: Evaluate
```bash
python open-cd/tools/test.py configs/phase1_snunet_s2only.py \
    experiments/phase1_s2only/checkpoints/best_model.pth
```

### 3. Phase 2: Train with S2 + S1

#### Step 1: Prepare data
```bash
jupyter notebook notebooks/02_preprocessing/2.2_prepare_phase2.ipynb
```

#### Step 2: Train
```bash
python open-cd/tools/train.py configs/phase2_snunet_s2s1.py
```

#### Step 3: Evaluate
```bash
python open-cd/tools/test.py configs/phase2_snunet_s2s1.py \
    experiments/phase2_s2s1/checkpoints/best_model.pth
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir experiments/phase1_s2only/logs
```

### Check GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```
