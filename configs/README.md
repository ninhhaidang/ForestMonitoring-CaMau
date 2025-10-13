# Configs Directory

## Config files cho Open-CD framework

### phase1_snunet_s2only.py
- Model: SNUNet-CD (c16)
- Input: 14 channels (Sentinel-2)
- Training: 40k iterations
- Batch size: 8
- Learning rate: 0.01

### phase2_snunet_s2s1.py  
- Model: SNUNet-CD (c16)
- Input: 18 channels (Sentinel-2 + Sentinel-1)
- Training: 40k iterations
- Batch size: 8
- Learning rate: 0.01

## 🔧 Sử dụng

```bash
# Training
python open-cd/tools/train.py configs/phase1_snunet_s2only.py

# Testing
python open-cd/tools/test.py configs/phase1_snunet_s2only.py \
    experiments/phase1_s2only/checkpoints/best_model.pth
```
