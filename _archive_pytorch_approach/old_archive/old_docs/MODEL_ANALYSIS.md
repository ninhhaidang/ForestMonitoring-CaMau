# PHÂN TÍCH 3 MODELS: BAN, CHANGER, SNUNET-CD

**Tác giả:** Ninh Hải Đăng (21021411)
**Ngày:** 2025-10-17

## TÓM TẮT QUYẾT ĐỊNH

**✅ 3 models được chọn: BAN, Changer, SNUNet-CD**

**Lý do chính:**
- ✅ **Tất cả đều hỗ trợ 9 channels natively** (yêu cầu bắt buộc)
- ✅ Đa dạng về kiến trúc: Heavy Transformer / Medium Transformer / Lightweight CNN
- ✅ Đa dạng về design: Asymmetric / Symmetric + Interaction / Dense Connections
- ✅ Phù hợp GPU 16GB: Batch sizes 4/6/8 đều fit trong VRAM
- ✅ State-of-the-art: 2022-2024 (mới nhất)

---

## PHÂN TÍCH CHI TIẾT

### 1. BAN (Bi-temporal Adapter Network)

**📄 Paper:** Chen et al., IEEE TGRS 2024

**🏗️ Architecture:**
- **Main Encoder:** CLIP ViT-B/16 (86M params)
  - Pretrained on 400M image-text pairs
  - Patch size: 16×16
  - Embedding: 768 dimensions
- **Side Encoder:** MiT-B0 (3.3M params)
  - Pretrained on ImageNet
  - Hierarchical multi-scale features
- **Fusion:** Adapter-based bi-temporal fusion
- **Total Params:** ~90M

**🔧 Technical Details:**
- Input: 256×256 (resized to 224×224 for ViT)
- Batch size: 4
- Learning rate: 1e-4
- Training time: ~100 epochs × 257 iters = 25,700 iterations
- **9-channel support:** ✅ Vision Transformer uses Conv2d patch embedding with `in_channels` parameter

**⚡ Performance:**
- Speed: ~3-4s per 256×256 tile
- Memory: ~8-10GB VRAM
- Expected F1: 0.89-0.91

**👍 Ưu điểm:**
- Strong semantic understanding từ CLIP
- Asymmetric design: CLIP cho high-level, MiT cho low-level
- State-of-the-art architecture (2024)

**👎 Nhược điểm:**
- Heavy model (~90M params)
- Slow inference
- Cần pretrained CLIP weights

---

### 2. Changer (Feature Interaction Network)

**📄 Paper:** Fang et al., IEEE TGRS 2023

**🏗️ Architecture:**
- **Backbone:** IA_MixVisionTransformer (MiT-B0)
  - Interaction-aware design
  - Spatial Exchange + Channel Exchange modules
  - Hierarchical features: [32, 64, 160, 256]
- **Total Params:** ~8-10M (dual encoder + decoder)

**🔧 Technical Details:**
- Input: 256×256
- Batch size: 6
- Learning rate: 1e-4
- Training time: ~100 epochs × 171 iters = 17,100 iterations
- **9-channel support:** ✅ MixVisionTransformer uses Conv2d patch embedding

**⚡ Performance:**
- Speed: ~1.5-2s per 256×256 tile
- Memory: ~6-8GB VRAM
- Expected F1: 0.88-0.90

**👍 Ưu điểm:**
- Explicit bi-temporal interaction (Spatial + Channel Exchange)
- Medium size (~8-10M params)
- Good balance: accuracy vs efficiency
- Pretrained on ImageNet

**👎 Nhược điểm:**
- Phức tạp hơn CNN truyền thống
- Cần pretrained MiT weights

---

### 3. SNUNet-CD (Dense Siamese Network)

**📄 Paper:** Fang et al., IEEE GRSL 2022

**🏗️ Architecture:**
- **Backbone:** Nested UNet with Dense Connections
  - Encoder: 5 levels [32, 64, 128, 256, 512]
  - Decoder: Dense skip connections (0_1, 0_2, 0_3, 0_4)
- **ECAM:** Enhanced Channel Attention Module
  - Inter-layer attention (ca1)
  - Intra-layer attention (ca)
- **Total Params:** ~4-8M (base_channels=32)

**🔧 Technical Details:**
- Input: 256×256
- Batch size: 8
- Learning rate: 1e-3
- Training time: ~100 epochs × 128 iters = 12,800 iterations
- **9-channel support:** ✅ Standard Conv2d with `in_channels` parameter

**⚡ Performance:**
- Speed: ~0.8-1s per 256×256 tile (fastest)
- Memory: ~4-6GB VRAM (smallest)
- Expected F1: 0.86-0.88

**👍 Ưu điểm:**
- Lightweight (~4-8M params)
- Fast inference (fastest trong 3 models)
- Train from scratch (không cần pretrained)
- Pure CNN → stable training
- Channel attention cho multi-spectral data

**👎 Nhược điểm:**
- Accuracy thấp hơn Transformer models
- Không tận dụng pretrained weights

---

## SO SÁNH

| Criterion | BAN | Changer | SNUNet-CD |
|-----------|-----|---------|-----------|
| **Architecture** | Asymmetric Dual Transformer | Symmetric Transformer + Interaction | Dense Siamese CNN |
| **Parameters** | ~90M | ~8-10M | ~4-8M |
| **Pretrained** | CLIP + ImageNet | ImageNet | None (from scratch) |
| **Batch Size** | 4 | 6 | 8 |
| **Speed** | ~3-4s/tile (slow) | ~1.5-2s/tile (medium) | ~0.8-1s/tile (fast) |
| **Memory** | ~8-10GB | ~6-8GB | ~4-6GB |
| **Expected F1** | 0.89-0.91 (highest) | 0.88-0.90 (medium) | 0.86-0.88 (lowest) |
| **9-ch Support** | ✅ Patch embedding | ✅ Patch embedding | ✅ Conv2d |
| **Complexity** | High | Medium | Low |

---

## DIVERSITY ANALYSIS

### Architecture Diversity ✅
- **BAN:** Heavy Transformer (ViT + MiT)
- **Changer:** Medium Transformer (MiT with Interaction)
- **SNUNet-CD:** Lightweight CNN (Nested UNet)

### Design Philosophy Diversity ✅
- **BAN:** Asymmetric (different encoders for different purposes)
- **Changer:** Symmetric with explicit interaction (Spatial + Channel Exchange)
- **SNUNet-CD:** Symmetric with attention (Dense connections + ECAM)

### Pretrained Strategy Diversity ✅
- **BAN:** CLIP (vision-language) + ImageNet (visual)
- **Changer:** ImageNet (visual only)
- **SNUNet-CD:** From scratch (no pretrained)

### Inference Speed Diversity ✅
- **BAN:** Slow (~3-4s)
- **Changer:** Medium (~1.5-2s)
- **SNUNet-CD:** Fast (~0.8-1s)

---

## TẠI SAO KHÔNG CHỌN TINYCD/TINYCDV2?

**❌ TinyCDv2 bị loại vì:**

1. **Architecture hardcoded cho 3 channels:**
```python
# File: open-cd/opencd/models/backbones/tinycd.py:164
self._first_mix = MixingMaskAttentionBlock(6, 3, [3, 10, 5], [10, 5, 1])
```
- Số `6` là cố định = 3 channels × 2 timesteps
- `MixingBlock` dùng grouped convolution với `groups=3`
- Nhận 18 channels nhưng expect 6 → Error

2. **EfficientNet backbone pretrained trên RGB:**
```python
entire_model = torchvision.models.efficientnet_b4(pretrained=True).features
```
- Pretrained weights cho 3 channels
- Không thể load weights cho 9 channels

3. **Để sửa phải rewrite toàn bộ architecture:**
- Thay đổi tất cả MixingBlock layers
- Loại bỏ pretrained weights
- Mất ưu điểm "lightweight + pretrained"

**Quyết định:** Thay TinyCDv2 → SNUNet-CD
- SNUNet-CD cũng lightweight (~4-8M vs ~1.5M)
- Native support 9 channels
- Pure CNN như TinyCDv2
- Có ECAM attention mechanism

---

## KẾT LUẬN

### ✅ Bộ 3 models (BAN, Changer, SNUNet-CD) là lựa chọn tốt nhất vì:

1. **Tất cả đều hỗ trợ 9 channels natively** ← Yêu cầu bắt buộc
2. **Đa dạng tối đa:**
   - Architecture: Transformer (Heavy/Medium) vs CNN (Lightweight)
   - Design: Asymmetric vs Symmetric + Interaction vs Dense + Attention
   - Pretrained: CLIP+ImageNet vs ImageNet vs From Scratch
   - Speed: Slow vs Medium vs Fast
3. **Fit GPU 16GB:** Batch sizes 4/6/8 đều chạy được
4. **State-of-the-art:** Papers từ 2022-2024
5. **Comprehensive comparison:** Cover nhiều khía cạnh khác nhau

### 📊 Expected Outcome:
- BAN: Highest accuracy (0.89-0.91) but slowest
- Changer: Good balance (0.88-0.90) between accuracy and speed
- SNUNet-CD: Fastest (0.86-0.88) but lowest accuracy

### 🎯 Thesis Value:
- So sánh Transformer vs CNN cho multi-spectral change detection
- Phân tích trade-off: accuracy vs speed vs model size
- Đánh giá vai trò của pretrained weights (CLIP/ImageNet/None)
- Khuyến nghị deployment cho production (SNUNet-CD for speed, BAN for accuracy)

---

**Status:** ✅ Đã xác nhận 3 models phù hợp về mọi mặt
**Next:** Bắt đầu training (SNUNet-CD → Changer → BAN)
