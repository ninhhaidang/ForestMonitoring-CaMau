# 🧠 Giải Thích Chi Tiết 3 Kiến Trúc Deep Learning Models

## 📋 Mục Lục
- [Tổng Quan](#tổng-quan)
- [1. Spatial Context CNN](#1-spatial-context-cnn)
- [2. Multi-Scale CNN](#2-multi-scale-cnn)
- [3. Shallow U-Net](#3-shallow-u-net)
- [4. Multi-Scale CNN (NDVI-Weighted)](#4-multi-scale-cnn-ndvi-weighted)
- [So Sánh Các Models](#so-sánh-các-models)
- [Khi Nào Dùng Model Nào](#khi-nào-dùng-model-nào)

---

## 🎯 Tổng Quan

Cả 3 models đều là **shallow CNN** (CNN nông) được thiết kế cho:
- ✅ Dữ liệu hạn chế (~900 training samples)
- ✅ Input: 14 channels (Sentinel-2: 2024 + 2025)
- ✅ Output: Probability map (128×128) cho mỗi pixel
- ✅ Task: Binary classification (Deforestation vs No Deforestation)

**Tại sao dùng shallow CNN thay vì deep CNN (ResNet, VGG)?**
- Deep CNN cần hàng triệu training samples
- Với 900 samples, deep CNN sẽ **overfit nặng**
- Shallow CNN học được basic spatial patterns là đủ

---

## 1. Spatial Context CNN

### 🏗️ Kiến Trúc

```
Input (14, 128, 128)
       ↓
┌──────────────────┐
│  Conv 3×3 (32)   │  ← Layer 1: Extract basic features
│  BatchNorm       │
│  ReLU            │
└──────────────────┘
       ↓
┌──────────────────┐
│  Conv 3×3 (32)   │  ← Layer 2: Spatial smoothing
│  BatchNorm       │
│  ReLU            │
└──────────────────┘
       ↓
┌──────────────────┐
│  Conv 1×1 (1)    │  ← Layer 3: Output projection
└──────────────────┘
       ↓
Output (1, 128, 128)  ← Logits (before sigmoid)
```

### 🔍 Cách Hoạt Động (Chi Tiết)

#### **Bước 1: Input Processing**
```python
# Input: Batch of patches
x = torch.randn(64, 14, 128, 128)  # (Batch, Channels, Height, Width)

# 14 channels breakdown:
# Channels 0-6:  S2_2024 [Blue, Green, Red, NIR, NDVI, NBR, NDMI]
# Channels 7-13: S2_2025 [Blue, Green, Red, NIR, NDVI, NBR, NDMI]
```

#### **Bước 2: First Convolution Layer**
```python
# Conv2d(14 → 32, kernel=3×3, padding=1)
# Mỗi filter học 1 pattern từ 14 input channels

Filter 1: Có thể học "NDVI giảm mạnh" pattern
  - Weight cao cho NDVI_2024 (positive)
  - Weight cao cho NDVI_2025 (negative)
  - → Kích hoạt mạnh khi NDVI giảm

Filter 2: Học "NIR thay đổi" pattern
Filter 3: Học "edge detection" pattern
...
Filter 32: 32 patterns khác nhau
```

**Receptive Field sau layer 1:**
- Mỗi pixel output nhìn thấy vùng **3×3 pixels** từ input
- Tương đương **30m × 30m** trên thực địa (Sentinel-2: 10m/pixel)

#### **Bước 3: Second Convolution Layer**
```python
# Conv2d(32 → 32, kernel=3×3, padding=1)
# Combines patterns from layer 1

# Ví dụ:
# Layer 1 output: [NDVI_drop, NIR_change, edge, texture, ...]
# Layer 2 learns: "NDVI_drop AND NIR_change = likely deforestation"
```

**Receptive Field sau layer 2:**
- Mỗi pixel nhìn thấy **5×5 pixels** từ input gốc
- Tương đương **50m × 50m**

**BatchNorm + ReLU:**
```python
# BatchNorm: Normalize activations (giúp training stable)
# ReLU: max(0, x) - Loại bỏ negative values, thêm non-linearity
```

#### **Bước 4: Output Layer**
```python
# Conv 1×1 (32 → 1): Weighted sum of 32 features → single output
# Không có activation (sigmoid sẽ apply sau khi tính loss)

output_logits = conv3(x)  # (64, 1, 128, 128)
probabilities = torch.sigmoid(output_logits)  # (64, 1, 128, 128)
# Each pixel: probability of deforestation [0, 1]
```

### 📊 Thông Số

| Metric | Value |
|--------|-------|
| **Tổng tham số** | ~13,500 |
| **Receptive field** | 5×5 pixels (50m × 50m) |
| **Layers** | 3 conv layers |
| **Ưu điểm** | Nhanh nhất, ít tham số nhất |
| **Nhược điểm** | Receptive field nhỏ, chỉ nhìn thấy context gần |

### 💡 Khi Nào Dùng

- ✅ Cần inference nhanh
- ✅ Tài nguyên hạn chế (embedded devices)
- ✅ Baseline đơn giản
- ❌ Không phù hợp khi cần context rộng

---

## 2. Multi-Scale CNN

### 🏗️ Kiến Trúc

```
Input (14, 128, 128)
       ↓
    ╔═══╦═══╗
    ║   ║   ║
┌───▼───────▼────┐
│ Branch A  Branch B │  ← Layer 1: Multi-scale extraction
│ Conv 3×3  Conv 5×5 │     3×3: Fine details
│   (32)     (32)    │     5×5: Coarse patterns
└────┬───────┬──────┘
     └───┬───┘
         ↓
   Concat (64)
         ↓
┌──────────────────┐
│  Conv 3×3 (64)   │  ← Layer 2: Fuse multi-scale info
│  Conv 5×5 (64)   │
└──────────────────┘
         ↓
   Concat (128)
         ↓
┌──────────────────┐
│  Conv 3×3 (64)   │  ← Layer 3: Refine features
└──────────────────┘
         ↓
┌──────────────────┐
│  Conv 3×3 (32)   │  ← Layer 4: Reduce dimensions
└──────────────────┘
         ↓
┌──────────────────┐
│  Conv 1×1 (1)    │  ← Output
└──────────────────┘
         ↓
Output (1, 128, 128)
```

### 🔍 Cách Hoạt Động (Chi Tiết)

#### **Đặc Điểm Chính: Multi-Scale Branches**

```python
# Layer 1: Parallel branches
branch_3x3 = Conv2d(14, 32, kernel=3)  # Fine-grained
branch_5x5 = Conv2d(14, 32, kernel=5)  # Coarse

# Example:
# Branch 3×3 learns: Small clearings, edges, local changes
# Branch 5×5 learns: Large patches, spatial patterns, context
```

**Tại sao cần multi-scale?**
```
Deforestation patterns có nhiều scales khác nhau:

Scale nhỏ (3×3):        Scale lớn (5×5):
┌─┬─┬─┐                ┌─┬─┬─┬─┬─┐
│ │▓│ │   ← Cây        │ │ │ │ │ │
├─┼─┼─┤    đơn lẻ      ├─┼─┼─┼─┼─┤
│▓│▓│▓│    chặt phá     │ │▓│▓│▓│ │  ← Khu vực
├─┼─┼─┤                ├─┼─┼─┼─┼─┤    rộng lớn
│ │ │ │                │▓│▓│▓│▓│▓│    bị chặt
└─┴─┴─┘                └─┴─┴─┴─┴─┘
```

#### **Bước 2: Feature Fusion**

```python
# Concatenate multi-scale features
x = torch.cat([branch_3x3, branch_5x5], dim=1)  # (64 channels)

# Now có cả fine details VÀ coarse context!
# Model có thể học:
# - "Nếu 3×3 detect edge VÀ 5×5 detect large clearing → Deforestation!"
```

#### **Receptive Fields**

```
Layer 1:
- Branch 3×3: RF = 3×3 (30m × 30m)
- Branch 5×5: RF = 5×5 (50m × 50m)

Layer 2:
- RF = 7×7 (3×3 path) hoặc 9×9 (5×5 path)
- Tương đương 70m - 90m

Final RF: ~9×9 pixels (90m × 90m)
→ Gấp đôi Spatial Context CNN!
```

### 📊 Thông Số

| Metric | Value |
|--------|-------|
| **Tổng tham số** | ~90,000 |
| **Receptive field** | 9×9 pixels (90m × 90m) |
| **Layers** | 5 conv layers (2 branches) |
| **Ưu điểm** | Cân bằng tốt, multi-scale features |
| **Nhược điểm** | Chậm hơn Spatial CNN ~2× |

### 💡 Khi Nào Dùng

- ✅ **Production use** (khuyến nghị)
- ✅ Cần detect cả small và large deforestation
- ✅ Cân bằng giữa accuracy và speed
- ❌ Không phù hợp khi cần real-time

---

## 3. Shallow U-Net

### 🏗️ Kiến Trúc

```
Input (14, 128×128)
       ↓
┌──────────────────┐
│ Encoder Block 1  │  32 channels, 128×128
│  Conv + Conv     │
└────────┬─────────┘
         │ Skip Connection 1
         ↓
    MaxPool 2×2
         ↓
┌──────────────────┐
│ Encoder Block 2  │  64 channels, 64×64
│  Conv + Conv     │
└────────┬─────────┘
         │ Skip Connection 2
         ↓
    MaxPool 2×2
         ↓
┌──────────────────┐
│ Encoder Block 3  │  128 channels, 32×32
│  Conv + Conv     │
└────────┬─────────┘
         │ Skip Connection 3
         ↓
    MaxPool 2×2
         ↓
┌──────────────────┐
│   Bottleneck     │  256 channels, 16×16
│  Conv + Conv     │  ← Deepest point
└──────────────────┘
         ↓
    Upsample 2×2
         ↓
┌──────────────────┐
│ Decoder Block 3  │  128 channels, 32×32
│ Concat Skip 3    │  ← Fuse encoder features
│  Conv + Conv     │
└──────────────────┘
         ↓
    Upsample 2×2
         ↓
┌──────────────────┐
│ Decoder Block 2  │  64 channels, 64×64
│ Concat Skip 2    │
│  Conv + Conv     │
└──────────────────┘
         ↓
    Upsample 2×2
         ↓
┌──────────────────┐
│ Decoder Block 1  │  32 channels, 128×128
│ Concat Skip 1    │
│  Conv + Conv     │
└──────────────────┘
         ↓
┌──────────────────┐
│  Conv 1×1 (1)    │  Output
└──────────────────┘
         ↓
Output (1, 128×128)
```

### 🔍 Cách Hoạt Động (Chi Tiết)

#### **Khái Niệm U-Net**

U-Net giống như "zoom out → zoom in":
1. **Encoder** (downsampling): Thu nhỏ ảnh, tăng features → Nhìn context rộng
2. **Bottleneck**: Representation ở level cao nhất
3. **Decoder** (upsampling): Phóng to lại, giảm features → Recover spatial details
4. **Skip Connections**: Ghép nối encoder-decoder → Giữ lại chi tiết

#### **Bước 1: Encoder (Contracting Path)**

```python
# Encoder Block 1 (128×128)
x1 = conv_block(input, out_channels=32)
# Learn: Low-level features (edges, textures)

# Downsample
x_pool = MaxPool2d(2)(x1)  # → 64×64

# Encoder Block 2 (64×64)
x2 = conv_block(x_pool, out_channels=64)
# Learn: Mid-level features (small objects, patterns)

# Downsample
x_pool = MaxPool2d(2)(x2)  # → 32×32

# Encoder Block 3 (32×32)
x3 = conv_block(x_pool, out_channels=128)
# Learn: High-level features (large structures)

# Downsample
x_pool = MaxPool2d(2)(x3)  # → 16×16
```

**Receptive Field tăng dần:**
- Block 1 (128×128): RF ~ 5×5 pixels (50m)
- Block 2 (64×64): RF ~ 13×13 pixels (130m)
- Block 3 (32×32): RF ~ 29×29 pixels (290m)
- Bottleneck (16×16): RF ~ 61×61 pixels (610m) ← Nhìn rất rộng!

#### **Bước 2: Bottleneck**

```python
# Smallest spatial resolution, highest channels
bottleneck = conv_block(x_pool, out_channels=256)  # 16×16×256

# Tại đây model có "global understanding" của patch
# Mỗi pixel trong bottleneck nhìn thấy ~600m × 600m!
```

#### **Bước 3: Decoder (Expanding Path)**

```python
# Upsample + Skip Connection 3
up3 = Upsample(bottleneck)  # 16×16 → 32×32
concat3 = torch.cat([up3, x3], dim=1)  # Fuse với encoder features
dec3 = conv_block(concat3, out_channels=128)

# Skip connection là QUAN TRỌNG:
# - x3 chứa spatial details từ encoder
# - up3 chứa semantic info từ bottleneck
# → Concat = Chi tiết + Ngữ nghĩa!

# Tương tự cho decoder 2, 1
```

**Tại sao cần skip connections?**

```
Không có skip:              Có skip:
Encoder → Bottleneck      Encoder ──┐
   ↓                         ↓       │
Decoder (mất details)     Decoder ←─┘ (keep details)

Output: Blurry            Output: Sharp
```

#### **Bước 4: Output**

```python
# Final conv 1×1
output = Conv2d(32, 1, kernel=1)(dec1)  # 128×128×1

# Output combines:
# - Low-level spatial details (từ skip connections)
# - High-level semantic understanding (từ bottleneck)
# → Best of both worlds!
```

### 📊 Thông Số

| Metric | Value |
|--------|-------|
| **Tổng tham số** | ~476,000 |
| **Receptive field** | 61×61 pixels (610m × 610m) |
| **Layers** | 8-10 conv layers + skip connections |
| **Ưu điểm** | RF rất lớn, smoothest output, best accuracy |
| **Nhược điểm** | Chậm nhất, nhiều tham số nhất |

### 💡 Khi Nào Dùng

- ✅ Cần **best quality** predictions
- ✅ Cần smooth, connected deforestation maps
- ✅ Có GPU mạnh, không quan tâm speed
- ❌ Không dùng cho embedded/mobile

---

## 4. Multi-Scale CNN (NDVI-Weighted)

### 🏗️ Kiến Trúc (Mới!)

```
Input (14, 128, 128)
       ↓
  ╔════╩════╗
  ║         ║
  ║    ┌────▼────────┐
  ║    │  Channel    │  ← Learn importance weights
  ║    │  Attention  │     for 14 channels
  ║    └────┬────────┘
  ║         ↓
  ║    Weighted Input
  ║         ↓
  ║    [Original Multi-Scale CNN Architecture]
  ║         ↓
  ║    Main Features (128)
  ║         │
  └────────┼──────────┐
           │          │
      ┌────▼────┐     │
      │  NDVI   │     │  ← Explicit NDVI change branch
      │  Change │     │
      │ Branch  │     │
      └────┬────┘     │
           │          │
           └────┬─────┘
                ↓
          Concat (144)
                ↓
           [Fusion Layers]
                ↓
        Output (1, 128, 128)
```

### 🔍 Cách Hoạt Động

#### **Component 1: Channel Attention**

```python
# Squeeze: Global average pooling
gap = AdaptiveAvgPool2d(1)(x)  # (B, 14, 1, 1)

# Excitation: Learn channel weights
weights = FC_layers(gap)  # (B, 14, 1, 1)
weights = sigmoid(weights)  # [0, 1]

# Example learned weights:
# Channel 4 (NDVI_2024): 0.85  ← High!
# Channel 11 (NDVI_2025): 0.90  ← High!
# Channel 0 (Blue_2024): 0.35  ← Low
# ...

# Reweight input
x_weighted = x * weights

# Effect: NDVI channels được "nhấn mạnh" hơn!
```

#### **Component 2: NDVI Difference Branch**

```python
# Extract NDVI
ndvi_2024 = input[:, 4, :, :]   # (B, 1, 128, 128)
ndvi_2025 = input[:, 11, :, :]  # (B, 1, 128, 128)

# Compute change
ndvi_change = ndvi_2025 - ndvi_2024  # (B, 1, 128, 128)

# Process with small CNN
ndvi_features = conv_layers(ndvi_change)  # (B, 16, 128, 128)

# Fuse với main features
final = concat([main_features, ndvi_features])  # (B, 144, 128, 128)
```

**Lợi ích:**
1. **Channel Attention**: Model tự học NDVI quan trọng
2. **NDVI Branch**: Force model phải xem NDVI change
3. **Fusion**: Combine spatial patterns + temporal change

### 📊 Thông Số

| Metric | Value |
|--------|-------|
| **Tổng tham số** | ~100,000 (+10K so với MultiScale) |
| **Receptive field** | 9×9 pixels (90m × 90m) |
| **Ưu điểm** | Emphasize NDVI, better align với ground truth |
| **Nhược điểm** | Phức tạp hơn, cần train riêng |

---

## 📊 So Sánh Các Models

### Bảng So Sánh Tổng Quan

| Feature | Spatial CNN | Multi-Scale CNN | Shallow U-Net | MultiScale NDVI-Weighted |
|---------|-------------|-----------------|---------------|-------------------------|
| **Parameters** | 13K | 90K | 476K | 100K |
| **Receptive Field** | 50m | 90m | 610m | 90m |
| **Layers** | 3 | 5 | 8-10 | 6 |
| **Speed** | ⚡⚡⚡ Fastest | ⚡⚡ Fast | ⚡ Slow | ⚡⚡ Fast |
| **Accuracy** | ⭐⭐ Good | ⭐⭐⭐ Better | ⭐⭐⭐⭐ Best | ⭐⭐⭐? TBD |
| **Smoothness** | Low | Medium | High | Medium-High? |
| **GPU Memory** | 50MB | 150MB | 400MB | 160MB |

### Receptive Field Visualization

```
Spatial CNN (50m × 50m):
┌─────┐
│ 5×5 │  ← Local context only
└─────┘

Multi-Scale CNN (90m × 90m):
┌─────────┐
│   9×9   │  ← Medium context
└─────────┘

Shallow U-Net (610m × 610m):
┌───────────────────────────┐
│                           │
│         61×61             │  ← Very large context!
│                           │
└───────────────────────────┘
```

### Trade-offs

```
                  Simple ←──────────────→ Complex
                  Fast   ←──────────────→ Slow

Spatial CNN ●────────────────────────────────

MultiScale CNN ────────●─────────────────────

U-Net ─────────────────────────────────●─────

         Low Accuracy ←──────────────→ High Accuracy
         Less Smooth  ←──────────────→ More Smooth
```

---

## 🎯 Khi Nào Dùng Model Nào?

### Use Cases

#### **Spatial Context CNN**
```
✅ Khi nào dùng:
- Cần inference real-time
- Deploy trên edge devices (Raspberry Pi, drones)
- Baseline nhanh để test
- Dữ liệu cực kỳ hạn chế

❌ Không nên dùng khi:
- Cần highest accuracy
- Có nhiều small, scattered deforestation
- Có GPU mạnh
```

#### **Multi-Scale CNN** ⭐ **KHUYẾN NGHỊ**
```
✅ Khi nào dùng:
- PRODUCTION USE
- Cần balance tốt giữa accuracy & speed
- Detect cả small lẫn large clearings
- GPU trung bình (GTX 1060+)

❌ Không nên dùng khi:
- Cần absolutely best quality
- Speed không quan trọng
```

#### **Shallow U-Net**
```
✅ Khi nào dùng:
- Cần BEST QUALITY maps
- Smooth, connected predictions
- Research/analysis purposes
- Có GPU mạnh (RTX 3060+)

❌ Không nên dùng khi:
- Cần real-time inference
- RAM/GPU memory hạn chế
- Deploy lên mobile
```

#### **Multi-Scale NDVI-Weighted**
```
✅ Khi nào dùng:
- NDVI change là strong indicator
- Muốn model align với physical process
- Có thời gian train thêm model
- Cần interpretability

❌ Không nên dùng khi:
- Chưa validate NDVI change effectiveness
- Cần đơn giản
```

---

## 🧮 Ví Dụ Tính Toán

### Memory Usage (Batch size = 64)

```python
# Spatial CNN
Input: 64 × 14 × 128 × 128 × 4 bytes = 58.7 MB
Features: ~100 MB
Total: ~200 MB

# Multi-Scale CNN
Input: 64 × 14 × 128 × 128 × 4 bytes = 58.7 MB
Features: ~300 MB
Total: ~400 MB

# Shallow U-Net
Input: 64 × 14 × 128 × 128 × 4 bytes = 58.7 MB
Features: ~800 MB (do skip connections)
Total: ~1 GB
```

### Inference Speed (1 patch on RTX A4000)

```
Spatial CNN:      0.5 ms
Multi-Scale CNN:  1.2 ms
Shallow U-Net:    3.5 ms

Full image (10917 × 12547, ~33K patches):
Spatial CNN:      16 seconds
Multi-Scale CNN:  40 seconds
Shallow U-Net:    115 seconds (2 minutes)
```

---

## 📚 Tài Liệu Tham Khảo

1. **U-Net**: Ronneberger et al. (2015) "U-Net: Convolutional Networks for Biomedical Image Segmentation"
2. **Multi-Scale**: Inception architecture (Szegedy et al., 2015)
3. **Channel Attention**: Hu et al. (2018) "Squeeze-and-Excitation Networks"
4. **Shallow CNNs for Remote Sensing**: Zhong et al. (2020)

---

## 💡 Tips Để Hiểu Rõ Hơn

### 1. Visualize Receptive Field
```python
# Run this to see what area each model "sees"
from src.models import get_model
model = get_model('shallow_unet', in_channels=14)
# → Receptive field calculator
```

### 2. Xem Feature Maps
```python
# Hook vào intermediate layers để xem model học gì
# Notebook 04, cell visualization
```

### 3. Compare Predictions
```python
# Run notebook 04 để xem side-by-side comparison
# RGB | Spatial | MultiScale | U-Net | NDVI Change
```

---

**Tóm lại:**
- 🏃 **Spatial CNN**: Nhanh nhưng basic
- 🎯 **Multi-Scale CNN**: Sweet spot cho production
- 🎨 **Shallow U-Net**: Best quality
- 🌿 **NDVI-Weighted**: Thêm domain knowledge vào CNN
