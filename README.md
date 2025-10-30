# 🌲 Ứng Dụng Viễn Thám và Học Sâu Trong Giám Sát Biến Động Rừng Tỉnh Cà Mau

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Đồ Án Tốt Nghiệp**  
> Sinh viên: Ninh Hải Đăng (MSSV: 21021411)  
> Viện Công nghệ Hàng không Vũ trụ  
> Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội  
> Email: ninhhaidangg@gmail.com | GitHub: [@ninhhaidang](https://github.com/ninhhaidang)

---

## 📋 Mục Lục

- [Tóm Tắt](#tóm-tắt)
- [Giới Thiệu](#giới-thiệu)
- [Khu Vực Nghiên Cứu](#khu-vực-nghiên-cứu)
- [Phương Pháp Nghiên Cứu](#phương-pháp-nghiên-cứu)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Cài Đặt](#cài-đặt)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Kiến Trúc Mô Hình](#kiến-trúc-mô-hình)
- [Kết Quả](#kết-quả)
- [Thảo Luận](#thảo-luận)
- [Hướng Phát Triển](#hướng-phát-triển)
- [Tài Liệu](#tài-liệu)
- [Tài Liệu Tham Khảo](#tài-liệu-tham-khảo)
- [Lời Cảm Ơn](#lời-cảm-ơn)
- [Giấy Phép](#giấy-phép)

---

## 📖 Tóm Tắt

Giám sát biến động rừng là nhiệm vụ quan trọng đối với bảo tồn môi trường và quản lý tài nguyên rừng. Tỉnh Cà Mau với hệ sinh thái rừng ngập mặn đặc trưng đang đối mặt với nhiều áp lực từ hoạt động nuôi trồng thủy sản và biến đổi khí hậu, đòi hỏi phương pháp giám sát hiệu quả và kịp thời.

Các phương pháp học máy truyền thống (Random Forest, Gradient Boosting, SVM) đạt độ chính xác cao trong phân loại từng pixel nhưng gặp phải vấn đề nhiễu muối tiêu (salt-and-pepper noise) do thiếu nhận thức về ngữ cảnh không gian. Điều này dẫn đến bản đồ kết quả có nhiều pixel bị phân loại sai rời rạc, làm giảm chất lượng thông tin cho quản lý rừng.

Đồ án này đề xuất **khung deep learning đa thời gian** tận dụng dữ liệu đa phổ Sentinel-2 để phát hiện các khu vực biến động rừng tại tỉnh Cà Mau giữa hai thời điểm 2024 và 2025. Ba kiến trúc mạng nơ-ron tích chập nông (shallow CNN) và một mô hình machine learning truyền thống được triển khai và so sánh:

**CNN Models:**
1. **Spatial Context CNN** (~30K tham số) - Gần nhất với phương pháp ML, bổ sung làm mượt không gian
2. **Multi-Scale CNN** (~80K tham số) - Cân bằng, học đặc trưng đa tỷ lệ
3. **Shallow U-Net** (~120K tham số) - Kiến trúc encoder-decoder cho tính liên kết không gian tối ưu

**Traditional ML:**
4. **Random Forest** (100 trees) - Baseline machine learning cho so sánh

Khung nghiên cứu xử lý 14 kênh phổ (7 kênh × 2 thời điểm từ Sentinel-2) sử dụng các patches 128×128 pixels, huấn luyện trên 1.285 điểm có nhãn với các lớp cân bằng (49,4% mất rừng vs 50,6% không mất rừng). Các mô hình được tối ưu hóa cho GPU NVIDIA RTX A4000 16GB và tạo ra bản đồ xác suất liên tục (0-1), kỳ vọng sẽ giảm nhiễu đáng kể so với phương pháp ML truyền thống.

**Từ khóa:** Giám sát rừng, Cà Mau, Rừng ngập mặn, Phân tích đa thời gian, Deep Learning, Sentinel-1/2, Viễn thám, CNN

---

## 🎯 Giới Thiệu

### Bối Cảnh

Tỉnh Cà Mau nằm ở cực Nam Việt Nam, sở hữu hệ sinh thái rừng ngập mặn rộng lớn với vai trò quan trọng trong việc bảo vệ bờ biển, duy trì đa dạng sinh học và lưu trữ carbon. Tuy nhiên, rừng ngập mặn Cà Mau đang đối mặt với nhiều thách thức:

- **Chuyển đổi mục đích sử dụng đất**: Mở rộng diện tích nuôi trồng thủy sản (tôm, cua)
- **Biến đổi khí hậu**: Xâm nhập mặn, nước biển dâng, bão lũ
- **Khai thác không bền vững**: Chặt phá để lấy gỗ, than
- **Suy thoái tự nhiên**: Già cỗi, bệnh hại

Việc giám sát biến động rừng truyền thống dựa vào điều tra thực địa tốn kém và không thể cập nhật thường xuyên trên diện rộng. Viễn thám vệ tinh cung cấp giải pháp hiệu quả nhưng cần phương pháp xử lý tiên tiến để tạo thông tin chính xác và kịp thời.

### Phát Biểu Bài Toán

**Đầu vào:**
- **Dữ liệu**: Ảnh Sentinel-2 (đa phổ: B, G, R, NIR và các chỉ số NDVI, NBR, NDMI) từ hai thời điểm (2024 và 2025)
- **Khu vực**: Tỉnh Cà Mau
- **Ground truth**: 1.285 điểm có nhãn (635 điểm mất rừng, 650 điểm không mất rừng)
- **Thách thức**: Phương pháp ML hiện tại (RF/GBT/SVM) đạt độ chính xác cao (>90%) nhưng tạo bản đồ có nhiễu pixel rời rạc

**Mục tiêu:**
1. Phát triển các mô hình deep learning nông phù hợp với dữ liệu hạn chế
2. Tích hợp ngữ cảnh không gian để giảm nhiễu muối tiêu
3. Duy trì hoặc cải thiện độ chính xác so với ML baseline
4. Tạo bản đồ xác suất mượt, dễ diễn giải cho công tác quản lý
5. So sánh 3 kiến trúc CNN và 1 mô hình Random Forest

### Câu Hỏi Nghiên Cứu

1. Liệu các kiến trúc CNN nông có thể học được đặc trưng không gian hiệu quả với chỉ ~1.300 mẫu huấn luyện?
2. Kích thước vùng tiếp nhận (receptive field) nào phù hợp nhất cho đặc điểm rừng ngập mặn Cà Mau?
3. Kiến trúc nào cân bằng tốt nhất giữa độ chính xác, độ mượt và tốc độ tính toán?
4. Dữ liệu đa nguồn (SAR + đa phổ) và đa thời gian có cải thiện khả năng phát hiện biến động so với đơn nguồn?

---

## 🗺️ Khu Vực Nghiên Cứu

### Vị Trí Địa Lý

- **Tỉnh**: Cà Mau
- **Vị trí**: Cực Nam Việt Nam (8°30' - 9°30' Bắc, 104°45' - 105°30' Đông)
- **Diện tích tự nhiên**: ~5.331 km²
- **Đặc điểm**: Địa hình thấp, nhiều sông rạch, chịu ảnh hưởng triều cường

### Đặc Điểm Rừng

- **Loại rừng chính**: Rừng ngập mặn (mangrove forest)
- **Các loài ưu thế**: Đước (Rhizophora), Tràm (Melaleuca), Mắm (Avicennia)
- **Diện tích rừng**: ~40.000 ha (số liệu tham khảo, cần cập nhật)
- **Phân bố**: Tập trung ven biển và ven sông

### Áp Lực Lên Rừng

1. **Nuôi trồng thủy sản**: Chuyển đổi rừng thành ao nuôi tôm
2. **Khai thác gỗ**: Lấy gỗ xây dựng, làm than
3. **Biến đổi khí hậu**: Nước biển dâng, xâm nhập mặn
4. **Phát triển cơ sở hạ tầng**: Xây dựng đường, khu dân cư

---

## 🔬 Phương Pháp Nghiên Cứu

### Thu Thập và Tiền Xử Lý Dữ Liệu

#### Đặc Tả Dữ Liệu Đầu Vào

**Sentinel-1 (SAR C-band):**
- **Kênh**: 
  - VH (phân cực chéo Vertical-Horizontal)
  - R = VV - VH (tỷ số phân cực)
- **Độ phân giải không gian**: 10m
- **Ngày thu thập**: 
  - Thời điểm 1: 04/02/2024
  - Thời điểm 2: 22/02/2025
- **Ưu điểm**: Xuyên mây, hoạt động cả ngày đêm, nhạy cảm với cấu trúc thảm thực vật
- **Tiền xử lý**: 
  - Hiệu chuẩn bức xạ (radiometric calibration)
  - Lọc nhiễu đốm (speckle filtering) - Lee filter
  - Hiệu chính địa hình (terrain correction) - Range-Doppler

**Sentinel-2 (Multispectral):**
- **Kênh gốc**:
  - B4 (Red): 665 nm, 10m
  - B8 (NIR): 842 nm, 10m  
  - B11 (SWIR1): 1.610 nm, 20m → nội suy về 10m
  - B12 (SWIR2): 2.190 nm, 20m → nội suy về 10m
- **Chỉ số tính toán**:
  - NDVI = (B8 - B4) / (B8 + B4) - Chỉ số thực vật
  - NBR = (B8 - B12) / (B8 + B12) - Chỉ số cháy
  - NDMI = (B8 - B11) / (B8 + B11) - Chỉ số độ ẩm
- **Ngày thu thập**:
  - Thời điểm 1: 30/01/2024
  - Thời điểm 2: 28/02/2025
- **Độ che phủ mây**: <10%
- **Tiền xử lý**:
  - Hiệu chính khí quyển (atmospheric correction) - Sen2Cor
  - Loại bỏ mây (cloud masking)
  - Resample B11, B12 về 10m

#### Stack Đặc Trưng Đa Thời Gian

**Tổng cộng: 14 kênh phổ (Sentinel-2 only)**

| STT | Tên Kênh | Nguồn | Thời điểm | Ý nghĩa |
|-----|----------|-------|-----------|---------|
| 1 | Blue_2024 | S2 | 2024 | Phản xạ vùng xanh lam |
| 2 | Green_2024 | S2 | 2024 | Phản xạ vùng xanh lục |
| 3 | Red_2024 | S2 | 2024 | Phản xạ vùng đỏ |
| 4 | NIR_2024 | S2 | 2024 | Phản xạ cận hồng ngoại |
| 5 | NDVI_2024 | S2 | 2024 | Độ xanh thực vật |
| 6 | NBR_2024 | S2 | 2024 | Chỉ số cháy |
| 7 | NDMI_2024 | S2 | 2024 | Chỉ số độ ẩm |
| 8-14 | [Lặp lại] | S2 | 2025 | Cùng 7 kênh năm 2025 |

**Lý do sử dụng đa thời gian:**
- Phát hiện **thay đổi** giữa hai thời điểm chính xác hơn so với phân loại đơn thời điểm
- Giảm ảnh hưởng của biến động theo mùa (phenology)
- Tăng độ tin cậy thông qua so sánh trực tiếp

**Lý do chỉ dùng Sentinel-2 (không dùng Sentinel-1):**
- Sentinel-2 đa phổ cung cấp đủ thông tin về thảm thực vật
- Đơn giản hóa preprocessing (không cần xử lý SAR speckle noise)
- Giảm số lượng kênh đầu vào → giảm overfitting với dữ liệu hạn chế
- Sentinel-2 10m resolution phù hợp với kích thước mảng rừng

#### Trích Xuất Patches

**Quy trình:**
1. **Đầu vào**: File CSV chứa tọa độ UTM (x, y) và nhãn (0/1) của 1.285 điểm
2. **Trích xuất**: Với mỗi điểm (x, y):
   - Cắt vùng 128×128 pixels (1,28 km × 1,28 km) xung quanh điểm làm tâm
   - Lấy đầy đủ 14 kênh phổ (S2 only) → patch có kích thước 128×128×14
3. **Lưu trữ**: Mỗi patch lưu thành file `.npy` (NumPy array)

**Lý do chọn 128×128 pixels:**
- **Ngữ cảnh không gian**: 1,28×1,28 km đủ lớn để bao quát mẫu rừng/không rừng xung quanh
- **Bộ nhớ GPU**: Phù hợp với batch size 16-32 trên GPU 16GB
- **Receptive field**: Cho phép model học đặc trưng từ vùng lân cận rộng

**Phân bố lớp:**
- Lớp 0 (Không mất rừng): 650 mẫu (50,6%)
- Lớp 1 (Mất rừng): 635 mẫu (49,4%)
- **Nhận xét**: Dữ liệu cân bằng tốt, không cần weighted loss

#### Tăng Cường Dữ Liệu (Data Augmentation)

Do số lượng mẫu hạn chế (~1.300), áp dụng augmentation để tăng tính đa dạng:

| Kỹ thuật | Tham số | Mục đích |
|----------|---------|----------|
| **Rotation** | 90°, 180°, 270° | Bất biến với hướng quay |
| **Horizontal Flip** | p=0.5 | Tăng tính đối xứng |
| **Vertical Flip** | p=0.5 | Tăng tính đối xứng |
| **Gaussian Noise** | σ=0.01 | Tăng tính robust với nhiễu |

**Kích thước tập hiệu quả**: ~2.500-3.000 mẫu sau augmentation

#### Chia Tập Train/Validation/Test

```
Tổng: 1.285 patches
├── Training:   70% ≈ 900 patches   (huấn luyện model)
├── Validation: 15% ≈ 190 patches   (tuning hyperparameters, early stopping)
└── Test:       15% ≈ 195 patches   (đánh giá cuối cùng)
```

**Chia phân tầng (stratified split)**: Đảm bảo tỷ lệ class 0/1 giống nhau ở cả 3 tập

---

### Kiến Trúc Mô Hình

#### Mô Hình 1: Spatial Context CNN

**Triết lý thiết kế:**
- Giữ đơn giản như ML nhưng thêm khả năng học không gian
- "RF + spatial smoothing"

**Kiến trúc chi tiết:**

```
┌──────────────────────────────────────────┐
│ INPUT: 128×128×14                        │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ Conv2D(kernel=3×3, filters=32)           │
│ BatchNorm → ReLU                         │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ Conv2D(kernel=3×3, filters=32)           │
│ BatchNorm → ReLU                         │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ Conv2D(kernel=1×1, filters=1)            │
│ Sigmoid (output probability [0,1])       │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ OUTPUT: 128×128×1 (probability map)      │
└──────────────────────────────────────────┘
```

**Thông số:**
- Số lớp tích chập: 3
- Tổng tham số: ~30.000
- Vùng tiếp nhận: 5×5 pixels (50m × 50m)

**Đặc điểm:**
- Conv 3×3 đầu tiên: Học đặc trưng cục bộ
- Conv 3×3 thứ hai: Mở rộng receptive field
- Conv 1×1: Giống linear classifier của ML, kết hợp các đặc trưng
- Không có pooling → giữ nguyên độ phân giải

**Khi nào dùng:**
- Cần baseline đơn giản để so sánh
- Tài nguyên tính toán hạn chế
- Ưu tiên tốc độ hơn chất lượng

---

#### Mô Hình 2: Multi-Scale CNN

**Triết lý thiết kế:**
- Học đồng thời ở nhiều tỷ lệ không gian
- Phù hợp với mảng rừng có kích thước khác nhau

**Kiến trúc chi tiết:**

```
┌──────────────────────────────────────────┐
│ INPUT: 128×128×14                        │
└────────────────┬─────────────────────────┘
                 ↓
        ┌────────┴────────┐
        ↓                 ↓
┌───────────────┐  ┌──────────────┐
│   BRANCH 1    │  │   BRANCH 2   │
│ Conv(3×3, 32) │  │ Conv(5×5, 32)│
│ BatchNorm     │  │ BatchNorm    │
│ ReLU          │  │ ReLU         │
└───────┬───────┘  └──────┬───────┘
        └────────┬─────────┘
                 ↓
         ┌──────────────┐
         │ CONCATENATE  │
         │ 32+32=64 ch  │
         └──────┬───────┘
                ↓
┌──────────────────────────────────────────┐
│ Conv2D(3×3, 64) + BatchNorm + ReLU       │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ Conv2D(3×3, 64) + BatchNorm + ReLU       │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ Conv2D(1×1, 1) + Sigmoid                 │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ OUTPUT: 128×128×1                        │
└──────────────────────────────────────────┘
```

**Thông số:**
- Số lớp tích chập: 5 (2 branches + 3 fusion)
- Tổng tham số: ~80.000
- Vùng tiếp nhận: Branch 1 (7×7), Branch 2 (9×9)

**Đặc điểm:**
- **Branch 1 (3×3)**: Bắt giữ chi tiết nhỏ (cạnh, texture)
- **Branch 2 (5×5)**: Bắt giữ ngữ cảnh rộng (mảng rừng)
- **Concatenation**: Kết hợp thông tin đa tỷ lệ
- **Fusion layers**: Học cách kết hợp tối ưu hai nhánh

**Khi nào dùng:**
- **Khuyến nghị cho production**
- Cân bằng accuracy-speed-smoothness
- Khi kích thước mảng rừng thay đổi nhiều

---

#### Mô Hình 3: Shallow U-Net

**Triết lý thiết kế:**
- Encoder-decoder với skip connections
- "Shallow" = chỉ 1 level downsampling (không quá sâu)

**Kiến trúc chi tiết:**

```
┌──────────────────────────────────────────┐
│ INPUT: 128×128×14                        │
└──────────────┬───────────────────────────┘
               ↓
┌───────────── ENCODER ────────────────────┐
│ Conv(3×3, 32) → Conv(3×3, 32)            │──┐
│ BatchNorm, ReLU                          │  │
│ MaxPool(2×2) ↓ [64×64×32]                │  │
│                                          │  │
│ Conv(3×3, 64) → Conv(3×3, 64)            │  │
│ BatchNorm, ReLU                          │  │
│ MaxPool(2×2) ↓ [32×32×64]                │  │
└──────────────┬───────────────────────────┘  │
               ↓                               │
┌───────────── BOTTLENECK ─────────────────┐  │
│ Conv(3×3, 128) → Conv(3×3, 128)          │  │
│ BatchNorm, ReLU  [32×32×128]             │  │
└──────────────┬───────────────────────────┘  │
               ↓                               │
┌───────────── DECODER ────────────────────┐  │
│ Upsample(2×2) ↑ [64×64×128]              │  │
│                                          │  │
│ Concat ←─────────────────────────────────┘  │
│ [64×64×(128+64)=192]                     │  │
│                                          │  │
│ Conv(3×3, 64) → Conv(3×3, 64)            │  │
│ BatchNorm, ReLU                          │  │
│                                          │  │
│ Upsample(2×2) ↑ [128×128×64]             │  │
│                                          │  │
│ Concat ←─────────────────────────────────┘
│ [128×128×(64+32)=96]                     │
│                                          │
│ Conv(3×3, 32) → Conv(3×3, 32)            │
│ BatchNorm, ReLU                          │
│                                          │
│ Conv(1×1, 1) + Sigmoid                   │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ OUTPUT: 128×128×1                        │
└──────────────────────────────────────────┘
```

**Thông số:**
- Số lớp tích chập: 8-10
- Tổng tham số: ~120.000
- Vùng tiếp nhận: 13×13 pixels (130m × 130m)

**Đặc điểm:**
- **Encoder**: Trích xuất đặc trưng cấp cao thông qua downsampling
- **Bottleneck**: Biểu diễn ngữ nghĩa ở độ phân giải thấp
- **Decoder**: Phục hồi độ phân giải thông qua upsampling
- **Skip connections**: Giữ lại chi tiết không gian từ encoder
- Chỉ 1 level downsampling (shallow) tránh overfitting với ít data

**Khi nào dùng:**
- Cần chất lượng bản đồ tốt nhất
- Độ mượt quan trọng (xuất bản, báo cáo)
- Có đủ thời gian tính toán

---

### Cấu Hình Huấn Luyện

#### Hàm Loss

**Binary Cross-Entropy (BCE):**

```
L = -1/N Σ [y_i · log(ŷ_i) + (1-y_i) · log(1-ŷ_i)]
```

Trong đó:
- y_i: Nhãn thực tế (0 hoặc 1)
- ŷ_i: Xác suất dự đoán [0, 1]
- N: Số pixels trong batch

**Lý do chọn BCE:**
- Phù hợp cho bài toán phân loại nhị phân
- Dữ liệu cân bằng (49.4% vs 50.6%) → không cần weighted loss
- Đơn giản, ổn định trong training

#### Optimizer và Learning Rate

**Optimizer: Adam**
- β₁ = 0.9 (momentum)
- β₂ = 0.999 (RMSprop)
- ε = 1e-8
- Weight decay (L2 regularization) = 1e-4

**Learning Rate Schedule:**
- Initial LR: 1e-3
- **ReduceLROnPlateau**:
  - Monitor: Validation loss
  - Factor: 0.5 (giảm một nửa)
  - Patience: 5 epochs
  - Min LR: 1e-6

#### Training Configuration

```python
EPOCHS = 100
BATCH_SIZE = 16  # Tối ưu cho GPU 16GB
EARLY_STOPPING_PATIENCE = 10  # Stop nếu val_loss không giảm sau 10 epochs
```

#### Regularization

1. **Batch Normalization**: Sau mỗi Conv layer
2. **Dropout**: 0.2 (chỉ Model 3 - U-Net)
3. **Data Augmentation**: Như mô tả ở trên
4. **L2 Weight Decay**: 1e-4

---

### Chiến Lược Suy Luận (Inference)

#### Sliding Window với Overlap

**Vấn đề:** Ảnh đầy đủ Cà Mau rất lớn (vd: 20.000 × 20.000 pixels) → không thể input 1 lần

**Giải pháp:** Sliding window với overlap

```
Bước 1: Chia ảnh thành các cửa sổ 128×128
Bước 2: Stride = 64 pixels (overlap 50%)
Bước 3: Dự đoán từng window
Bước 4: Blend các vùng overlap (average)
Bước 5: Ghép thành bản đồ hoàn chỉnh
```

**Code logic:**

```python
stride = 64  # 50% overlap
output = zeros_like(image)
count = zeros_like(image)

for y in range(0, H-128+1, stride):
    for x in range(0, W-128+1, stride):
        patch = image[y:y+128, x:x+128, :]
        prob = model.predict(patch)  # 128×128×1
        
        output[y:y+128, x:x+128] += prob
        count[y:y+128, x:x+128] += 1

probability_map = output / count  # Average overlaps
```

**Xử lý biên:** Reflect padding cho vùng sát mép

**Output:** Bản đồ xác suất liên tục [0, 1] cho toàn bộ khu vực

---

## 📁 Cấu Trúc Dự Án

```
ca-mau-deforestation/
│
├── README.md                          # File này ✅
├── requirements.txt                   # Python dependencies (pip) ✅
├── environment.yml                    # Conda environment export ✅
├── DATA_METADATA_REPORT.md            # Báo cáo chi tiết metadata ✅
├── LICENSE                            # MIT License ✅
│
├── data/
│   ├── raw/                           # Dữ liệu thô ✅
│   │   ├── sentinel1/                 ✅
│   │   │   ├── S1_2024_02_04_matched_S2_2024_01_30.tif      (490 MB) ✅
│   │   │   └── S1_2025_02_22_matched_S2_2025_02_28.tif      (489 MB) ✅
│   │   ├── sentinel2/                 ✅
│   │   │   ├── S2_2024_01_30.tif                            (1.5 GB) ✅
│   │   │   └── S2_2025_02_28.tif                            (1.5 GB) ✅
│   │   └── ground_truth/              ✅
│   │       ├── Training_Points_CSV.csv       (1,285 points) ✅
│   │       └── Training_Points__SHP.*        (Shapefile)    ✅
│   │
│   └── patches/                       # Patches đã extract ⚠️ (CHƯA TẠO)
│       ├── train/                     ⚠️ TRỐNG
│       ├── val/                       ⚠️ TRỐNG
│       └── test/                      ⚠️ TRỐNG
│
├── src/                               ✅ (PYTHON MODULES - Dùng trong notebooks)
│   ├── __init__.py                   ✅ Package initialization
│   ├── utils.py                      ✅ Load data & metadata
│   ├── preprocessing.py              ✅ Normalize, NaN handling, patch extraction
│   ├── visualization.py              ✅ Plotting functions
│   ├── models.py                     ✅ 3 CNN architectures (30K-120K params)
│   ├── ml_models.py                  ✅ Random Forest model wrapper
│   ├── README.md                     ✅ Module documentation
│   └── dataset.py                    ✅ PyTorch Dataset class
│
├── docs/                              ✅ (DOCUMENTATION)
│   ├── DATA_METADATA_REPORT.md       ✅ Metadata report
│   ├── RANDOM_FOREST_GUIDE.md        ✅ RF guide & salt-pepper noise
│   └── normalization_fix.md          ✅ Normalization fix documentation
│
├── notebooks/                         ✅ (JUPYTER NOTEBOOKS)
│   ├── 00_module_usage_example.ipynb ✅ Hướng dẫn import & sử dụng modules
│   ├── 01_data_exploration.ipynb     ✅ Khám phá dữ liệu (metadata, stats, viz)
│   ├── 02_create_patches_dataset.ipynb ✅ Tạo patches dataset (128×128×14)
│   ├── 03_train_models.ipynb         ✅ Huấn luyện 3 CNN models
│   ├── 04_evaluate_and_visualize_results.ipynb ✅ Đánh giá kết quả trên test set
│   ├── 05_visualize_full_deforestation_map.ipynb ✅ Inference toàn ảnh (1 model demo)
│   ├── 06_train_random_forest.ipynb  ✅ Huấn luyện Random Forest model
│   ├── 07_compare_all_models.ipynb   ✅ So sánh tất cả 4 models (3 CNNs + RF)
│   └── README.md                     ✅ Hướng dẫn sử dụng notebooks
│
├── checkpoints/                       ✅ (ĐÃ TẠO - Chờ model weights)
│   ├── spatial_cnn_best.pth          ⬜
│   ├── multiscale_cnn_best.pth       ⬜
│   └── shallow_unet_best.pth         ⬜
│
├── outputs/                           ✅ (ĐÃ TẠO - Chờ inference)
│   ├── probability_maps/             ⬜
│   ├── binary_maps/                  ⬜
│   └── statistics/                   ⬜
│
├── logs/                              ✅ (ĐÃ TẠO - Chờ training)
│   └── training_history.csv          ⬜
│
└── figures/                           ✅ (ĐÃ TẠO - Chờ plots)
    ├── training_curves/              ⬜
    ├── confusion_matrices/           ⬜
    └── maps/                         ⬜
```

---

## 🛠️ Cài Đặt

### Yêu Cầu Hệ Thống

- **OS**: Windows 10+ / Linux / macOS
- **Python**: 3.8.20 (đã test)
- **CUDA**: 11.7 (cho GPU)
- **GPU**: NVIDIA với ≥8GB VRAM (đã test trên RTX A4000 16GB)
- **RAM**: ≥16GB (khuyến nghị 32GB)
- **Disk**: ~20GB (data + checkpoints + outputs)

### Môi Trường Đã Cài Đặt (Current Setup)

Dự án đã có môi trường conda hoàn chỉnh tên **`dang`** với các thư viện chính:

| Thư viện | Phiên bản | Mục đích |
|----------|-----------|----------|
| PyTorch | 1.13.1+cu117 | Deep learning framework |
| GDAL | 3.6.2 | Xử lý dữ liệu địa không gian |
| Rasterio | 1.3.11 | Đọc/ghi file GeoTIFF |
| NumPy | 1.24.4 | Tính toán mảng số học |
| OpenCV | 4.12.0.88 | Xử lý ảnh |
| Albumentations | 1.4.18 | Data augmentation |
| Scikit-learn | 1.3.2 | Machine learning utilities |
| MMSegmentation | 1.2.2 | Segmentation framework (optional) |
| JupyterLab | 4.2.5 | Môi trường notebook |

### Bước 1: Clone Repository (nếu chưa có)

```bash
git clone https://github.com/ninhhaidang/ca-mau-deforestation.git
cd ca-mau-deforestation
```

### Bước 2: Kích Hoạt Môi Trường

Môi trường `dang` đã được cài đặt sẵn:

```bash
conda activate dang
```

### Bước 3: (Tùy chọn) Cài Đặt Môi Trường Mới

Nếu muốn tạo môi trường mới từ đầu:

**Lựa chọn A: Từ environment.yml (Conda - Khuyến nghị)**

```bash
# Tạo môi trường mới tên 'camau-forest'
conda env create -f environment.yml -n camau-forest
conda activate camau-forest
```

**Lựa chọn B: Từ requirements.txt (pip)**

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Cài đặt PyTorch với CUDA 11.7 trước
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Cài đặt các thư viện còn lại
pip install -r requirements.txt
```

⚠️ **Lưu ý:**
- GDAL/Rasterio cài đặt qua conda dễ hơn pip (trên Windows)
- Nếu dùng pip, có thể cần cài GDAL wheel từ [Christoph Gohlke's site](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)

### Bước 4: Kiểm Tra Cài Đặt

```bash
# Kiểm tra PyTorch và CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Kiểm tra Rasterio
python -c "import rasterio; print(f'Rasterio: {rasterio.__version__}')"

# Kiểm tra GDAL
python -c "import osgeo.gdal as gdal; print(f'GDAL: {gdal.__version__}')"
```

**Kết quả mong đợi:**
```
PyTorch: 1.13.1+cu117
CUDA available: True
CUDA version: 11.7
Rasterio: 1.3.11
GDAL: 3.6.2
```

---

## 🚀 Hướng Dẫn Sử Dụng

### Bước 0: Khám Phá Dữ Liệu (Tùy chọn)

Trước khi preprocessing, khuyến nghị chạy notebook để khám phá dữ liệu:

```bash
# Kích hoạt môi trường
conda activate dang

# Khởi động JupyterLab
jupyter lab

# Mở notebook: notebooks/01_data_exploration.ipynb
# Hoặc chạy từ command line:
jupyter nbconvert --execute --to notebook notebooks/01_data_exploration.ipynb
```

**Notebook này sẽ:**
- ✅ Kiểm tra metadata của 4 ảnh TIFF
- ✅ Phân tích statistics (min, max, mean, std, NaN%)
- ✅ Visualize bands và vegetation indices
- ✅ So sánh 2024 vs 2025
- ✅ Tạo báo cáo và figures

**Outputs:**
- `data/metadata_summary.csv`
- `figures/band_nan_comparison.png`
- `figures/band_mean_comparison.png`
- `figures/indices_2024_vs_2025.png`
- `figures/sample_band_images.png`

**Thời gian:** ~2-3 phút

**Chi tiết:** Xem `notebooks/README.md`

---

### Bước 1: Chuẩn Bị Dữ Liệu

Extract patches 128×128×18 từ ảnh Sentinel.

**Lựa chọn A: Sử dụng Jupyter Notebook (Khuyến nghị)**

```bash
# Kích hoạt môi trường
conda activate dang

# Khởi động JupyterLab
jupyter lab

# Mở và chạy notebook: notebooks/02_create_patches_dataset.ipynb
```

Notebook này sẽ:
- ✅ Load 4 file TIFF (~4GB)
- ✅ Stack thành 18 channels
- ✅ Extract 1,285 patches tại các điểm ground truth
- ✅ Handle NaN values
- ✅ Normalize tất cả các bands
- ✅ Split train/val/test (70/15/15)
- ✅ Lưu thành file .npy
- ✅ Visualize sample patches

**Lựa chọn B: Sử dụng Script Python**

```bash
python -c "from src.preprocessing import create_patches_dataset; \
create_patches_dataset( \
    s1_2024_path='data/raw/sentinel1/S1_2024_02_04_matched_S2_2024_01_30.tif', \
    s1_2025_path='data/raw/sentinel1/S1_2025_02_22_matched_S2_2025_02_28.tif', \
    s2_2024_path='data/raw/sentinel2/S2_2024_01_30.tif', \
    s2_2025_path='data/raw/sentinel2/S2_2025_02_28.tif', \
    ground_truth_csv='data/raw/ground_truth/Training_Points_CSV.csv', \
    output_dir='data/patches', \
    patch_size=128, \
    train_ratio=0.70, \
    val_ratio=0.15, \
    test_ratio=0.15, \
    normalize=True, \
    handle_nan_method='fill', \
    random_seed=42)"
```

**Output:**
- ~900 training patches
- ~190 validation patches
- ~195 test patches

**Thời gian dự kiến:** 10-15 phút

---

### Bước 2: Huấn Luyện Mô Hình

```bash
# Kích hoạt môi trường
conda activate dang

# Khởi động JupyterLab
jupyter lab
```

**A. Huấn luyện 3 CNN Models:**
- Mở notebook: `notebooks/03_train_models.ipynb`
- Chạy tất cả cells (Restart & Run All)
- Notebook sẽ train cả 3 models: Spatial Context CNN, Multi-Scale CNN, Shallow U-Net
- Features: Early stopping, training curves, automatic checkpointing

**B. Huấn luyện Random Forest:**
- Mở notebook: `notebooks/06_train_random_forest.ipynb`
- Chạy tất cả cells
- Features: Feature importance analysis, confusion matrix, ROC curves

**Output:**
- `checkpoints/spatial_cnn_best.pth` - Spatial Context CNN weights
- `checkpoints/multiscale_cnn_best.pth` - Multi-Scale CNN weights
- `checkpoints/shallow_unet_best.pth` - Shallow U-Net weights
- `checkpoints/random_forest_best.pkl` - Random Forest model
- `figures/training_curves/` - Training loss/accuracy curves
- `logs/training_history.csv` - Training logs

**Thời gian dự kiến:**
- CNN models: 30-60 phút/model trên RTX A4000
- Random Forest: 2-5 phút

---

### Bước 3: Đánh Giá và So Sánh

```bash
jupyter lab
```

**A. Đánh giá CNN models trên test set:**
- Mở notebook: `notebooks/04_evaluate_and_visualize_results.ipynb`
- Chạy tất cả cells
- Features: Confusion matrices, ROC curves, sample predictions

**B. So sánh tất cả 4 models:**
- Mở notebook: `notebooks/07_compare_all_models.ipynb`
- Chạy tất cả cells
- Features: Side-by-side comparison, model agreement analysis, statistics

**Output:**
- `figures/roc_curves_all_models.png` - ROC curves comparison
- `figures/confusion_matrices/` - Confusion matrices cho từng model
- `figures/sample_predictions/` - Sample predictions
- `figures/model_agreement_analysis.png` - Agreement heatmap

**Thời gian:** 2-10 phút

---

### Bước 4: Dự Đoán Toàn Ảnh

Tạo bản đồ xác suất cho toàn tỉnh Cà Mau:

**Lựa chọn A: Sử dụng Script (Khuyến nghị cho production)**

```bash
# Tạo bản đồ full-image với model tốt nhất
python scripts/inference_full_image.py
```

**Lựa chọn B: Sử dụng Jupyter Notebook (Khuyến nghị cho exploration)**

```bash
jupyter lab
# Mở notebook: notebooks/05_visualize_full_deforestation_map.ipynb
```

**Output:**
- `figures/full_probability_map.png` - Bản đồ xác suất [0-1]
- `figures/full_binary_map.png` - Bản đồ nhị phân (threshold=0.5)
- `figures/comparison_prob_vs_binary.png` - So sánh probability vs binary
- `figures/probability_distribution.png` - Phân bố xác suất

**Thời gian:** 10-30 phút (tùy kích thước ảnh)

**Lưu ý:**
- Script sử dụng sliding window với 50% overlap
- Cần GPU ≥8GB VRAM
- Có thể chỉnh model path trong script để dùng model khác

---

## 🏗️ Kiến Trúc Mô Hình - Tóm Tắt

### Bảng So Sánh

| Tiêu Chí | Spatial Context CNN | Multi-Scale CNN | Shallow U-Net | Random Forest |
|----------|---------------------|-----------------|---------------|---------------|
| **Kiểu** | CNN | CNN | CNN | Traditional ML |
| **Số lớp** | 3 | 5 | 8-10 | N/A (100 trees) |
| **Tham số** | ~30K | ~80K | ~120K | N/A |
| **Receptive field** | 5×5 px (50m) | 7×7 px (70m) | 13×13 px (130m) | Toàn patch |
| **Độ phức tạp** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Gần ML nhất** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Thời gian train dự kiến** | ~20-30 phút | ~30-45 phút | ~45-60 phút | ~2-5 phút |
| **Thời gian inference dự kiến** | Nhanh nhất | Trung bình | Chậm nhất | Trung bình |
| **GPU cần thiết** | ✅ Có | ✅ Có | ✅ Có | ❌ Không |

### Khuyến Nghị Sử Dụng

**Spatial Context CNN:**
- ✅ Baseline đơn giản
- ✅ Tài nguyên hạn chế
- ✅ Cần kết quả nhanh

**Multi-Scale CNN:**
- ✅ **Production model (khuyến nghị)**
- ✅ Cân bằng tốt nhất
- ✅ Đa dạng kích thước mảng rừng

**Shallow U-Net:**
- ✅ Chất lượng tốt nhất
- ✅ Bản đồ xuất bản
- ✅ Có thời gian tính toán

**Random Forest:**
- ✅ Baseline để so sánh
- ✅ Không cần GPU
- ✅ Feature importance dễ diễn giải
- ✅ Huấn luyện nhanh

---

## 📊 Kết Quả

> **LƯU Ý QUAN TRỌNG:**  
> Phần này sẽ được cập nhật sau khi hoàn thành thực nghiệm. Các bảng dưới đây là **template** để điền kết quả thực tế.

### Metrics Định Lượng (Test Set, n≈195)

**Bảng 1: Độ chính xác tổng thể**

| Mô Hình | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---------|----------|-----------|--------|----------|---------|
| Random Forest | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Spatial Context CNN | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Multi-Scale CNN | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Shallow U-Net | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Bảng 2: Giảm nhiễu (Qualitative Assessment)**

| Mô Hình | Pixel Nhiễu Rời Rạc* | Điểm Độ Mượt** |
|---------|----------------------|----------------|
| Random Forest | _TBD_ | _TBD_ |
| Spatial Context CNN | _TBD_ | _TBD_ |
| Multi-Scale CNN | _TBD_ | _TBD_ |
| Shallow U-Net | _TBD_ | _TBD_ |

*Số pixel nhiễu trên 1000px²  
**Đánh giá chủ quan (1-5, 5=rất mượt)

### Hiệu Suất Tính Toán

**Bảng 3: Training Performance (RTX A4000 16GB)**

| Mô Hình | Epochs Hội Tụ | Thời Gian Training | GPU Memory |
|---------|---------------|-------------------|------------|
| Spatial Context CNN | _TBD_ | _TBD_ | _TBD_ |
| Multi-Scale CNN | _TBD_ | _TBD_ | _TBD_ |
| Shallow U-Net | _TBD_ | _TBD_ | _TBD_ |

**Bảng 4: Inference Performance**

| Mô Hình | Thời Gian (toàn ảnh) | Throughput | GPU Memory |
|---------|---------------------|-----------|------------|
| Spatial Context CNN | _TBD_ | _TBD_ | _TBD_ |
| Multi-Scale CNN | _TBD_ | _TBD_ | _TBD_ |
| Shallow U-Net | _TBD_ | _TBD_ | _TBD_ |

### Phân Tích Định Tính

_(Sẽ cập nhật sau thực nghiệm)_

- So sánh visual giữa 3 models
- Ví dụ vùng giảm nhiễu tốt
- Trường hợp khó (edges, vùng chuyển tiếp)

---

## 💬 Thảo Luận

### Đóng Góp Khoa Học

1. **Áp dụng DL cho rừng ngập mặn Việt Nam**: Nghiên cứu đầu tiên sử dụng shallow CNN cho monitoring rừng ngập mặn tại Cà Mau
2. **Giải quyết vấn đề dữ liệu hạn chế**: Chứng minh shallow networks hiệu quả với ~1.300 samples
3. **Kết hợp đa nguồn**: SAR + Optical + Multi-temporal trong một framework
4. **Practical deployment**: Models nhẹ, deploy được trên GPU thông thường

### So Sánh Với Các Nghiên Cứu Trước

| Tiêu Chí | Nghiên Cứu Này | Các Nghiên Cứu Trước |
|----------|----------------|---------------------|
| **Khu vực** | Cà Mau, Việt Nam | Chủ yếu nước ngoài |
| **Loại rừng** | Rừng ngập mặn | Đa dạng |
| **Model** | Shallow CNN (3-10 layers) | Deep networks (50+ layers) |
| **Training data** | ~1.300 samples | Thường >10.000 |
| **Độ phức tạp** | 30K-120K params | >1M params |
| **Focus** | Giảm nhiễu + chính xác | Chủ yếu chính xác |

### Hạn Chế

1. **Phạm vi thời gian**: Chỉ 2 thời điểm (2024-2025)
2. **Khu vực**: Chỉ Cà Mau, chưa test khả năng tổng quát
3. **Training data**: Point labels, chưa phải polygon
4. **Cloud cover**: Sentinel-2 bị ảnh hưởng bởi mây
5. **Validation**: Chưa có cross-regional validation

### Ý Nghĩa Thực Tiễn

**Cho Quản Lý Rừng:**
- Cung cấp bản đồ cập nhật nhanh, chi phí thấp
- Hỗ trợ phát hiện sớm biến động bất thường
- Định lượng diện tích mất rừng cho báo cáo

**Cho Nghiên Cứu:**
- Framework mở rộng cho các khu vực khác
- Benchmark cho các nghiên cứu sau
- Code mở, dễ replicate

---

## 🔮 Hướng Phát Triển

### Ngắn Hạn (3-6 tháng)

1. **Mở rộng thời gian**: Tích hợp thêm các thời điểm khác (2023, 2026)
2. **Tăng training data**: Bổ sung thêm điểm ground truth
3. **Ensemble**: Kết hợp 3 models để tăng độ tin cậy
4. **Hyperparameter tuning**: Tối ưu learning rate, batch size, augmentation

### Trung Hạn (6-12 tháng)

1. **Cross-validation**: Test trên các vùng khác (Bạc Liêu, Kiên Giang)
2. **Temporal extension**: Sử dụng time-series (LSTM/Transformer)
3. **Multi-task learning**: Phát hiện đồng thời nhiều loại biến động (cháy, chặt phá, suy thoái)
4. **Weakly supervised**: Giảm nhu cầu labeling chính xác

### Dài Hạn (1-2 năm)

1. **Operational system**: Tự động hóa toàn bộ pipeline từ download ảnh đến cảnh báo
2. **Google Earth Engine**: Scale lên toàn vùng Đồng bằng sông Cửu Long
3. **Mobile app**: Ứng dụng di động cho kiểm lâm thực địa
4. **Carbon accounting**: Kết hợp với mô hình sinh khối để ước tính CO₂

---

## 📚 Tài Liệu

Các tài liệu chi tiết về dự án được tổ chức trong thư mục `docs/`:

### 1. [Data Metadata Report](docs/DATA_METADATA_REPORT.md)
**Mô tả:** Báo cáo chi tiết về metadata của dữ liệu Sentinel-1 và Sentinel-2

**Nội dung:**
- Thông tin chi tiết về 4 file TIFF (kích thước, độ phân giải, CRS, số lượng bands)
- Phân tích NaN values trong từng band
- Thống kê reflectance và vegetation indices (min, max, mean, std)
- So sánh 2024 vs 2025

**Khi nào đọc:** Trước khi bắt đầu preprocessing hoặc khi cần hiểu rõ đặc điểm dữ liệu

---

### 2. [Random Forest Guide](docs/RANDOM_FOREST_GUIDE.md)
**Mô tả:** Hướng dẫn sử dụng và hiểu kết quả từ mô hình Random Forest

**Nội dung:**
- Giải thích về salt-and-pepper noise (nhiễu muối tiêu) và nguyên nhân
- So sánh Random Forest vs CNN về spatial context
- Phân tích feature importance (band importance, spatial importance)
- Cách cải thiện kết quả với morphological filtering
- Trade-offs giữa Random Forest và CNN

**Khi nào đọc:** Khi cần hiểu tại sao Random Forest tạo bản đồ có nhiễu hoặc cần so sánh với CNN

---

### 3. [Normalization Fix Documentation](docs/normalization_fix.md)
**Mô tả:** Tài liệu về lỗi normalization và cách sửa chữa

**Nội dung:**
- **Vấn đề:** NDVI values bị nén về 0.99-1.0 do normalization sai
- **Root cause:** Vegetation indices bị scale từ [-1,1] sang [0,1] không đúng
- **Impact:** Mất 97-99% tín hiệu phân biệt giữa mất rừng và không mất rừng
- **Solution:** Giữ nguyên natural range [-1,1] thay vì scale
- **Expected improvements:** Cải thiện class separation từ 0.01 lên 0.57 (57x)
- **Files modified:** `src/preprocessing.py`, `inference_all_models.py`, `inference_full_image.py`

**Khi nào đọc:**
- Để hiểu tại sao patches được tạo lại
- Khi viết phần Discussion trong luận văn
- Khi cần giải thích về data preprocessing trong báo cáo

**Quan trọng:** Tất cả patches đã được tạo lại với normalization đúng. Models cần được train lại để có kết quả chính xác.

---

## 📚 Tài Liệu Tham Khảo

### Nghiên Cứu Chính

1. Hansen, M. C., et al. (2013). High-Resolution Global Maps of 21st-Century Forest Cover Change. *Science*.

2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.

3. Kattenborn, T., et al. (2021). Review on CNN in Vegetation Remote Sensing. *ISPRS Journal*.

4. Pham, T. D., et al. (2019). Monitoring Mangrove Biomass Change in Vietnam using SPOT Images and an Object-Based Approach. *GIScience & Remote Sensing*.

### Về Rừng Cà Mau

5. Nguyễn Hữu Đức, et al. (2018). Đánh giá hiện trạng rừng ngập mặn tỉnh Cà Mau. *Tạp chí Khoa học Lâm nghiệp*.

6. Vũ Văn Vụ, et al. (2020). Biến động sử dụng đất rừng ngập mặn ven biển ĐBSCL. *Tạp chí Khoa học ĐHQGHN*.

### Tools và Dữ Liệu

- **Sentinel Data**: European Space Agency Copernicus Programme
- **PyTorch**: Paszke et al. (2019). *NeurIPS*.
- **Rasterio**: Geospatial raster I/O for Python

---

## 🙏 Lời Cảm Ơn

Đồ án này được hoàn thành dưới sự hướng dẫn của:

**Giảng viên hướng dẫn:** [Tên Giảng Viên]  
Viện Công nghệ Hàng không Vũ trụ  
Trường Đại học Công nghệ, ĐHQGHN

Chân thành cảm ơn:
- **ĐHQG Hà Nội - ĐH Công Nghệ** - Cung cấp tài nguyên và hỗ trợ
- **Sở Nông nghiệp và PTNT tỉnh Cà Mau** - Hỗ trợ dữ liệu thực địa (nếu có)
- **ESA Copernicus** - Dữ liệu Sentinel miễn phí
- **Cộng đồng PyTorch và GDAL** - Công cụ mã nguồn mở
- **Gia đình và bạn bè** - Động viên trong suốt quá trình nghiên cứu

---

## 📄 Giấy Phép

Dự án này được phát hành theo giấy phép MIT License. Xem file [LICENSE](LICENSE) để biết chi tiết.

### Trích Dẫn

Nếu bạn sử dụng code hoặc phương pháp này trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@thesis{ninh2025camau,
  author       = {Ninh Hải Đăng},
  title        = {Ứng Dụng Viễn Thám và Học Sâu Trong Giám Sát Biến Động Rừng Tỉnh Cà Mau},
  school       = {Trường Đại học Công nghệ, ĐHQG Hà Nội},
  year         = {2025},
  type         = {Đồ án Tốt nghiệp},
  address      = {Hà Nội, Việt Nam},
  note         = {Viện Công nghệ Hàng không Vũ trụ}
}
```

---

## 📞 Liên Hệ

**Ninh Hải Đăng**  
MSSV: 21021411  
Email: ninhhaidangg@gmail.com  
GitHub: [@ninhhaidang](https://github.com/ninhhaidang)

**Đơn vị:**  
Viện Công nghệ Hàng không Vũ trụ  
Trường Đại học Công nghệ  
Đại học Quốc gia Hà Nội  
144 Xuân Thủy, Cầu Giấy, Hà Nội

---

<p align="center">
  <sub>Xây dựng với ❤️ vì bảo vệ rừng ngập mặn Cà Mau 🌲🦀</sub>
</p>