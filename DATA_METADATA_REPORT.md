# 📊 BÁO CÁO METADATA DỮ LIỆU SENTINEL

**Ngày kiểm tra:** 2025-10-22
**Khu vực:** Tỉnh Cà Mau, Việt Nam
**Hệ tọa độ:** EPSG:32648 (WGS 84 / UTM zone 48N)

---

## 🎯 TÓM TẮT TỔNG QUAN

### Dữ liệu có sẵn

| Loại | Thời điểm | File | Bands | Kích thước | Dung lượng |
|------|-----------|------|-------|-----------|-----------|
| **Sentinel-1** | 2024-02-04 | S1_2024_02_04_matched_S2_2024_01_30.tif | 2 | 7970 × 11261 | 490 MB |
| **Sentinel-1** | 2025-02-22 | S1_2025_02_22_matched_S2_2025_02_28.tif | 2 | 7970 × 11261 | 489 MB |
| **Sentinel-2** | 2024-01-30 | S2_2024_01_30.tif | 7 | 7970 × 11261 | 1.5 GB |
| **Sentinel-2** | 2025-02-28 | S2_2025_02_28.tif | 7 | 7970 × 11261 | 1.5 GB |

**Tổng dung lượng:** ~4 GB
**Tổng số bands:** 2×2 (S1) + 7×2 (S2) = **18 bands**

---

## 🗺️ THÔNG TIN ĐỊA LÝ

### Hệ tọa độ & Độ phân giải
- **CRS:** EPSG:32648 (WGS 84 / UTM zone 48N)
- **Pixel size:** 10m × 10m
- **Extent (UTM):**
  - X: 467,390m → 547,090m (79.7 km)
  - Y: 945,340m → 1,057,950m (112.61 km)
- **Diện tích:** ~8,976 km²

### Kích thước ảnh
- **Width:** 7,970 pixels
- **Height:** 11,261 pixels
- **Total pixels/band:** 89,750,170 pixels

---

## 📡 SENTINEL-1 (SAR) - 2 BANDS

### Band 1: VH (Vertical-Horizontal Polarization)

**Đặc tả:**
- **Loại:** SAR Backscatter (dB)
- **Data type:** float32
- **Range:** -54.9 dB → +9.1 dB
- **Mean:** -18.8 dB (2024), -19.0 dB (2025)
- **Std:** ~4.3 dB
- **NoData:** None (không có NaN)

**Ý nghĩa:**
- Phản hồi radar phân cực chéo
- Nhạy cảm với cấu trúc thực vật (canopy)
- Giá trị thấp: nước/bề mặt phẳng
- Giá trị cao: rừng/cấu trúc phức tạp

---

### Band 2: R (VV - VH Ratio)

**Đặc tả:**
- **Loại:** Ratio/Difference giữa 2 phân cực
- **Data type:** float32
- **Range:** -22.1 → +40.5 (2024), -16.1 → +40.4 (2025)
- **Mean:** +7.2 dB (cả 2 năm)
- **Std:** ~3.7 dB
- **NoData:** None

**Ý nghĩa:**
- Chỉ số phân biệt loại bề mặt
- Giá trị cao: nước/đất trống
- Giá trị thấp: rừng/thực vật dày

---

## 🛰️ SENTINEL-2 (OPTICAL) - 7 BANDS

### Band 1: B4 (Red - 665nm)

**Đặc tả:**
- **Loại:** Surface reflectance
- **Data type:** float32
- **Range:** 0.0 → 1.58 (2024), 0.0 → 1.36 (2025)
- **Mean:** 0.055 (2024), 0.050 (2025)
- **NaN:** ~2.8% (2024), ~0.5% (2025)

**Ý nghĩa:**
- Phản xạ bề mặt vùng đỏ
- Hấp thụ mạnh bởi chlorophyll
- Dùng tính NDVI

---

### Band 2: B8 (NIR - 842nm)

**Đặc tả:**
- **Range:** 0.0 → 1.43 (2024), 0.0 → 1.02 (2025)
- **Mean:** 0.118 (2024), 0.101 (2025)
- **NaN:** ~2.8% (2024), ~0.5% (2025)

**Ý nghĩa:**
- Phản xạ cận hồng ngoại
- Phản xạ mạnh từ thực vật khỏe
- Dùng tính NDVI, NDMI

---

### Band 3: B11 (SWIR1 - 1610nm)

**Đặc tả:**
- **Range:** 0.006 → 0.786 (2024), 0.0 → 0.781 (2025)
- **Mean:** 0.080 (2024), 0.084 (2025)
- **NaN:** ~2.8% (2024), ~0.5% (2025)

**Ý nghĩa:**
- Phản xạ SWIR1
- Nhạy cảm với độ ẩm thực vật
- Dùng tính NDMI

---

### Band 4: B12 (SWIR2 - 2190nm)

**Đặc tả:**
- **Range:** 0.005 → 0.826 (2024), 0.0 → 0.850 (2025)
- **Mean:** 0.052 (2024), 0.055 (2025)
- **NaN:** ~2.8% (2024), ~0.5% (2025)

**Ý nghĩa:**
- Phản xạ SWIR2
- Nhạy cảm với độ ẩm đất và thực vật
- Dùng tính NBR (burn index)

---

### Band 5: NDVI (Normalized Difference Vegetation Index)

**Đặc tả:**
- **Formula:** (NIR - Red) / (NIR + Red) = (B8 - B4) / (B8 + B4)
- **Range:** -1.0 → +0.95 (2024), -1.0 → +1.0 (2025)
- **Mean:** 0.224 (2024), 0.121 (2025)
- **NaN:** ~2.8% (2024), ~0.5% (2025)

**Ý nghĩa:**
- Chỉ số độ xanh thực vật
- **< 0:** Nước, đất trống
- **0-0.2:** Đất trống, thực vật thưa
- **0.2-0.4:** Cỏ, bụi
- **0.4-0.6:** Cây trồng, rừng thưa
- **> 0.6:** Rừng dày đặc

**Phân tích:**
- Mean giảm từ 0.224 (2024) → 0.121 (2025) ⚠️
- Có thể chỉ ra sự suy giảm thực vật

---

### Band 6: NBR (Normalized Burn Ratio)

**Đặc tả:**
- **Formula:** (NIR - SWIR2) / (NIR + SWIR2) = (B8 - B12) / (B8 + B12)
- **Range:** -1.0 → +0.89 (cả 2 năm)
- **Mean:** 0.312 (2024), 0.106 (2025)
- **NaN:** ~2.8% (2024), ~0.5% (2025)

**Ý nghĩa:**
- Chỉ số phát hiện khu vực cháy/mất rừng
- **> 0.4:** Thực vật khỏe
- **0.1-0.4:** Thực vật trung bình
- **< 0.1:** Đất trống, cháy, mất rừng

**Phân tích:**
- Mean giảm mạnh 0.312 → 0.106 ⚠️⚠️
- Dấu hiệu mất rừng hoặc suy thoái

---

### Band 7: NDMI (Normalized Difference Moisture Index)

**Đặc tả:**
- **Formula:** (NIR - SWIR1) / (NIR + SWIR1) = (B8 - B11) / (B8 + B11)
- **Range:** -1.0 → +0.81 (2024), -1.0 → +0.80 (2025)
- **Mean:** 0.116 (2024), -0.079 (2025)
- **NaN:** ~2.8% (2024), ~0.5% (2025)

**Ý nghĩa:**
- Chỉ số độ ẩm thực vật
- **> 0.4:** Độ ẩm cao (rừng, vùng ngập)
- **0-0.4:** Độ ẩm trung bình
- **< 0:** Khô hạn, stress

**Phân tích:**
- Mean giảm từ +0.116 → -0.079 (chuyển sang âm!) ⚠️⚠️
- Có thể chỉ ra stress thủy văn hoặc mất rừng

---

## 🔍 PHÂN TÍCH SƠ BỘ

### ✅ Ưu điểm

1. **Độ phân giải cao:** 10m × 10m
2. **Dữ liệu đa nguồn:** SAR + Optical
3. **Đa thời gian:** 2 thời điểm cách nhau ~1 năm
4. **Coverage tốt:** NaN thấp (0.5-2.8%)
5. **Đồng nhất:** Cùng CRS, resolution, extent

### ⚠️ Phát hiện quan trọng

**Các chỉ số thực vật giảm đáng kể:**

| Chỉ số | 2024 | 2025 | Thay đổi |
|--------|------|------|----------|
| NDVI | 0.224 | 0.121 | **-46%** ⚠️⚠️ |
| NBR | 0.312 | 0.106 | **-66%** ⚠️⚠️⚠️ |
| NDMI | +0.116 | -0.079 | **Âm** ⚠️⚠️ |

**Giải thích có thể:**
1. **Mất rừng thực sự:** Chuyển đổi rừng → ao nuôi/đất trống
2. **Suy thoái rừng:** Cháy, chặt phá, bệnh hại
3. **Biến động mùa:** Khác biệt thời điểm thu thập (01/30 vs 02/28)
4. **Điều kiện thời tiết:** Hạn hán, ngập mặn

→ **Cần phân tích chi tiết để xác định nguyên nhân**

### 📝 Lưu ý khi xử lý

1. **NaN values:**
   - S2_2024: ~2.8% (cloud/shadow)
   - S2_2025: ~0.5% (tốt hơn)
   - Cần xử lý NaN khi extract patches

2. **Data range:**
   - S1: dB scale (âm) → cần normalize
   - S2 reflectance: [0, ~1.5] → có thể normalize hoặc clip
   - S2 indices: [-1, +1] → đã normalized

3. **Memory:**
   - Toàn ảnh 18 bands: ~6.8 GB RAM
   - Patches 128×128×18: ~1.2 MB/patch
   - Batch size 16: ~19 MB

---

## 📋 CẤU TRÚC STACK 18 KÊNH

Khi ghép 4 files TIFF, ta có stack 18 kênh theo thứ tự:

### Thời điểm 1 (2024) - 9 kênh:
1. S1_VH_2024 (SAR backscatter)
2. S1_R_2024 (SAR ratio)
3. S2_B4_2024 (Red)
4. S2_B8_2024 (NIR)
5. S2_B11_2024 (SWIR1)
6. S2_B12_2024 (SWIR2)
7. S2_NDVI_2024
8. S2_NBR_2024
9. S2_NDMI_2024

### Thời điểm 2 (2025) - 9 kênh:
10. S1_VH_2025
11. S1_R_2025
12. S2_B4_2025
13. S2_B8_2025
14. S2_B11_2025
15. S2_B12_2025
16. S2_NDVI_2025
17. S2_NBR_2025
18. S2_NDMI_2025

**Preprocessing cần thiết:**
- Đọc 4 files TIFF
- Stack thành array 18 channels
- Xử lý NaN (interpolate hoặc mask)
- Normalize/standardize
- Extract patches 128×128×18 tại vị trí ground truth

---

## 📊 THỐNG KÊ CHI TIẾT

### Sentinel-1 Statistics

| Band | File | Min | Max | Mean | Std | NaN% |
|------|------|-----|-----|------|-----|------|
| VH | S1_2024 | -54.92 | +9.13 | -18.80 | 4.31 | 0% |
| R | S1_2024 | -22.13 | +40.47 | +7.18 | 3.68 | 0% |
| VH | S1_2025 | -51.56 | +7.29 | -18.98 | 4.38 | 0% |
| R | S1_2025 | -16.07 | +40.42 | +7.29 | 3.75 | 0% |

### Sentinel-2 Statistics

| Band | File | Min | Max | Mean | Std | NaN% |
|------|------|-----|-----|------|-----|------|
| B4 | S2_2024 | 0.005 | 1.585 | 0.055 | 0.030 | 2.84% |
| B8 | S2_2024 | 0.000 | 1.426 | 0.118 | 0.082 | 2.84% |
| B11 | S2_2024 | 0.006 | 0.786 | 0.080 | 0.054 | 2.84% |
| B12 | S2_2024 | 0.005 | 0.826 | 0.052 | 0.043 | 2.84% |
| NDVI | S2_2024 | -1.000 | 0.954 | 0.224 | 0.398 | 2.84% |
| NBR | S2_2024 | -1.000 | 0.888 | 0.312 | 0.312 | 2.84% |
| NDMI | S2_2024 | -1.000 | 0.810 | 0.116 | 0.293 | 2.84% |
| B4 | S2_2025 | 0.000 | 1.362 | 0.050 | 0.038 | 0.54% |
| B8 | S2_2025 | 0.000 | 1.023 | 0.101 | 0.088 | 0.54% |
| B11 | S2_2025 | 0.000 | 0.781 | 0.084 | 0.061 | 0.54% |
| B12 | S2_2025 | 0.000 | 0.850 | 0.055 | 0.049 | 0.54% |
| NDVI | S2_2025 | -1.000 | 1.000 | 0.121 | 0.538 | 0.54% |
| NBR | S2_2025 | -1.000 | 0.895 | 0.106 | 0.481 | 0.54% |
| NDMI | S2_2025 | -1.000 | 0.803 | -0.079 | 0.420 | 0.54% |

---

## 🚀 KHUYẾN NGHỊ

### 1. Preprocessing Pipeline
```python
# Pseudocode
1. Load 4 TIFF files
2. Stack into 18-channel array (7970 × 11261 × 18)
3. Handle NaN:
   - Option A: Interpolate từ neighbors
   - Option B: Mask (đánh dấu NaN pixels)
   - Option C: Loại bỏ patches có >10% NaN
4. Normalize:
   - S1: (x - mean) / std
   - S2 reflectance: clip [0, 1] hoặc standardize
   - S2 indices: đã trong [-1, 1]
5. Extract patches 128×128×18 tại ground truth coords
6. Save patches as .npy files
```

### 2. Data Augmentation
- ✅ Rotation (90°, 180°, 270°)
- ✅ Horizontal/Vertical flip
- ❌ Color jitter (không phù hợp với spectral data)
- ❌ Elastic transform (giữ nguyên cấu trúc địa lý)

### 3. Normalization Strategy
**Đề xuất A (Per-band Standardization):**
```python
# Tính mean/std từ toàn bộ training set
for band in range(18):
    mean_b = train_data[:, :, band].mean()
    std_b = train_data[:, :, band].std()
    data[:, :, band] = (data[:, :, band] - mean_b) / std_b
```

**Đề xuất B (Mixed Normalization):**
```python
# S1 bands: Standardize
s1_bands = [0, 1, 9, 10]  # VH, R cho cả 2 năm
for i in s1_bands:
    data[:, :, i] = (data[:, :, i] - mean[i]) / std[i]

# S2 reflectance: Clip + scale to [0, 1]
ref_bands = [2, 3, 4, 5, 11, 12, 13, 14]
for i in ref_bands:
    data[:, :, i] = np.clip(data[:, :, i], 0, 1)

# S2 indices: Đã normalized [-1, 1], giữ nguyên hoặc scale to [0, 1]
index_bands = [6, 7, 8, 15, 16, 17]
for i in index_bands:
    data[:, :, i] = (data[:, :, i] + 1) / 2  # [-1,1] → [0,1]
```

---

## ✅ KẾT LUẬN

### Trạng thái dữ liệu: **SẴN SÀNG**

✅ **Có đầy đủ:**
- 4 files TIFF với metadata hợp lệ
- 18 bands đa nguồn, đa thời gian
- Ground truth 1,285 points

✅ **Chất lượng tốt:**
- Coverage cao (NaN < 3%)
- Độ phân giải phù hợp (10m)
- Extent đủ lớn (~9,000 km²)

⚠️ **Cần lưu ý:**
- Xử lý NaN khi extract patches
- Chọn chiến lược normalization phù hợp
- Phân tích sự suy giảm chỉ số thực vật

### Bước tiếp theo:
1. ✅ Viết `src/prepare_data.py` - Extract patches
2. ⬜ Viết `src/models.py` - 3 CNN architectures
3. ⬜ Viết `src/train.py` - Training pipeline

---

**Generated:** 2025-10-22
**Tool:** Python 3.8.20 + Rasterio 1.3.11
**Author:** Claude Code Assistant
