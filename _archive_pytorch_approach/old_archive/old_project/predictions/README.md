# Prediction Visualizations

Bản đồ phân loại thay đổi rừng Ca Mau (2017-2023) từ mô hình SNUNet-CD.

---

## 📁 Thư mục

### `snunet/` (20 files - Basic Visualizations)

Bản đồ cơ bản với 3 panel riêng biệt:

**Files:**
- `*_comparison.png` - So sánh 3 panel: Time1 | Time2 | Change Map
- `*_change_map.png` - Bản đồ phân loại đơn thuần

**Đặc điểm:**
- Time 1 & 2: False color composite (SWIR-Red-NIR)
- Change map: Màu phẳng (xanh=không đổi, đỏ=thay đổi)
- Dễ nhìn, rõ ràng

---

### `snunet_overlay/` (30 files - Advanced Overlays) ⭐

Bản đồ nâng cao với overlay màu lên ảnh vệ tinh thực tế:

#### 1. Side-by-side (10 files: `*_sidebyside.png`)
**Format:** 2 panels
- Panel 1: Ảnh Time 1 (Before) - Nguyên gốc
- Panel 2: Ảnh Time 2 (After) + Overlay đỏ cho vùng thay đổi

**Ưu điểm:**
- Thấy rõ vùng thay đổi ngay trên ảnh vệ tinh
- Transparency α=0.5 giúp vẫn nhìn thấy cả ảnh gốc và vùng đổi
- Phù hợp cho báo cáo/trình bày

#### 2. Triple View (10 files: `*_triple.png`)
**Format:** 3 panels
- Panel 1: Time 1 (Before)
- Panel 2: Time 2 (After) - Không overlay
- Panel 3: Time 2 (After) + Overlay

**Ưu điểm:**
- So sánh trực quan: trước → sau → sau+overlay
- Thấy rõ hiệu quả của overlay
- Phù hợp cho phân tích chi tiết

#### 3. Change Highlight (10 files: `*_highlight.png`)
**Format:** 4 panels
- Panel 1: Time 1 (Before)
- Panel 2: Time 2 (After)
- Panel 3: Change Highlight (Vùng không đổi → grayscale, vùng đổi → color + red tint)
- Panel 4: Change Classification (Bản đồ phân loại)

**Ưu điểm:**
- Phân tích toàn diện nhất
- Change Highlight làm nổi bật vùng thay đổi bằng màu sắc
- Vùng không đổi mờ đi (grayscale) để tập trung vào vùng đổi
- Phù hợp cho phân tích khoa học

---

## 🎨 Color Scheme

### Change Detection Colors
- **Xanh lá (34, 139, 34)**: Không thay đổi - Rừng còn nguyên
- **Đỏ (220, 20, 60)**: Có thay đổi - Mất rừng/Phá rừng

### False Color Composite
- **Red channel**: SWIR1 (B11) - Nhạy cảm với độ ẩm
- **Green channel**: Red (B4) - Thảm thực vật
- **Blue channel**: NIR (B8) - Sinh khối

---

## 📊 Statistics

- **Total samples**: 10 (từ 129 test samples)
- **Total visualizations**: 50 files (20 basic + 30 overlay)
- **Resolution**: 150 DPI
- **Format**: PNG (RGB)
- **Image size**: 256×256 pixels (original data)

---

## 🔧 Scripts sử dụng

1. **`visualize_predictions.py`** - Tạo bản đồ cơ bản
   ```bash
   python visualize_predictions.py
   ```

2. **`create_overlay_maps.py`** - Tạo overlay nâng cao
   ```bash
   python create_overlay_maps.py
   ```

---

## 💡 Cách sử dụng

### Cho báo cáo/luận văn:
- Dùng `*_sidebyside.png` hoặc `*_triple.png`
- Rõ ràng, dễ hiểu cho người đọc

### Cho phân tích khoa học:
- Dùng `*_highlight.png`
- Đầy đủ thông tin, phân tích chi tiết

### Cho trình bày:
- Dùng `*_comparison.png` (đơn giản)
- Hoặc `*_sidebyside.png` (nổi bật hơn)

---

## 📈 Model Performance

**SNUNet-CD Test Results:**
- mIoU: 79.50%
- F1-Score: 88.56%
- Precision: 88.86%
- Recall: 88.39%

Chi tiết: Xem `../SNUNET_RESULTS.md`

---

## 🗺️ Samples

Các mẫu được visualize:
```
0001, 0002, 0003, 0004, 0005, 0006, 0007, 0008, 0009, 0010
```

**Vị trí:** Ca Mau, Vietnam
**Thời gian:** 2017 (before) → 2023 (after)
**Dữ liệu:** Sentinel-2 (optical) + Sentinel-1 (SAR)

---

**Generated:** 2025-10-18
**Model:** SNUNet-CD
**Framework:** Open-CD
