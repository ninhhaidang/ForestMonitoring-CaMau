# Ứng dụng Viễn thám và Học sâu trong Giám sát Biến động Rừng tỉnh Cà Mau

**Đồ án tốt nghiệp - Công nghệ Hàng không Vũ trụ**

Sinh viên: **Ninh Hải Đăng** (MSSV: 21021411)

Cán bộ hướng dẫn: **TS. Hà Minh Cường**, **ThS. Hoàng Tích Phúc**

Năm học: 2025 - 2026, Học kỳ I

---

## Tổng quan

Dự án phát triển hệ thống tự động giám sát biến động rừng tại tỉnh Cà Mau (theo địa giới hành chính mới sau khi sáp nhập với tỉnh Bạc Liêu từ 01/07/2025) sử dụng kết hợp dữ liệu viễn thám đa nguồn:
- **Sentinel-1** (SAR - Synthetic Aperture Radar): Hoạt động mọi điều kiện thời tiết
- **Sentinel-2** (Optical Multispectral): Thông tin quang phổ phong phú

**Phương pháp:** Convolutional Neural Network (CNN) với kiến trúc lightweight (~36,676 tham số), sử dụng patches 3×3 pixels để khai thác ngữ cảnh không gian. Dữ liệu được tiền xử lý trên GEE (lọc mây, chuẩn hóa, tính chỉ số thực vật).

**Kết quả:** Đạt độ chính xác **98.86%** trên tập test với ROC-AUC **99.98%**.

---

## Khu vực nghiên cứu

**Lưu ý:** Theo Nghị quyết số 1278/NQ-UBTVQH15 ngày 24/10/2024 của Ủy ban Thường vụ Quốc hội, kể từ ngày 01/07/2025, tỉnh Cà Mau và tỉnh Bạc Liêu được sáp nhập thành tỉnh Cà Mau mới.

| Thông số | Giá trị |
|----------|---------|
| **Vị trí** | Cực Nam Việt Nam, vùng Đồng bằng sông Cửu Long |
| **Tọa độ địa lý** | 8°36'–9°40' Bắc, 104°43'–105°50' Đông |
| **Diện tích tự nhiên** | 7,942.38 km² |
| **Dân số** | ~2.6 triệu người |
| **Diện tích ranh giới quy hoạch lâm nghiệp** | 170,178.82 ha (1,701.79 km²) |
| **Diện tích thực tế được phân loại** | 162,469.25 ha (~95.5% ranh giới) |
| **Hệ quy chiếu** | EPSG:32648 (WGS 84 / UTM Zone 48N) |
| **Độ phân giải không gian** | 10m |

**Nguồn dữ liệu ranh giới:** Công ty TNHH Tư vấn và Phát triển Đồng Xanh — đối tác của Chi cục Kiểm lâm tỉnh Cà Mau.

---

## Dữ liệu

### Ground Truth
- **File:** `data/raw/samples/4labels.csv`
- **Tổng số điểm:** 2,630 điểm
- **Phân bố labels (4 classes):**

| Class | Tên | Số điểm | Tỷ lệ |
|-------|-----|---------|-------|
| 0 | Rừng ổn định (Forest Stable) | 656 | 24.9% |
| 1 | Mất rừng (Deforestation) | 650 | 24.7% |
| 2 | Phi rừng (Non-forest) | 664 | 25.3% |
| 3 | Phục hồi rừng (Reforestation) | 660 | 25.1% |

### Sentinel-2 (Optical)
- **7 bands:** B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2), NDVI, NBR, NDMI
- **Độ phân giải:** 10m
- **Kỳ ảnh:**
  - Trước: 30/01/2024
  - Sau: 28/02/2025

### Sentinel-1 (SAR)
- **2 bands:** VV, VH polarization
- **Độ phân giải:** 10m (co-registered với Sentinel-2)
- **Kỳ ảnh:**
  - Trước: 04/02/2024
  - Sau: 22/02/2025

### Feature Stack (27 features)
```
Sentinel-2 (21 features):
  - Before[0:7]:  B4, B8, B11, B12, NDVI, NBR, NDMI
  - After[0:7]:   B4, B8, B11, B12, NDVI, NBR, NDMI
  - Delta[0:7]:   ΔB4, ΔB8, ΔB11, ΔB12, ΔNDVI, ΔNBR, ΔNDMI

Sentinel-1 (6 features):
  - Before[0:2]:  VV, VH
  - After[0:2]:   VV, VH
  - Delta[0:2]:   ΔVV, ΔVH
```

---

## Kiến trúc CNN

![Kiến trúc CNN](THESIS/img/chapter3/CNN-architecture.png)

### Tóm tắt kiến trúc

| Layer | Operation | Input → Output | Parameters |
|-------|-----------|----------------|------------|
| **Input** | Permute | (B, 3, 3, 27) → (B, 27, 3, 3) | 0 |
| **Conv Block 1** | Conv2d + BN + ReLU + Dropout(0.7) | (B, 27, 3, 3) → (B, 64, 3, 3) | 15,680 |
| **Conv Block 2** | Conv2d + BN + ReLU + Dropout(0.7) | (B, 64, 3, 3) → (B, 32, 3, 3) | 18,496 |
| **Global Avg Pool** | AdaptiveAvgPool2d(1,1) + Flatten | (B, 32, 3, 3) → (B, 32) | 0 |
| **FC Block** | Linear + BN + ReLU + Dropout(0.7) | (B, 32) → (B, 64) | 2,240 |
| **Output** | Linear | (B, 64) → (B, 4) | 260 |
| | | **Total** | **36,676** |

### Đặc điểm thiết kế

| Thành phần | Chi tiết | Lý do thiết kế |
|------------|----------|----------------|
| **Patch size** | 3×3 pixels (30m × 30m) | Cân bằng ngữ cảnh không gian và tính toán |
| **Dropout rate** | 0.7 (70%) | Regularization mạnh cho dataset nhỏ (2,630 mẫu) |
| **BatchNorm** | Sau mỗi Conv và FC | Ổn định training, tăng tốc hội tụ |
| **Activation** | ReLU | Tránh vanishing gradient, tính toán nhanh |
| **Global Pooling** | AdaptiveAvgPool2d | Giảm tham số, translation invariance |
| **Weight Init** | Kaiming (Conv) + Xavier (FC) | Phù hợp với ReLU activation |
| **No pooling layers** | Không dùng MaxPool | Giữ nguyên spatial info do patch nhỏ (3×3) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=0.001, weight_decay=1e-3) |
| Loss | CrossEntropyLoss |
| Batch size | 64 |
| Max epochs | 200 |
| Early stopping | patience=15 |
| LR Scheduler | ReduceLROnPlateau (patience=10) |
| Dropout rate | 0.7 |
| Data split | Stratified 80% Train+Val / 20% Test |
| Cross-validation | 5-Fold Stratified CV |

---

## Kết quả

### Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 98.86% |
| **ROC-AUC** | 99.98% |
| **CV Accuracy** | 98.15% ± 0.28% |

### Kết quả phân loại toàn vùng (162,469 ha)

| Class | Diện tích (ha) | Tỷ lệ |
|-------|---------------|-------|
| Rừng ổn định | 120,716.91 | 74.30% |
| Mất rừng | 7,282.15 | 4.48% |
| Phi rừng | 29,528.54 | 18.17% |
| Phục hồi rừng | 4,941.90 | 3.04% |

---

## Quy trình xử lý dữ liệu (Methodology Flowchart)

![Quy trình xử lý dữ liệu](THESIS/img/chapter3/flowchart.png)

### Chi tiết phương pháp nghiên cứu

| Giai đoạn | Tên | Đầu vào | Đầu ra | Phương pháp |
|:---------:|-----|---------|--------|-------------|
| **1** | Thu thập & Tiền xử lý | Vùng nghiên cứu, Thời gian | S2 (7 kênh) + S1 (2 kênh) + Mẫu | GEE: Lọc mây, chia 10000, tính chỉ số |
| **2** | Trích xuất đặc trưng | S2(T₁,T₂), S1(T₁,T₂) | Chồng 27 kênh | Trước, Sau, Hiệu |
| **3** | Chuẩn bị mẫu | Chồng 27, Điểm 2630 | Mảnh (N,3,3,27) | Trích 3×3, Chuẩn hóa-Z |
| **4** | Huấn luyện | Mảnh, Nhãn | CNN .pth | Chia 80/20 → KC-chéo 5 → Cuối |
| **5** | Dự đoán | Raster đầy đủ, Mô hình | GeoTIFF 4 lớp | Cửa sổ trượt, Theo lô |

### Cấu trúc 27 Features

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FEATURE STACK (27 bands)                      │
├─────────────────────────────────────────────────────────────────────┤
│  SENTINEL-2 (21 features)                                            │
│  ├── Before [0-6]:  B4, B8, B11, B12, NDVI, NBR, NDMI               │
│  ├── After  [7-13]: B4, B8, B11, B12, NDVI, NBR, NDMI               │
│  └── Delta [14-20]: ΔB4, ΔB8, ΔB11, ΔB12, ΔNDVI, ΔNBR, ΔNDMI        │
├─────────────────────────────────────────────────────────────────────┤
│  SENTINEL-1 (6 features)                                             │
│  ├── Before [21-22]: VV, VH                                          │
│  ├── After  [23-24]: VV, VH                                          │
│  └── Delta  [25-26]: ΔVV, ΔVH                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Cấu trúc dự án

```
25-26_HKI_DATN_21021411_DangNH/
├── README.md
├── environment.yml
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── sentinel-1/          # Ảnh SAR (VV, VH)
│   │   ├── sentinel-2/          # Ảnh Optical (7 bands)
│   │   ├── boundary/            # Ranh giới quy hoạch lâm nghiệp
│   │   └── samples/             # Ground truth points
│   └── inference/
│
├── src/
│   ├── config.py                # Cấu hình tập trung
│   ├── main_cnn.py              # Entry point cho CNN pipeline
│   ├── utils.py
│   │
│   ├── core/
│   │   ├── data_loader.py
│   │   ├── feature_extraction.py
│   │   ├── evaluation.py
│   │   └── visualization.py
│   │
│   └── models/
│       └── cnn/
│           ├── architecture.py
│           ├── trainer.py
│           ├── patch_extractor.py
│           └── predictor.py
│
├── notebook/
│   └── cnn_deforestation_detection.ipynb
│
├── results/
│   ├── models/                  # Trained models (.pth)
│   ├── data/                    # Output data files
│   ├── rasters/                 # GeoTIFF classification maps
│   └── plots/                   # Visualization outputs
│
└── THESIS/
    └── Latex/                   # LaTeX thesis files
```

---

## Dependencies chính

- `torch` >= 2.0 - Deep learning framework
- `scikit-learn` - Machine learning utilities
- `rasterio` - GeoTIFF I/O
- `geopandas` - Geospatial data
- `numpy`, `pandas` - Data manipulation
- `matplotlib`, `seaborn` - Visualization

**Full dependencies:** Xem `environment.yml`

---

## Liên hệ

- **Sinh viên:** Ninh Hải Đăng
- **Email:** ninhhaidangg@gmail.com
- **GitHub:** [ninhhaidang](https://github.com/ninhhaidang)
- **Đơn vị:** Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội

---

## License

Dự án này được phát triển cho mục đích nghiên cứu và học thuật.

---

**Last updated:** November 2025