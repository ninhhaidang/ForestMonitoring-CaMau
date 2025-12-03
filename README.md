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

**Nguồn dữ liệu ranh giới:** Công ty TNHH Tư vấn và Công nghệ Đồng Xanh — đối tác của Chi cục Kiểm lâm tỉnh Cà Mau.

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

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                                      │
│  Shape: (batch, 3, 3, 27)                                                │
│  - 3×3 spatial patch                                                     │
│  - 27 feature channels (S1+S2 fusion)                                   │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          PERMUTE                                          │
│  Output: (batch, 27, 3, 3)                                               │
│  - Convert to PyTorch format (N, C, H, W)                                │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      CONVOLUTIONAL BLOCK 1                                │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Conv2D: in=27, out=64, kernel=3×3, padding=1, bias=False          │  │
│  │ Output: (batch, 64, 3, 3)                                          │  │
│  │ Parameters: 27×3×3×64 = 15,552                                     │  │
│  └────────────────────────────────┬───────────────────────────────────┘  │
│                                   ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ BatchNorm2d(64)                                                    │  │
│  │ Parameters: 64×2 = 128                                             │  │
│  └────────────────────────────────┬───────────────────────────────────┘  │
│                                   ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ ReLU (activation)                                                  │  │
│  └────────────────────────────────┬───────────────────────────────────┘  │
│                                   ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Dropout2d(p=0.7) - High dropout for regularization                │  │
│  │ Output: (batch, 64, 3, 3)                                          │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      CONVOLUTIONAL BLOCK 2                                │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Conv2D: in=64, out=32, kernel=3×3, padding=1, bias=False          │  │
│  │ Output: (batch, 32, 3, 3)                                          │  │
│  │ Parameters: 64×3×3×32 = 18,432                                     │  │
│  └────────────────────────────────┬───────────────────────────────────┘  │
│                                   ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ BatchNorm2d(32)                                                    │  │
│  │ Parameters: 32×2 = 64                                              │  │
│  └────────────────────────────────┬───────────────────────────────────┘  │
│                                   ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ ReLU (activation)                                                  │  │
│  └────────────────────────────────┬───────────────────────────────────┘  │
│                                   ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Dropout2d(p=0.7)                                                   │  │
│  │ Output: (batch, 32, 3, 3)                                          │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    GLOBAL AVERAGE POOLING                                 │
│  AdaptiveAvgPool2d((1, 1))                                               │
│  - Reduces (batch, 32, 3, 3) → (batch, 32, 1, 1)                        │
│  - Then flatten → (batch, 32)                                            │
│  - No learnable parameters                                               │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      FULLY CONNECTED BLOCK                                │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Linear: in=32, out=64                                              │  │
│  │ Output: (batch, 64)                                                │  │
│  │ Parameters: 32×64 + 64 = 2,112                                     │  │
│  └────────────────────────────────┬───────────────────────────────────┘  │
│                                   ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ BatchNorm1d(64)                                                    │  │
│  │ Parameters: 64×2 = 128                                             │  │
│  └────────────────────────────────┬───────────────────────────────────┘  │
│                                   ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ ReLU (activation)                                                  │  │
│  └────────────────────────────────┬───────────────────────────────────┘  │
│                                   ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Dropout(p=0.7)                                                     │  │
│  │ Output: (batch, 64)                                                │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT LAYER                                     │
│  Linear: in=64, out=4                                                    │
│  Output: (batch, 4) - Logits for 4 classes                              │
│  Parameters: 64×4 + 4 = 260                                              │
│                                                                           │
│  Class mapping:                                                          │
│    0 → Rừng ổn định (Forest Stable)                                     │
│    1 → Mất rừng (Deforestation)                                          │
│    2 → Phi rừng (Non-forest)                                             │
│    3 → Phục hồi rừng (Reforestation)                                     │
└───────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                      PARAMETER SUMMARY                                    │
├──────────────────────────────────────────────────────────────────────────┤
│  Conv Block 1:     15,552 + 128 = 15,680 params                         │
│  Conv Block 2:     18,432 + 64  = 18,496 params                         │
│  FC Block:          2,112 + 128 = 2,240 params                          │
│  Output Layer:                     260 params                            │
├──────────────────────────────────────────────────────────────────────────┤
│  TOTAL TRAINABLE PARAMETERS: 36,676                                      │
└──────────────────────────────────────────────────────────────────────────┘
```

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

```mermaid
flowchart TD
    %% ===== GIAI ĐOẠN 1: THU THẬP & TIỀN XỬ LÝ DỮ LIỆU =====
    subgraph P1["1. THU THẬP & TIỀN XỬ LÝ DỮ LIỆU"]
        direction LR
        A1[("GEE")]
        A2[("GFD Ltd.")]
        A3["Ranh giới lâm nghiệp<br/>tỉnh Cà Mau"]
        A4["Lọc mây<br/>Tính các chỉ số<br/>Cắt theo ranh giới"]
        A5[/"Sentinel-1 (T₁, T₂):<br/>2 phân cực"/]
        A6[/"Sentinel-2 (T₁, T₂):<br/>4 băng tần + 3 chỉ số"/]
        A7[/"2630 điểm<br/>dữ liệu mẫu<br/>(4 lớp)"/]

        A1 --> A4
        A2 --> A3
        A3 --> A4
        A4 --> A5
        A4 --> A6
        A2 --> A7
    end

    %% ===== GIAI ĐOẠN 2: TRÍCH XUẤT ĐẶC TRƯNG =====
    subgraph P2["2. TRÍCH XUẤT ĐẶC TRƯNG"]
        direction TB
        B1["S2: Trước + Sau + Δ<br/>7 × 3 = 21 kênh"]
        B2["S1: Trước + Sau + Δ<br/>2 × 3 = 6 kênh"]
        B3["Hiệu thời gian<br/>Δ = Sau − Trước"]
        B4[("Chồng đặc trưng<br/>21 + 6 = 27 kênh")]

        B1 --> B3
        B2 --> B3
        B3 --> B4
    end

    %% ===== GIAI ĐOẠN 3: CHUẨN BỊ MẪU =====
    subgraph P3["3. CHUẨN BỊ MẪU"]
        direction TB
        C1["Chuyển tọa độ<br/>Địa lý → Pixel (hàng, cột)"]
        C2["Trích mảnh 3×3<br/>tại vị trí mẫu"]
        C3["Tính thống kê toàn cục<br/>từ Huấn luyện+Kiểm chứng 80%"]
        C4["Chuẩn hóa Z-score<br/>μ_hl, σ_hl"]
        C5[("Tập dữ liệu<br/>N × 3 × 3 × 27")]

        C1 --> C2 --> C3 --> C4 --> C5
    end

    %% ===== GIAI ĐOẠN 4: HUẤN LUYỆN MÔ HÌNH =====
    subgraph P4["4. HUẤN LUYỆN MÔ HÌNH"]
        direction TB
        D1["Phân chia phân tầng"]
        D2[/"Huấn luyện+Kiểm chứng 80%<br/>(~2.104 mẫu)"/]
        D3[/"Kiểm tra 20%<br/>(~526 mẫu)"/]
        D4["Kiểm chứng chéo 5 lần"]
        D5["Tính số epochs<br/>tối ưu"]
        D6["Huấn luyện cuối<br/>100% của 80%"]
        D7["Đánh giá<br/>trên tập kiểm tra 20%"]
        D8[("Mô hình CNN<br/>.pth")]

        D1 --> D2 & D3
        D2 --> D4
        D4 --> D5
        D5 --> D6
        D6 --> D7
        D3 --> D7
        D7 --> D8
    end

    %% ===== GIAI ĐOẠN 5: DỰ ĐOÁN TOÀN VÙNG =====
    subgraph P5["5. DỰ ĐOÁN TOÀN VÙNG"]
        direction TB
        E1["Dự đoán theo lô"]
        E2[("Bản đồ phân loại<br/>4 lớp GeoTIFF")]

        E1 --> E2
    end

    %% ===== LUỒNG CHÍNH - NỐI CÁC NODE GIỮA CÁC PHASE =====
    A5 --> B2
    A6 --> B1
    A7 --> C1
    B4 --> C2
    C5 --> D1
    D8 --> E1
    B4 --> E1
    C5 --> E1

    %% ===== ĐỊNH DẠNG =====
    classDef phase1 fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    classDef phase2 fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef phase3 fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px
    classDef phase4 fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px
    classDef phase5 fill:#FCE4EC,stroke:#C2185B,stroke-width:2px

    class P1 phase1
    class P2 phase2
    class P3 phase3
    class P4 phase4
    class P5 phase5
```

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
