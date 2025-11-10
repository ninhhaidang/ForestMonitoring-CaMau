# Ứng dụng Viễn thám và Học sâu trong Giám sát Biến động Rừng tỉnh Cà Mau

**Đồ án tốt nghiệp - Công nghệ Hàng không Vũ trụ**

Sinh viên: **Ninh Hải Đăng** (MSSV: 21021411)
Năm học: 2025 - 2026, Học kỳ I

---

## 📋 Tổng quan

Dự án này phát triển một hệ thống tự động giám sát biến động rừng tại tỉnh Cà Mau sử dụng kết hợp dữ liệu viễn thám đa nguồn (Sentinel-1 SAR và Sentinel-2 Optical) với hai phương pháp tiếp cận: Machine Learning truyền thống (Random Forest) và Deep Learning (CNN). Hệ thống có khả năng phát hiện và phân loại các khu vực mất rừng dựa trên phân tích chuỗi thời gian ảnh vệ tinh, với độ chính xác > 98%.

---

## 📊 Dữ liệu

### Ground Truth Points
- **Tổng số điểm:** 1,285 điểm training
- **Phân bố:**
  - Label 0 (Không mất rừng): 650 điểm (50.6%)
  - Label 1 (Mất rừng): 635 điểm (49.4%)
- **Format:** CSV file với các trường: `id`, `label`, `x`, `y` (tọa độ UTM Zone 48N)
- **File:** `data/raw/ground_truth/Training_Points_CSV.csv`

### Sentinel-2 (Optical)
- **7 bands** gồm spectral bands và spectral indices:
  - **Spectral bands:** B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2)
  - **Spectral indices:** NDVI, NBR, NDMI
- **Độ phân giải không gian:** 10m
- **Kỳ ảnh:**
  - Trước: 30/01/2024 (`S2_2024_01_30.tif`)
  - Sau: 28/02/2025 (`S2_2025_02_28.tif`)
- **Đã xử lý:** Cắt theo ranh giới rừng tỉnh Cà Mau, masked NoData

### Sentinel-1 (SAR)
- **2 bands:** VV và VH polarization
- **Độ phân giải không gian:** 10m (matched với Sentinel-2)
- **Kỳ ảnh:**
  - Trước: 04/02/2024 (`S1_2024_02_04_matched_S2_2024_01_30.tif`)
  - Sau: 22/02/2025 (`S1_2025_02_22_matched_S2_2025_02_28.tif`)
- **Đã xử lý:** Co-registered với Sentinel-2, cắt theo ranh giới rừng

### Boundary Shapefile
- **File:** `data/raw/boundary/forest_boundary.shp`
- **Mục đích:** Giới hạn khu vực phân tích chỉ trong vùng rừng

---

## 📦 Output Files

Sau khi chạy xong, kết quả được lưu trong folder `results/`:

**Random Forest Outputs:**
```
results/
├── rasters/
│   ├── rf_classification.tif               # Binary classification map (0/1)
│   └── rf_probability.tif                  # Probability map (0.0-1.0)
├── vectors/
│   └── rf_deforestation_polygons.geojson   # Deforestation polygons
├── models/
│   └── rf_model.pkl                        # Trained Random Forest (277 KB)
├── data/
│   ├── training_data.csv                   # Training features (1,285 samples)
│   ├── rf_feature_importance.csv           # Feature importance rankings
│   └── rf_evaluation_metrics.json          # Performance metrics
└── plots/
    ├── rf_confusion_matrices.png           # Confusion matrices
    ├── rf_roc_curve.png                    # ROC curve
    ├── rf_feature_importance.png           # Top 20 features
    ├── rf_classification_maps.png          # Binary & probability maps
    └── rf_cv_scores.png                    # 5-fold CV scores
```

**CNN Outputs:**
```
results/
├── rasters/
│   ├── cnn_classification.tif              # Binary classification map
│   └── cnn_probability.tif                 # Probability map
├── models/
│   └── cnn_model.pth                       # Trained CNN (448 KB)
├── data/
│   ├── cnn_training_patches.npz            # Saved patches data
│   ├── cnn_evaluation_metrics.json         # Performance metrics
│   └── cnn_training_history.json           # Training curves (loss, acc)
└── plots/
    ├── cnn_confusion_matrices.png          # Confusion matrices
    ├── cnn_roc_curve.png                   # ROC curve
    ├── cnn_training_curves.png             # Loss & accuracy curves
    └── cnn_classification_maps.png         # Binary & probability maps
```

---

---

## 📧 Liên hệ

- **Sinh viên:** Ninh Hải Đăng
- **Email:** ninhhaidangg@gmail.com
- **GitHub:** [ninhhaidang](https://github.com/ninhhaidang)
- **Đơn vị:** Trường Đại học Công nghệ - ĐHQGHN

---

## 📄 License

Dự án này được phát triển cho mục đích nghiên cứu và giáo dục.

---

**Cập nhật lần cuối:** 08/01/2025
**Version:** 3.0 (Complete implementation: RF + CNN + Comparison)
**Status:** Production-ready
