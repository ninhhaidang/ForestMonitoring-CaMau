# 🌲 Phát Hiện Mất Rừng Cà Mau Sử Dụng Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8.20-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1+cu117-red.svg)](https://pytorch.org/)
[![Open-CD](https://img.shields.io/badge/Framework-Open--CD-green.svg)](https://github.com/likyoo/open-cd)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Giới Thiệu

Dự án tốt nghiệp sử dụng Deep Learning (SNUNet-CD) để phát hiện mất rừng ngập mặn tại tỉnh Cà Mau, Việt Nam với chu kỳ giám sát 2 lần/tháng.

### 🎯 Mục Tiêu
- Phát hiện tự động các khu vực mất rừng ngập mặn tại Cà Mau (7,942.39 km²)
- Sử dụng dữ liệu đa nguồn: Sentinel-2 (optical) + Sentinel-1 (SAR)
- Chu kỳ giám sát: 2 lần/tháng (đầu-giữa tháng và giữa-cuối tháng)

### 📊 Dữ Liệu
- **Sentinel-2 (MSI):** 7 bands (B4, B8, B11, B12) + 3 indices (NDVI, NBR, NDMI)
- **Sentinel-1 (SAR):** VH polarization + Ratio (VV-VH)
- **Thời điểm:** T1 (30/01/2024) → T2 (28/02/2025)
- **Training data:** 635 điểm mất rừng + 650 điểm không mất rừng (1285 điểm ground truth)

### 🧠 Model
- **Architecture:** SNUNet-CD (Siamese Nested U-Net)
- **Framework:** Open-CD 1.1.0 (OpenMMLab ecosystem)
- **Input:** 14-18 channels (Phase 1: S2 only, Phase 2: S2+S1)
- **Output:** Binary change detection map

---

## 🗂️ Cấu Trúc Thư Mục

```
25-26_HKI_DATN_21021411_DangNH/
│
├── 📁 configs/                           # Training configurations
│   ├── snunet_camau_s2only.py           # Config Phase 1: S2 only (14 channels)
│   ├── snunet_camau_s2s1.py             # Config Phase 2: S2+S1 (18 channels)
│   └── snunet_baseline.py               # Baseline comparison
│
├── 📁 data/                              # ⭐ Dữ liệu chính của dự án
│   │
│   ├── 📁 ground_truth/                 # Ground truth points (thực địa)
│   │   ├── training_points.shp          # 1285 điểm label (shapefile/geojson/csv)
│   │   │                                # - 635 điểm mất rừng (label=1)
│   │   │                                # - 650 điểm không mất rừng (label=0)
│   │   └── README.txt                   # Mô tả cấu trúc dữ liệu
│   │
│   ├── 📁 sentinel2/                    # Dữ liệu vệ tinh quang học
│   │   ├── 📁 raw/                      # File GeoTIFF gốc (chưa xử lý)
│   │   │   ├── S2_2024_01_30.tif       # T1: 7 bands (B4,B8,B11,B12,NDVI,NBR,NDMI)
│   │   │   └── S2_2025_02_28.tif       # T2: 7 bands
│   │   │
│   │   └── 📁 processed/                # Sau xử lý (clipped, normalized, cloud masked)
│   │       ├── S2_T1_processed.tif
│   │       └── S2_T2_processed.tif
│   │
│   ├── 📁 sentinel1/                    # Dữ liệu SAR (tất cả thời tiết)
│   │   ├── 📁 raw/                      # File SAR gốc (GRD format)
│   │   │   ├── S1_2024_01_30.tif       # T1: 2 bands (VH, R=VV-VH)
│   │   │   └── S1_2025_02_28.tif       # T2: 2 bands (VH, R=VV-VH)
│   │   │
│   │   └── 📁 processed/                # Sau calibration & filtering
│   │       ├── S1_T1_calibrated.tif    # Radiometric calibrated, speckle filtered
│   │       └── S1_T2_calibrated.tif
│   │
│   ├── 📁 labels/                       # ⭐ Training/validation/test samples
│   │   │                                # Được tạo từ ground_truth + sentinel data
│   │   ├── 📁 train/                    # 80% data (~1028 samples)
│   │   │   ├── sample_0001_img.tif     # Multi-channel image (14 or 18 bands)
│   │   │   ├── sample_0001_mask.tif    # Binary mask (0=no change, 1=forest loss)
│   │   │   ├── sample_0002_img.tif
│   │   │   ├── sample_0002_mask.tif
│   │   │   └── ...
│   │   │
│   │   ├── 📁 val/                      # 10% data (~128 samples)
│   │   │   └── (tương tự train/)
│   │   │
│   │   └── 📁 test/                     # 10% data (~129 samples)
│   │       └── (tương tự train/)
│   │
│   └── 📁 augmented/                    # Dữ liệu tăng cường (optional)
│       └── (augmented samples nếu cần thêm data)
│
├── 📁 notebooks/                         # Jupyter notebooks
│   ├── 01_explore_s2_data.ipynb         # Khám phá dữ liệu Sentinel-2
│   ├── 02_explore_s1_data.ipynb         # Khám phá dữ liệu Sentinel-1
│   ├── 03_visualize_training_points.ipynb
│   ├── 04_data_statistics.ipynb         # Thống kê dataset
│   ├── 05_model_demo.ipynb              # Demo model inference
│   └── 06_create_final_maps.ipynb       # Tạo bản đồ kết quả cuối
│
├── 📁 open-cd/                           # Open-CD framework (cloned)
│   ├── configs/                         # Config templates của Open-CD
│   │   ├── snunet/                      # SNUNet-CD configs
│   │   ├── changeformer/                # Các model khác
│   │   └── _base_/                      # Base configurations
│   │
│   ├── opencd/                          # Source code
│   │   ├── models/                      # Model architectures
│   │   ├── datasets/                    # Dataset loaders
│   │   ├── evaluation/                  # Evaluation metrics
│   │   └── ...
│   │
│   └── tools/
│       ├── train.py                     # ⭐ Script training chính
│       ├── test.py                      # ⭐ Script testing chính
│       └── ...
│
├── 📁 results/                           # ⭐ Tất cả outputs của dự án
│   │
│   ├── 📁 visualizations/               # Hình ảnh, bản đồ, biểu đồ
│   │   ├── 📁 maps/                     # Bản đồ change detection
│   │   │   ├── camau_forest_loss_2024_2025.tif
│   │   │   ├── camau_forest_loss_rgb.png
│   │   │   └── camau_deforestation_overlay.png
│   │   │
│   │   ├── 📁 figures/                  # Biểu đồ, curves
│   │   │   ├── training_curves.png     # Loss & accuracy curves
│   │   │   ├── confusion_matrix.png
│   │   │   ├── roc_curve.png
│   │   │   └── sample_predictions.png
│   │   │
│   │   └── 📁 comparisons/              # So sánh models
│   │       ├── before_after_comparison.png
│   │       ├── s2_vs_s2s1_comparison.png
│   │       └── model_comparison.png
│   │
│   ├── 📁 metrics/                      # Metrics và statistics
│   │   ├── test_metrics.json            # Overall Accuracy, F1, IoU, etc.
│   │   ├── confusion_matrix.csv
│   │   ├── per_class_metrics.csv
│   │   └── deforestation_statistics.xlsx
│   │
│   ├── 📁 models/                       # Model weights cuối cùng
│   │   ├── snunet_camau_best.pth       # Best model checkpoint
│   │   └── model_info.json             # Model metadata
│   │
│   └── 📁 reports/                      # Báo cáo và presentation
│       ├── final_report.pdf             # Báo cáo tốt nghiệp
│       ├── presentation.pptx            # Slide thuyết trình
│       └── technical_report.md          # Chi tiết kỹ thuật
│
├── 📁 scripts/                           # Python scripts automation
│   ├── 01_verify_s2_data.py            # Verify Sentinel-2 data
│   ├── 02_download_s1_data.py          # Download Sentinel-1
│   ├── 03_preprocess_s2.py             # Preprocess S2
│   ├── 04_preprocess_s1.py             # Preprocess S1
│   ├── 05_create_samples_from_points.py # ⭐ Tạo train/val/test từ ground truth
│   ├── 06_merge_s2_s1.py               # Merge S2+S1 → 18 channels
│   ├── 07_train.py                     # Wrapper cho training
│   ├── 08_test.py                      # Wrapper cho testing
│   ├── 09_inference.py                 # Inference toàn tỉnh Cà Mau
│   └── 10_calculate_metrics.py         # Tính toán metrics
│
├── 📁 work_dirs/                         # Training outputs (auto-generated)
│   │
│   ├── 📁 snunet_camau/                 # Main experiment
│   │   ├── 20251013_100000/            # Timestamp của mỗi training run
│   │   │   ├── checkpoints/
│   │   │   │   ├── epoch_10.pth
│   │   │   │   ├── epoch_20.pth
│   │   │   │   ├── ...
│   │   │   │   └── best_model.pth      # Best checkpoint
│   │   │   │
│   │   │   ├── logs/
│   │   │   │   ├── train.log
│   │   │   │   └── val.log
│   │   │   │
│   │   │   └── tensorboard/
│   │   │       └── events.out.tfevents.* # TensorBoard logs
│   │   │
│   │   └── 20251014_143000/             # Another training run
│   │       └── ...
│   │
│   └── 📁 ablation_studies/             # Ablation experiments
│       ├── 📁 s2_only/                  # Training chỉ dùng S2 (14 channels)
│       └── 📁 s1_only/                  # Training chỉ dùng S1 (4 channels)
│
├── 📄 .gitignore                         # Git ignore rules
├── 📄 conda_packages.txt                 # Conda packages list
├── 📄 environment.yml                    # Conda environment config
├── 📄 LICENSE                            # MIT License
├── 📄 PROJECT_REPORT.md                  # Báo cáo chi tiết dự án
├── 📄 README.md                          # File này
└── 📄 requirements.txt                   # Pip requirements
```

---

## 🔄 Workflow - Từ Đầu Đến Cuối

```
1️⃣ Ground Truth Points
   data/ground_truth/training_points.shp (1285 điểm)
   ↓

2️⃣ Tạo Training Samples
   [scripts/05_create_samples_from_points.py]
   ↓
   data/labels/train, val, test (patches 128x128 hoặc 256x256)
   ↓

3️⃣ Training Model
   [open-cd/tools/train.py + configs/snunet_camau.py]
   ↓
   work_dirs/snunet_camau/checkpoints/best_model.pth
   ↓

4️⃣ Testing & Evaluation
   [open-cd/tools/test.py]
   ↓
   results/metrics/test_metrics.json
   ↓

5️⃣ Inference Toàn Tỉnh
   [scripts/09_inference.py]
   ↓
   results/visualizations/maps/camau_forest_loss.tif
   ↓

6️⃣ Báo Cáo & Presentation
   results/reports/final_report.pdf
```

---

## 🚀 Quick Start

### 1. Cài Đặt Môi Trường

```bash
# Clone repository
git clone <repo-url>
cd 25-26_HKI_DATN_21021411_DangNH

# Tạo môi trường conda
conda env create -f environment.yml
conda activate dang

# Cài đặt Open-CD framework
cd open-cd
pip install -v -e .
cd ..

# Verify môi trường
python scripts/verify_environment.py

# Kết quả mong đợi:
# ✅ Python: 3.8.20
# ✅ PyTorch: 1.13.1+cu117
# ✅ CUDA available: True
# ✅ GPU: NVIDIA RTX A4000
# ✅ Open-CD: 1.1.0
```

### 2. Chuẩn Bị Dữ Liệu

```bash
# Verify Sentinel-2 data
python scripts/01_verify_s2_data.py

# Download Sentinel-1 (nếu chưa có)
python scripts/02_download_s1_data.py

# Preprocess data
python scripts/03_preprocess_s2.py
python scripts/04_preprocess_s1.py

# Tạo training samples từ ground truth points
python scripts/05_create_samples_from_points.py
```

### 3. Training Model

#### Phase 1: Sentinel-2 Only (14 channels)

```bash
# Training
python open-cd/tools/train.py configs/snunet_camau_s2only.py

# Testing
python open-cd/tools/test.py configs/snunet_camau_s2only.py \
    work_dirs/snunet_camau/latest.pth
```

#### Phase 2: Sentinel-2 + Sentinel-1 (18 channels)

```bash
# Merge S2 + S1 data
python scripts/06_merge_s2_s1.py

# Training với 18 channels
python open-cd/tools/train.py configs/snunet_camau_s2s1.py

# Testing
python open-cd/tools/test.py configs/snunet_camau_s2s1.py \
    work_dirs/snunet_camau/latest.pth
```

### 4. Inference & Visualization

```bash
# Inference trên toàn tỉnh Cà Mau
python scripts/09_inference.py

# Tạo bản đồ và biểu đồ
jupyter notebook notebooks/06_create_final_maps.ipynb
```

---

## 📊 Feature Set

### Phase 1: Sentinel-2 Only (14 channels)

| # | Feature | Thời điểm | Mô tả |
|---|---------|-----------|-------|
| 1 | b_B4 | Before (T1) | Red band |
| 2 | b_B8 | Before | Near-Infrared |
| 3 | b_B11 | Before | SWIR 1 |
| 4 | b_B12 | Before | SWIR 2 |
| 5 | b_NDVI | Before | Vegetation index |
| 6 | b_NBR | Before | Normalized Burn Ratio |
| 7 | b_NDMI | Before | Moisture index |
| 8 | a_B4 | After (T2) | Red band |
| 9 | a_B8 | After | Near-Infrared |
| 10 | a_B11 | After | SWIR 1 |
| 11 | a_B12 | After | SWIR 2 |
| 12 | a_NDVI | After | Vegetation index |
| 13 | a_NBR | After | Normalized Burn Ratio |
| 14 | a_NDMI | After | Moisture index |

### Phase 2: Sentinel-2 + Sentinel-1 (18 channels)

**Thêm 4 channels từ Sentinel-1:**

| # | Feature | Thời điểm | Mô tả |
|---|---------|-----------|-------|
| 15 | b_VH | Before (T1) | VH polarization (dB) |
| 16 | b_R | Before | Ratio: VV - VH (dB) |
| 17 | a_VH | After (T2) | VH polarization (dB) |
| 18 | a_R | After | Ratio: VV - VH (dB) |

---

## 💻 Hệ Thống & Môi Trường

### Phần cứng
- **CPU:** Intel Xeon E5-2678 v3
- **RAM:** 32GB DDR3 ECC
- **GPU:** NVIDIA RTX A4000
  - VRAM: 16GB
  - CUDA Cores: 6144
  - CUDA Version: 11.7
- **Storage:** 4TB HDD

### Phần mềm
- **OS:** Windows
- **Python:** 3.8.20
- **PyTorch:** 1.13.1+cu117
- **CUDA:** 11.7
- **NumPy:** 1.24.4
- **Framework:** Open-CD 1.1.0 (MMSegmentation-based)

### OpenMMLab Ecosystem
- **mmengine:** 0.10.4
- **mmcv:** 2.1.0
- **mmdet:** 3.3.0
- **mmseg:** 1.2.2
- **mmpretrain:** 1.2.0

### Data Processing Libraries
- **OpenCV:** 4.12.0
- **Rasterio:** 1.3.11
- **GDAL:** 3.9.2
- **Albumentations:** 1.4.18

### Development Tools
- **Jupyter Notebook:** 7.2.2
- **JupyterLab:** 4.2.5

### Môi trường Conda
```bash
# Activate environment
conda activate dang

# Verify packages
conda list

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📈 Dataset Statistics

### Ground Truth Points

| Category | Số lượng | Tỷ lệ |
|----------|----------|-------|
| **Mất rừng (label=1)** | 635 | 49.4% |
| **Không mất rừng (label=0)** | 650 | 50.6% |
| **Tổng cộng** | **1285** | **100%** |

✅ **Dataset balanced:** Tỷ lệ gần như 1:1 rất lý tưởng cho binary classification!

### Training/Validation/Test Split

| Split | Số lượng | Tỷ lệ | Mô tả |
|-------|----------|-------|-------|
| **Train** | ~1028 | 80% | Dùng để training model |
| **Validation** | ~128 | 10% | Dùng để tune hyperparameters |
| **Test** | ~129 | 10% | Dùng để đánh giá cuối cùng |
| **Tổng** | **1285** | **100%** | Ground truth points |

---

## 📈 Kết Quả Dự Kiến

### Metrics
- **Overall Accuracy:** > 90%
- **F1-Score (Forest Loss):** > 0.85
- **IoU (Intersection over Union):** > 0.75
- **Precision:** > 0.88
- **Recall:** > 0.82

### Outputs
- ✅ Bản đồ change detection toàn tỉnh Cà Mau
- ✅ Diện tích mất rừng theo từng khu vực
- ✅ Thống kê biến động rừng ngập mặn
- ✅ So sánh hiệu quả S2 vs S2+S1
- ✅ Báo cáo kỹ thuật chi tiết

---

## 📝 To-Do List

### Đã hoàn thành ✅
- [x] Setup môi trường (PyTorch, CUDA, Open-CD)
- [x] Thiết kế cấu trúc thư mục dự án
- [x] Xác định feature set (14-18 channels)
- [x] Thu thập dữ liệu Sentinel-2 (2 thời điểm)
- [x] Thu thập ground truth points (1285 điểm)
- [x] Verify môi trường làm việc

### Đang thực hiện 🔄
- [ ] Verify và organize dữ liệu S2
- [ ] Download dữ liệu Sentinel-1
- [ ] Tạo training samples từ 1285 ground truth points
- [ ] Viết config files cho SNUNet-CD

### Kế hoạch tiếp theo 📋
- [ ] Training Phase 1 (S2 only - 14 channels)
- [ ] Đánh giá kết quả Phase 1
- [ ] Bổ sung S1 data (Phase 2: S2+S1 - 18 channels)
- [ ] Training Phase 2
- [ ] So sánh hiệu quả S2 vs S2+S1
- [ ] Inference trên toàn tỉnh Cà Mau
- [ ] Viết báo cáo tốt nghiệp và presentation

---

## 🔍 Thông Tin Thêm

### Kích thước dữ liệu dự kiến
- **data/:** ~15-20 GB
- **work_dirs/:** ~3-5 GB
- **results/:** ~2-3 GB
- **Total:** ~25-30 GB

### Git Large Files
Do file dữ liệu quá lớn, các file sau đã được thêm vào `.gitignore`:
- Tất cả file `.tif`, `.tiff` trong `data/`
- Checkpoints `.pth` trong `work_dirs/`
- Large visualizations trong `results/`

Sử dụng Git LFS nếu cần version control cho files lớn.

---

## 📚 Tài Liệu Tham Khảo

### Papers
- [SNUNet-CD: A Densely Connected Siamese Network for Change Detection](https://ieeexplore.ieee.org/document/9355573)
- [Open-CD: A Comprehensive Toolbox for Change Detection](https://github.com/likyoo/open-cd)

### Data Sources
- [Sentinel-2 User Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi)
- [Sentinel-1 User Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar)
- [Copernicus Data Space](https://dataspace.copernicus.eu/)

### Frameworks & Libraries
- [Open-CD Documentation](https://github.com/likyoo/open-cd)
- [MMSegmentation Documentation](https://mmsegmentation.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Rasterio Documentation](https://rasterio.readthedocs.io/)
- [GDAL Documentation](https://gdal.org/)

---

## 📄 License

MIT License - xem file [LICENSE](LICENSE)

---

## 👤 Tác Giả

**Ninh Hải Đăng**  
MSSV: 21021411  
Khóa: 2021-2025  
Đồ Án Tốt Nghiệp - Học kỳ I 2025-2026  
Viện Công nghệ Hàng không Vũ trụ  
Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội

---

## 📧 Liên Hệ

Nếu có câu hỏi hoặc góp ý về dự án, vui lòng liên hệ qua:
- 📧 Email: ninhhaidangg@gmail.com
- 💻 GitHub: [@ninhhaidang](https://github.com/ninhhaidang)

---

*Cập nhật lần cuối: 13/10/2025*