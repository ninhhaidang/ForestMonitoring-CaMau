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

## 🗂️ Cấu Trúc Thư Mục (Đơn Giản Hóa)

```
25-26_HKI_DATN_21021411_DangNH/
│
├── � notebooks/                         # ⭐ Interactive analysis (Jupyter)
│   ├── 01_explore_data.ipynb            # Visualize S2 + S1 data
│   ├── 02_analyze_ground_truth.ipynb    # Phân tích 1285 điểm thực địa
│   ├── 03_training_workflow.ipynb       # Monitor training process
│   ├── 04_evaluation.ipynb              # Analyze metrics & results
│   └── 05_final_maps.ipynb              # Tạo bản đồ change detection
│
├── 📁 configs/                           # Training configurations
│   ├── snunet_camau_s2s1.py             # Config Phase 2: S2+S1 (18 channels)
│   └── snunet_camau_v3.py               # Config mới nhất
│
├── 📁 data/                              # ⭐ Dữ liệu chính
│   │
│   ├── 📁 ground_truth/                 # Ground truth points
│   │   ├── Training_Points__SHP.shp     # 1285 điểm (shapefile)
│   │   └── Training_Points_CSV.csv      # 1285 điểm (CSV format)
│   │
│   ├── 📁 sentinel2/                    # Dữ liệu quang học
│   │   ├── 📁 raw/                      # GeoTIFF gốc
│   │   └── 📁 processed/                # Sau xử lý
│   │
│   ├── 📁 sentinel1/                    # Dữ liệu SAR
│   │   ├── 📁 raw/                      # SAR gốc
│   │   └── 📁 processed/                # Sau xử lý
│   │
│   └── 📁 labels/                       # ⭐ Training samples
│       ├── 📁 train/                    # 80% (~1028 samples)
│       ├── 📁 val/                      # 10% (~128 samples)
│       └── 📁 test/                     # 10% (~129 samples)
│
├── 📁 open-cd/                           # Open-CD framework (cloned)
│   ├── configs/                         # Config templates
│   ├── opencd/                          # Source code
│   └── tools/
│       ├── train.py                     # ⭐ Training script
│       └── test.py                      # ⭐ Testing script
│
├── 📁 outputs/                           # ⭐ All results (gộp results + work_dirs)
│   │
│   ├── 📁 checkpoints/                  # Model weights
│   │   └── best_model.pth
│   │
│   ├── 📁 logs/                         # Training logs
│   │   ├── train.log
│   │   └── tensorboard/
│   │
│   ├── 📁 metrics/                      # Performance metrics
│   │   ├── test_metrics.json
│   │   └── confusion_matrix.csv
│   │
│   └── 📁 visualizations/               # Hình ảnh & bản đồ
│       ├── 📁 maps/                     # Change detection maps
│       ├── 📁 figures/                  # Charts & plots
│       └── 📁 comparisons/              # Model comparisons
│
├── 📁 scripts/                           # Automation scripts
│   ├── 01_verify_data.py                # Verify data quality
│   ├── 05_create_samples_from_points.py # ⭐ Tạo training samples
│   ├── 06_visualize_samples.py          # Visualize samples
│   ├── compute_normalization_stats.py   # Compute stats
│   ├── test_setup.py                    # Test environment
│   └── verify_environment.py            # Verify setup
│
├── 📁 work_dirs/                         # Training runs (auto-generated)
│   ├── snunet_camau/
│   ├── snunet_camau_s2s1/
│   └── ablation_studies/
│
├── � environment.yml                    # Conda environment
├── 📄 requirements.txt                   # Pip requirements
├── 📄 README.md                          # ⭐ File này
└── 📄 LICENSE                            # MIT License
```

### � Đơn Giản Hóa Chính:
- ✅ **Gộp results → outputs/** (checkpoints, logs, metrics, visualizations)
- ✅ **5 notebooks chính** thay vì nhiều notebooks rời rạc
- ✅ **Giảm số lượng scripts** (từ 10 → 6 scripts cốt lõi)
- ✅ **Cấu trúc rõ ràng hơn**, dễ navigate hơn

---

## 🔄 Workflow - Từ Đầu Đến Cuối

```
1️⃣ 📊 Explore Data (Notebook 01)
   ├─ Visualize Sentinel-2 (7 bands + indices)
   ├─ Visualize Sentinel-1 (SAR data)
   └─ Check data quality
   ↓

2️⃣ 📍 Analyze Ground Truth (Notebook 02)
   ├─ Load 1285 điểm thực địa
   ├─ Check class balance (635 loss / 650 no change)
   └─ Spatial distribution analysis
   ↓

3️⃣ 🔨 Prepare Training Samples
   [scripts/05_create_samples_from_points.py]
   ├─ Extract patches around ground truth points
   ├─ Split: 80% train, 10% val, 10% test
   └─ Save to data/labels/
   ↓

4️⃣ 🎯 Training (Notebook 03)
   [open-cd/tools/train.py + configs/snunet_camau_s2s1.py]
   ├─ Train SNUNet-CD model
   ├─ Monitor with TensorBoard
   └─ Save checkpoints → outputs/checkpoints/
   ↓

5️⃣ 📈 Evaluation (Notebook 04)
   [open-cd/tools/test.py]
   ├─ Test on test set
   ├─ Calculate metrics (Accuracy, F1, IoU, Precision, Recall)
   ├─ Confusion matrix
   └─ Save results → outputs/metrics/
   ↓

6️⃣ 🗺️ Create Final Maps (Notebook 05)
   [Inference on full Ca Mau province]
   ├─ Run inference on entire area
   ├─ Generate change detection map
   ├─ Calculate deforestation statistics
   └─ Export → outputs/visualizations/maps/
   ↓

7️⃣ 📄 Report & Presentation
   └─ Compile results for thesis
```

### 🎯 Notebook Workflow:
1. **`01_explore_data.ipynb`** → Khám phá dữ liệu Sentinel
2. **`02_analyze_ground_truth.ipynb`** → Phân tích 1285 điểm
3. **`03_training_workflow.ipynb`** → Monitor training
4. **`04_evaluation.ipynb`** → Đánh giá model
5. **`05_final_maps.ipynb`** → Tạo bản đồ cuối cùng

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