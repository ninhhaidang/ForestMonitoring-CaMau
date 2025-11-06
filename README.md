# Ứng dụng Viễn thám và Học sâu trong Giám sát Biến động Rừng tỉnh Cà Mau

**Đồ án tốt nghiệp - Công nghệ Hàng không Vũ trụ**

Sinh viên: **Ninh Hải Đăng** (MSSV: 21021411)
Năm học: 2025 - 2026, Học kỳ I

---

## 📋 Tổng quan

Dự án này phát triển một hệ thống tự động giám sát biến động rừng tại tỉnh Cà Mau sử dụng kết hợp dữ liệu viễn thám đa nguồn (Sentinel-1 SAR và Sentinel-2 Optical) và mô hình học sâu (Deep Learning). Hệ thống có khả năng phát hiện và phân loại các khu vực mất rừng dựa trên phân tích chuỗi thời gian ảnh vệ tinh.

### Mục tiêu

- Phát triển mô hình deep learning để phát hiện mất rừng từ ảnh vệ tinh đa thời gian
- Kết hợp dữ liệu SAR (Sentinel-1) và Optical (Sentinel-2) để nâng cao độ chính xác
- Tạo bản đồ phân loại toàn bộ khu vực rừng tỉnh Cà Mau

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

## 📁 Cấu trúc thư mục

```
25-26_HKI_DATN_21021411_DangNH/
│
├── data/                           # Thư mục chứa dữ liệu
│   ├── raw/                        # Dữ liệu gốc
│   │   ├── ground_truth/           # Ground truth CSV
│   │   ├── sentinel-1/             # Ảnh Sentinel-1 SAR
│   │   ├── sentinel-2/             # Ảnh Sentinel-2 Optical
│   │   └── boundary/               # Shapefile ranh giới rừng
│   ├── processed/                  # Dữ liệu đã xử lý
│   └── patches/                    # Patches đã trích xuất
│
├── src/                            # Source code
│   ├── config.py                   # Cấu hình chung
│   ├── utils.py                    # Hàm tiện ích
│   ├── preprocessing.py            # Tiền xử lý dữ liệu
│   ├── dataset.py                  # PyTorch Dataset (nếu có)
│   └── (các module khác sẽ được thêm)
│
├── notebooks/                      # Jupyter notebooks
│   └── 01_data_exploration.ipynb   # Khám phá dữ liệu
│
├── models/                         # Thư mục lưu trained models
├── figures/                        # Visualizations và plots
├── logs/                           # Training logs
│
├── environment.yml                 # Conda environment
├── requirements.txt                # Python dependencies
└── README.md                       # File này

```

---

## 💻 Yêu cầu hệ thống

### Phần cứng sử dụng
- **CPU:** Intel Xeon X5670 (hoặc tương đương)
- **RAM:** 64GB DDR3
- **GPU:** NVIDIA GTX 1060 6GB hoặc cao hơn (hỗ trợ CUDA)
- **Storage:** ≥50GB dung lượng trống

### Phần mềm
- **OS:** Windows 10/11, Linux, macOS
- **Python:** 3.8 - 3.11
- **CUDA:** 11.8+ (nếu sử dụng GPU)
- **Conda/Miniconda:** Phiên bản mới nhất

---

## ⚙️ Cài đặt

### Bước 1: Clone repository

```bash
git clone https://github.com/Geospatial-Technology-Lab/25-26_HKI_DATN_21021411_DangNH.git
cd 25-26_HKI_DATN_21021411_DangNH
```

### Bước 2: Tạo Conda environment

```bash
conda env create -f environment.yml
conda activate dang
```

**Hoặc** sử dụng pip:

```bash
pip install -r requirements.txt
```

### Bước 3: Verify installation

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🚀 Sử dụng

### 1. Khám phá dữ liệu (Data Exploration)

Chạy notebook để khám phá và visualize dữ liệu:

```bash
cd notebooks
jupyter notebook 01_data_exploration.ipynb
```

**Notebook này sẽ:**
- Load và phân tích ground truth points
- Visualize Sentinel-1 và Sentinel-2 imagery
- Kiểm tra value ranges và data quality
- Trích xuất và hiển thị sample patches
- Tạo các visualizations trong folder `figures/`

**Outputs:**
- Các visualizations sẽ được lưu trong folder `figures/`
- Bao gồm: band comparisons, ground truth visualization, sample patches, etc.

### 2. Tiền xử lý dữ liệu (Data Preprocessing)

Trích xuất patches từ toàn bộ ground truth points:

```bash
python -c "from src.preprocessing import create_patches_dataset; create_patches_dataset(patch_size=64)"
```

**Output:**
- `data/patches/patches_64x64.pkl` - File chứa patches và labels

### 3. Training mô hình

> **Status:** Script training và pipeline chưa được hoàn thiện. Sẽ được develop sau khi xác định kiến trúc model.

### 4. Inference (Dự đoán toàn bộ khu vực)

> **Status:** Script inference sẽ được develop sau khi hoàn thành training và chọn được best model.

---

## 🧠 Input Data Structure

### Input Specification
- **18 channels** từ 2 kỳ ảnh:
  - **Kỳ 2024:** 7 bands S2 + 2 bands S1 = 9 channels
  - **Kỳ 2025:** 7 bands S2 + 2 bands S1 = 9 channels
- **Patch size:** 64×64 pixels
- **Channel order:**
  ```
  [0-6]:   S2 2024 (B4, B8, B11, B12, NDVI, NBR, NDMI)
  [7-8]:   S1 2024 (VV, VH)
  [9-15]:  S2 2025 (B4, B8, B11, B12, NDVI, NBR, NDMI)
  [16-17]: S1 2025 (VV, VH)
  ```

### Model Architecture

> **Status:** Kiến trúc mô hình deep learning chưa được xác định. Sẽ thử nghiệm và lựa chọn sau.

---

## ⚙️ Training Configuration

#### Đã xác định:
- **Mixed Precision (AMP):** Enabled - Tiết kiệm ~40% VRAM, tăng tốc training
- **Batch size:** 16-24 (tùy model, được test để tận dụng tối đa 6GB VRAM với AMP)
- **Gradient Accumulation:** 2 steps (Effective batch size = 32-48 tùy batch size thực tế)
- **Data split:** 70% train, 15% validation, 15% test
- **DataLoader Strategy:** Cache toàn bộ 1,285 patches trong RAM (~380MB) để tối ưu tốc độ

#### Chưa xác định (sẽ thử nghiệm):
- **Optimizer:** TBD (Adam, AdamW, SGD, etc.)
- **Learning rate:** TBD
- **Learning rate scheduler:** TBD (CosineAnnealing, ReduceLROnPlateau, etc.)
- **Loss function:** TBD (CrossEntropyLoss, Focal Loss, etc.)
- **Epochs:** TBD
- **Early stopping patience:** TBD
- **Data augmentation:** TBD (Rotation, Flip, Noise, etc.)

---

## 📈 Kết quả

> **Status:** Đang trong quá trình thử nghiệm và training models.

### Metrics

Các metrics đánh giá sẽ bao gồm:
- **Accuracy:** Độ chính xác tổng thể
- **Precision:** Độ chính xác của class "Mất rừng"
- **Recall:** Khả năng phát hiện mất rừng
- **F1-Score:** Trung bình điều hòa của Precision và Recall
- **Confusion Matrix:** Ma trận nhầm lẫn
- **ROC-AUC:** Diện tích dưới đường cong ROC

### Kết quả so sánh models

(Sẽ được cập nhật sau khi hoàn thành training và evaluation)

### Deforestation Map

(Bản đồ phân loại toàn bộ khu vực rừng Cà Mau sẽ được tạo sau khi chọn được best model)

---

## 📝 Preprocessing Pipeline

### 1. Sentinel-2 Preprocessing
- Đọc 7 bands từ GeoTIFF
- Xử lý NoData values (convert to NaN)
- Clip outliers về physical ranges:
  - Spectral bands (B4, B8, B11, B12): [0, 1]
  - Spectral indices (NDVI, NBR, NDMI): [-1, 1]
- Apply boundary mask (chỉ giữ pixels trong vùng rừng)

### 2. Sentinel-1 Preprocessing
- Đọc VV và VH bands (dB values)
- Apply boundary mask
- MinMax normalization: [min, max] → [0, 1]

### 3. Patch Extraction
- Extract 64×64 patches tại các ground truth points
- Stack 18 channels: [S2_2024, S1_2024, S2_2025, S1_2025]
- Reject patches chứa NaN hoặc all-zero values
- Lưu thành pickle file cho training

---

## 🔧 Tối ưu hóa cho GTX 1060 6GB + 64GB RAM

Dự án được tối ưu hóa đặc biệt cho cấu hình phần cứng:

### GPU Optimization (GTX 1060 6GB):
- **Mixed Precision Training (AMP):** Enabled
  - Giảm ~40% VRAM usage (float16 thay vì float32)
  - Tăng tốc training ~20-30%
  - Không ảnh hưởng độ chính xác

- **Batch size:** 16-24 (tùy độ phức tạp của model)
  - Được test để tận dụng tối đa 6GB VRAM
  - Model nhẹ → batch size lớn hơn
  - Model nặng → batch size nhỏ hơn

- **Gradient Accumulation:** 2 steps
  - Effective batch size = 32-48
  - Giúp training ổn định hơn với dataset nhỏ (1,285 samples)
  - Trade-off: chậm hơn ~15-20% nhưng kết quả tốt hơn

### RAM Optimization (64GB):
- **Cache patches trong RAM:**
  - Load toàn bộ 1,285 patches vào RAM (~380 MB)
  - Training CỰC NHANH (không cần đọc disk mỗi epoch)
  - Dataset nhỏ nên hoàn toàn khả thi

- **DataLoader minimal:**
  - `num_workers = 4` (đủ vì data đã trong RAM)
  - `pin_memory = True` (tăng tốc CPU → GPU transfer)
  - `prefetch_factor = 2` (giảm vì không cần prefetch nhiều)
  - `persistent_workers = True` (giữ workers alive giữa epochs)

### Estimated Resource Usage:
- **VRAM:** ~4.5-5.5 GB / 6 GB (~85-95% utilization)
- **RAM:** ~15-20 GB / 64 GB (patches cache + system + PyTorch)
- **Training Speed:** ~5-10 giây/epoch (với cached data)

---

## 📚 Thư viện chính

- **PyTorch** 2.0+ - Deep learning framework
- **torchvision** - Computer vision models
- **segmentation-models-pytorch** - U-Net implementation
- **rasterio** - Đọc/ghi GeoTIFF files
- **geopandas** - Xử lý vector data (shapefiles)
- **numpy** - Numerical operations
- **pandas** - Data manipulation
- **matplotlib, seaborn** - Visualization
- **scikit-learn** - Metrics và utilities
- **tqdm** - Progress bars

---

## 🤝 Đóng góp

Dự án này là đồ án tốt nghiệp cá nhân. Mọi đóng góp, ý kiến, và góp ý xin vui lòng liên hệ qua email hoặc tạo issue trên GitHub.

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

## 🙏 Lời cảm ơn

- Giảng viên hướng dẫn: TS. Hà Minh Cường, ThS, Hoàng Tích Phúc
- Phòng thí nghiệm: Geospatial Technology Lab
- Viện Công nghệ Hàng không Vũ trụ - Trường Đại học Công nghệ, ĐHQGHN

---

**Cập nhật lần cuối:** 06/01/2025
