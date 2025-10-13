# Phát Hiện Mất Rừng Cà Mau Sử Dụng SNUNet-CD

**Ninh Hải Đăng (21021411) - Đồ Án Tốt Nghiệp - 2025**

Phát hiện mất rừng ngập mặn tự động sử dụng Deep Learning kết hợp ảnh vệ tinh đa thời gian (Sentinel-2 + Sentinel-1).

---

## 📊 Dữ Liệu

### Ground Truth
- **1,285 điểm** thực địa (shapefile + CSV)
- **635 điểm mất rừng** (49.4%)
- **650 điểm không mất** (50.6%)
- Chia: 80% train (1,028) / 10% val (128) / 10% test (129)

### Sentinel-2 (Quang học)
- **T1:** 30/01/2024 → **T2:** 28/02/2025
- **4 bands:** B4 (Đỏ), B8 (Cận hồng ngoại), B11 (SWIR1), B12 (SWIR2)
- **3 chỉ số:** NDVI (thực vật), NBR (cháy rừng), NDMI (độ ẩm)
- **Độ phân giải:** 10-20m
- **Files:** 2 file × 1.5GB GeoTIFF

### Sentinel-1 (SAR)
- **T1:** 04/02/2024 → **T2:** 22/02/2025
- **2 features:** VH polarization, Ratio (VV-VH)
- **Độ phân giải:** 10m
- **Files:** 2 file × 1.5GB GeoTIFF

### Channels Input

**Phase 1 (chỉ S2): 14 channels**
```
Trước T1: [B4, B8, B11, B12, NDVI, NBR, NDMI] = 7 channels
Sau T2:   [B4, B8, B11, B12, NDVI, NBR, NDMI] = 7 channels
Tổng: 14 channels
```

**Phase 2 (S2+S1): 18 channels**
```
Trước T1: [B4, B8, B11, B12, NDVI, NBR, NDMI, VH, Ratio] = 9 channels
Sau T2:   [B4, B8, B11, B12, NDVI, NBR, NDMI, VH, Ratio] = 9 channels
Tổng: 18 channels
```

---

## 🧠 Model & Training

### Kiến Trúc: SNUNet-CD
```python
SNUNet-CD (Siamese Nested U-Net)
├── Encoder: Siamese (shared weights)
│   ├── in_channels: 7 (Phase 1) hoặc 9 (Phase 2)
│   ├── width: 16
│   ├── depth: 4 blocks
│   └── channels: [16, 32, 64, 128]
├── ECAM: Enhanced Channel Attention Module
├── Decoder: Nested với dense skip connections
│   └── channels: [128, 64, 32, 16]
└── Head: 2 classes (binary change detection)

Số parameters: ~1.2M
```

### Config Training
```python
# Hyperparameters
optimizer: AdamW(lr=0.01, weight_decay=0.0005)
scheduler: PolynomialLR(power=0.9, min_lr=1e-4)
loss: CrossEntropyLoss
batch_size: 8
patch_size: 256×256
max_iterations: 40,000
validation_interval: 4,000
workers: 4

# Data Augmentation
RandomRotate(prob=0.5, degree=180)
RandomCrop(256×256)
RandomFlip(horizontal + vertical, prob=0.5)
Normalize(mean=[...], std=[...])
```

### Metrics Đánh Giá
- Overall Accuracy (mục tiêu: >90%)
- F1-Score (mục tiêu: >0.85)
- IoU (mục tiêu: >0.75)
- Precision (mục tiêu: >0.88)
- Recall (mục tiêu: >0.82)

---

## 💻 Môi Trường

### Phần Cứng
```
CPU: Intel Xeon E5-2678 v3 (12 cores @ 2.5GHz)
RAM: 32GB DDR3 ECC
GPU: NVIDIA RTX A4000 (16GB VRAM, 6144 CUDA cores)
Storage: 4TB HDD
OS: Windows 11 Pro
```

### Thư Viện & Phiên Bản
```yaml
# Core
Python: 3.8.20
PyTorch: 1.13.1+cu117
CUDA: 11.7
cuDNN: 8.5.0

# Framework
Open-CD: 1.1.0
  ├── MMSegmentation: 1.2.2
  ├── MMEngine: 0.10.4
  ├── MMCV: 2.1.0
  └── MMPretrain: 1.2.0

# Geospatial
GDAL: 3.9.2
rasterio: 1.3.11
geopandas: 0.14.4
shapely: 2.0.4

# Image Processing
opencv-python: 4.12.0
albumentations: 1.4.18
pillow: 10.4.0

# Scientific
numpy: 1.24.4
scipy: 1.13.1
pandas: 2.0.3
scikit-learn: 1.3.2

# Visualization
matplotlib: 3.7.5
seaborn: 0.13.2
```

### Cài Đặt
```bash
# Tạo environment
conda env create -f environment.yml
conda activate dang

# Cài Open-CD
cd open-cd && pip install -v -e . && cd ..

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 1.13.1+cu117 True
```

---

## 📁 Cấu Trúc Dự Án

```
├── data/
│   ├── raw/                          # ✅ ĐÃ CÓ
│   │   ├── sentinel2/                # 2 files (3GB)
│   │   │   ├── S2_2024_01_30.tif    # Before T1
│   │   │   └── S2_2025_02_28.tif    # After T2
│   │   ├── sentinel1/                # 2 files (3GB)
│   │   │   ├── S1_2024_02_04_matched_S2_2024_01_30.tif
│   │   │   └── S1_2025_02_22_matched_S2_2025_02_28.tif
│   │   └── ground_truth/             # 11 files
│   │       ├── Training_Points_CSV.csv
│   │       └── Training_Points__SHP.*
│   ├── processed/                    # ⏳ CẦN TẠO
│   │   ├── phase1_s2only/
│   │   └── phase2_s2s1/
│   └── samples/                      # ⏳ CẦN TẠO
│       ├── phase1_s2only/train|val|test/
│       └── phase2_s2s1/train|val|test/
│
├── notebooks/                        # ⏳ CẦN CHẠY
│   ├── 01_exploration/
│   ├── 02_preprocessing/
│   ├── 03_phase1_s2only/
│   ├── 04_phase2_s2s1/
│   └── 05_comparison/
│
├── configs/                          # ✅ ĐÃ CÓ
│   ├── phase1_snunet_s2only.py       # Config 14 channels
│   └── phase2_snunet_s2s1.py         # Config 18 channels
│
├── src/                              # ✅ ĐÃ CÓ
│   ├── data_utils.py                 # Load, visualize, tính indices
│   ├── training_utils.py             # Checkpoint, logging
│   └── evaluation_utils.py           # Metrics, confusion matrix
│
├── experiments/                      # ⏳ SAU KHI TRAIN
│   ├── phase1_s2only/
│   │   ├── checkpoints/              # Model weights
│   │   ├── logs/                     # Training logs
│   │   ├── metrics/                  # JSON metrics
│   │   └── predictions/              # Predictions mẫu
│   └── phase2_s2s1/
│
├── results/                          # ⏳ SAU KHI INFERENCE
│   ├── maps/                         # Bản đồ change detection
│   ├── statistics/                   # Thống kê
│   └── figures/                      # Hình ảnh cho báo cáo
│
├── thesis/                           # ⏳ CHO BÁO CÁO
│   ├── figures/
│   ├── tables/
│   └── slides/
│
├── docs/                             # ✅ ĐÃ CÓ
│   ├── 00_project_overview.md
│   ├── 01_data_guide.md
│   └── 02_training_guide.md
│
└── open-cd/                          # ✅ ĐÃ CÓ (cloned)
    └── tools/
        ├── train.py
        └── test.py
```

---

## ✅ Tiến Độ Thực Hiện

### ĐÃ HOÀN THÀNH ✅ (3 tuần trước)

- [x] **Setup môi trường**
  - [x] Cài Python 3.8.20, PyTorch 1.13.1+cu117, CUDA 11.7
  - [x] Cài Open-CD 1.1.0 và dependencies
  - [x] Verify GPU RTX A4000 hoạt động tốt

- [x] **Thu thập dữ liệu**
  - [x] Sentinel-2: 2 files (3GB) - T1, T2
  - [x] Sentinel-1: 2 files (3GB) - matched với S2
  - [x] Ground truth: 1,285 điểm (shapefile + CSV)

- [x] **Thiết kế dự án**
  - [x] Cấu trúc thư mục rõ ràng
  - [x] Migration từ cấu trúc cũ
  - [x] Cleanup các file không cần thiết

- [x] **Tạo config files**
  - [x] `configs/phase1_snunet_s2only.py` (14 channels)
  - [x] `configs/phase2_snunet_s2s1.py` (18 channels)

- [x] **Viết utility functions**
  - [x] `src/data_utils.py` (load, visualize, NDVI/NBR/NDMI)
  - [x] `src/training_utils.py` (checkpoint handling)
  - [x] `src/evaluation_utils.py` (metrics, confusion matrix)

- [x] **Documentation**
  - [x] README.md (file này)
  - [x] docs/ (3 files hướng dẫn)

---

## 📅 TIMELINE 1 TUẦN (7 NGÀY)

### NGÀY 1 (Thứ 2): Khám Phá & Tiền Xử Lý ⏳
**Thời gian: 8-10 giờ**

**Sáng (4h):**
- [ ] **1.1. Explore Sentinel-2** (1.5h)
  - Load T1, T2
  - Visualize RGB composite
  - Tính NDVI, NBR, NDMI
  - Phân tích thống kê
  
- [ ] **1.2. Explore Sentinel-1** (1h)
  - Load SAR data
  - Visualize VH backscatter
  - So sánh T1 vs T2
  
- [ ] **1.3. Analyze Ground Truth** (1.5h)
  - Load 1,285 điểm
  - Visualize phân bố không gian
  - Verify class balance

**Chiều (4-6h):**
- [ ] **2.1. Preprocess Phase 1** (2h)
  - Extract 4 bands từ S2
  - Compute 3 indices
  - Normalize [0,1]
  - Save → `data/processed/phase1_s2only/`
  
- [ ] **2.2. Preprocess Phase 2** (2h)
  - Merge S2 (7ch) + S1 (2ch)
  - Verify co-registration
  - Save → `data/processed/phase2_s2s1/`

**Kết quả:** Data đã sẵn sàng để tạo training samples

---

### NGÀY 2 (Thứ 3): Tạo Training Samples ⏳
**Thời gian: 6-8 giờ**

- [ ] **2.3. Create Training Samples** (6-8h)
  - Extract 256×256 patches xung quanh 1,285 ground truth points
  - Implement coordinate transformation (lat/lon → pixel)
  - Stratified split: 80/10/10
  - Save patches:
    - `data/samples/phase1_s2only/train/` (1,028 patches)
    - `data/samples/phase1_s2only/val/` (128 patches)
    - `data/samples/phase1_s2only/test/` (129 patches)
    - `data/samples/phase2_s2s1/train/` (same split)
  - Visualize một số samples để verify
  - Test dataloader với Open-CD

**Kết quả:** 1,285 × 2 phases = 2,570 training patches sẵn sàng

---

### NGÀY 3 (Thứ 4): Training Phase 1 (Buổi 1) ⏳
**Thời gian: Training chạy 12-16h, monitor 2-3h**

**Sáng:**
- [ ] **Bắt đầu training Phase 1** (10-15 phút setup)
  ```bash
  python open-cd/tools/train.py configs/phase1_snunet_s2only.py
  ```
- [ ] Setup TensorBoard monitoring
  ```bash
  tensorboard --logdir experiments/phase1_s2only/logs
  ```
- [ ] Verify training bắt đầu:
  - Loss giảm
  - GPU utilization ~90%
  - No errors

**Trong ngày:**
- [ ] Monitor training mỗi 2-3h
- [ ] Check validation metrics (mỗi 4k iterations)
- [ ] **Training chạy qua đêm** (40k iterations ≈ 12-16h)

**Chiều (tùy chọn):**
- [ ] Chuẩn bị notebook evaluation
- [ ] Viết script để parse logs
- [ ] Chuẩn bị visualizations

**Kết quả buổi sáng ngày 4:** Phase 1 training hoàn thành

---

### NGÀY 4 (Thứ 5): Evaluate Phase 1 & Start Phase 2 ⏳
**Thời gian: 3h evaluate + Training Phase 2 chạy qua đêm**

**Sáng (3h):**
- [ ] **Evaluate Phase 1** 
  - Chờ training Phase 1 hoàn thành (~7-8h sáng)
  - Run test:
    ```bash
    python open-cd/tools/test.py \
        configs/phase1_snunet_s2only.py \
        experiments/phase1_s2only/checkpoints/best_model.pth
    ```
  - Phân tích metrics:
    - Overall Accuracy
    - F1-Score
    - IoU
    - Precision/Recall
  - Plot confusion matrix
  - Visualize predictions (10-20 samples)
  - Save results → `experiments/phase1_s2only/metrics/`

**Trưa (1h):**
- [ ] Tổng kết Phase 1
- [ ] Note các vấn đề/cải thiện

**Chiều (10-15 phút + chạy qua đêm):**
- [ ] **Bắt đầu training Phase 2**
  ```bash
  python open-cd/tools/train.py configs/phase2_snunet_s2s1.py
  ```
- [ ] Setup monitoring
- [ ] Verify training bắt đầu
- [ ] **Training chạy qua đêm** (40k iterations ≈ 12-16h)

**Kết quả buổi sáng ngày 5:** Phase 2 training hoàn thành

---

### NGÀY 5 (Thứ 6): Evaluate Phase 2 & So Sánh ⏳
**Thời gian: 6-8 giờ**

**Sáng (3h):**
- [ ] **Evaluate Phase 2**
  - Chờ training hoàn thành (~7-8h sáng)
  - Run test
  - Phân tích metrics
  - Plot confusion matrix
  - Visualize predictions
  - Save results

**Chiều (3-5h):**
- [ ] **So sánh Phase 1 vs Phase 2**
  - Tạo comparison table:
    | Metric | Phase 1 | Phase 2 | Δ |
    |--------|---------|---------|---|
    | Accuracy | ... | ... | ... |
    | F1-Score | ... | ... | ... |
  - Confusion matrices side-by-side
  - Sample predictions comparison
  - Statistical significance test (t-test)
  - Error analysis:
    - Identify failure cases
    - Analyze where S1 helps
  - Save report → `results/statistics/comparison.md`

**Kết quả:** Hiểu rõ Phase 2 cải thiện bao nhiêu so với Phase 1

---

### NGÀY 6 (Thứ 7): Inference Toàn Tỉnh ⏳
**Thời gian: 6-10 giờ (tùy diện tích inference)**

- [ ] **Inference trên toàn bộ tỉnh Cà Mau**
  - Chọn best model (Phase 1 hoặc Phase 2)
  - Implement sliding window inference (256×256 với overlap)
  - Run inference trên toàn bộ region (có thể mất 4-8h)
  - Merge predictions → bản đồ change detection
  
- [ ] **Tính toán thống kê**
  - Tổng diện tích mất rừng (km²)
  - Phân bố theo vùng
  - Temporal analysis
  - Export → `results/statistics/deforestation_stats.csv`

- [ ] **Tạo visualizations**
  - Change detection map (GeoTIFF + PNG)
  - Heatmap thay đổi
  - Comparison with ground truth overlay
  - Save → `results/maps/` và `results/figures/`

**Kết quả:** Bản đồ change detection hoàn chỉnh cho toàn tỉnh

---

### NGÀY 7 (Chủ Nhật): Finalize & Documentation ⏳
**Thời gian: 6-8 giờ**

**Sáng (3-4h):**
- [ ] **Tổng hợp kết quả**
  - Compile tất cả metrics
  - Tạo summary tables
  - Export figures chất lượng cao cho thesis
  - Organize trong `thesis/figures/` và `thesis/tables/`

**Chiều (3-4h):**
- [ ] **Update documentation**
  - Update README với actual results
  - Ghi chú lessons learned
  - Document final metrics
  - List limitations & future work
  
- [ ] **Prepare presentation materials**
  - Key findings slides
  - Demo materials
  - Screenshots và visualizations

- [ ] **Backup & Archive**
  - Backup toàn bộ code + data quan trọng
  - Archive experiments
  - Clean up temporary files

**Kết quả:** Dự án hoàn thành, sẵn sàng báo cáo

---

## 🚀 Quick Commands

### Environment
```bash
conda activate dang
conda deactivate
```

### GPU Check
```bash
nvidia-smi
nvidia-smi -l 1  # Monitor mỗi 1 giây
```

### Training
```bash
# Phase 1 (S2 only)
python open-cd/tools/train.py configs/phase1_snunet_s2only.py

# Phase 2 (S2+S1)
python open-cd/tools/train.py configs/phase2_snunet_s2s1.py
```

### Testing
```bash
# Phase 1
python open-cd/tools/test.py \
    configs/phase1_snunet_s2only.py \
    experiments/phase1_s2only/checkpoints/best_model.pth

# Phase 2
python open-cd/tools/test.py \
    configs/phase2_snunet_s2s1.py \
    experiments/phase2_s2s1/checkpoints/best_model.pth
```

### Monitoring
```bash
# TensorBoard
tensorboard --logdir experiments/phase1_s2only/logs
tensorboard --logdir experiments/phase2_s2s1/logs

# Check logs
Get-Content experiments\phase1_s2only\*.log -Tail 50
```

### Jupyter
```bash
jupyter lab
jupyter notebook
```

---

## 📊 Expected Results (Dự Kiến)

### Phase 1 (S2 only)
- **Accuracy:** ~88-92%
- **F1-Score:** ~0.83-0.87
- **IoU:** ~0.72-0.78
- **Training time:** ~12-16h (40k iterations)

### Phase 2 (S2 + S1)
- **Accuracy:** ~90-94% (+2-4%)
- **F1-Score:** ~0.86-0.91 (+0.03-0.05)
- **IoU:** ~0.76-0.82 (+0.04-0.06)
- **Training time:** ~12-16h (40k iterations)

### Improvement với S1
- Giảm false positives (precision tăng)
- Giảm false negatives trong vùng mây (recall tăng)
- Robust hơn với điều kiện thời tiết

---

## 📝 Files Quan Trọng

### Configs
```python
# configs/phase1_snunet_s2only.py
model = dict(
    backbone=dict(in_channels=7),  # S2 only
    decode_head=dict(num_classes=2)
)
data = dict(
    samples_per_gpu=8,
    data_root='data/samples/phase1_s2only'
)
optimizer = dict(type='AdamW', lr=0.01)
runner = dict(max_iters=40000)
```

```python
# configs/phase2_snunet_s2s1.py
model = dict(
    backbone=dict(in_channels=9),  # S2 + S1
)
data = dict(
    data_root='data/samples/phase2_s2s1'
)
# Còn lại giống Phase 1
```

### Utility Functions
```python
# src/data_utils.py
load_geotiff(filepath)              # Load GeoTIFF
visualize_rgb(data, bands)          # Visualize RGB
calculate_ndvi(nir, red)            # NDVI
calculate_nbr(nir, swir2)           # NBR
calculate_ndmi(nir, swir1)          # NDMI

# src/evaluation_utils.py
calculate_metrics(y_true, y_pred)   # All metrics
plot_confusion_matrix(y_true, y_pred)
```

---

## ⏰ Thời Gian Ước Tính Chi Tiết

| Ngày | Task | Giờ làm | Giờ chờ | Tổng |
|------|------|---------|---------|------|
| **1** | Explore + Preprocess | 8-10h | - | 8-10h |
| **2** | Create samples | 6-8h | - | 6-8h |
| **3** | Start Phase 1 training | 0.5h | 12-16h | ~16h |
| **4** | Eval P1 + Start P2 | 3h | 12-16h | ~19h |
| **5** | Eval P2 + Compare | 6-8h | - | 6-8h |
| **6** | Inference | 6-10h | - | 6-10h |
| **7** | Finalize | 6-8h | - | 6-8h |
| **Tổng** | | **36-50h** làm việc | **24-32h** chờ training |

**Lưu ý:** 
- Training chạy tự động qua đêm → tiết kiệm thời gian
- Ngày 3-4 có thể làm việc khác trong khi training
- Cần monitor định kỳ để catch errors

---

## 🎯 Checklist Tổng Quan

### Tuần Này (7 Ngày)
- [ ] Ngày 1: Explore & Preprocess data
- [ ] Ngày 2: Create training samples
- [ ] Ngày 3: Training Phase 1 (qua đêm)
- [ ] Ngày 4: Evaluate Phase 1 + Training Phase 2 (qua đêm)
- [ ] Ngày 5: Evaluate Phase 2 + Comparison
- [ ] Ngày 6: Inference toàn tỉnh
- [ ] Ngày 7: Finalize & Documentation

### Deliverables
- [ ] Trained models (2 phases)
- [ ] Metrics reports (JSON + markdown)
- [ ] Change detection maps
- [ ] Statistics & analysis
- [ ] Visualizations cho thesis
- [ ] Updated documentation

---

**Cập nhật lần cuối:** 13/10/2025  
**Trạng thái:** Chuẩn bị bắt đầu (Ngày 1/7)  
**Timeline:** 1 tuần (aggressive)  
**Tiến độ hiện tại:** Setup hoàn tất, sẵn sàng execution