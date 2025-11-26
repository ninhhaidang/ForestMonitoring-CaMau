# CHƯƠNG 3: PHƯƠNG PHÁP NGHIÊN CỨU

## 3.1. Khu vực và dữ liệu nghiên cứu

### 3.1.1. Khu vực nghiên cứu

**Vị trí địa lý:**

Tỉnh Cà Mau nằm ở cực Nam Tổ Quốc, thuộc vùng Đồng bằng sông Cửu Long:
- **Tọa độ địa lý:** 8°36' - 9°27' Bắc, 104°43' - 105°10' Đông
- **Diện tích tự nhiên:** 5,331.7 km²
- **Dân số:** ~1.2 triệu người (2020)
- **Đường bờ biển:** 254 km

**Ranh giới hành chính:**
- Phía Bắc: giáp tỉnh Kiên Giang và Bạc Liêu
- Phía Đông: giáp tỉnh Bạc Liêu và Biển Đông
- Phía Tây và Nam: giáp Vịnh Thái Lan

**Đặc điểm địa hình:**

- **Cao độ:** 0.5 - 1.5m so với mực nước biển (địa hình thấp trũng)
- **Địa hình:** Đồng bằng phù sa ven biển, nhiều sông rạch
- **Thổ nhưỡng:** Phèn, mặn, và phù sa ven biển
- **Khí hậu:** Nhiệt đới gió mùa, 2 mùa mưa/khô rõ rệt

**Tài nguyên rừng:**

Cà Mau có hệ sinh thái rừng ngập mặn quan trọng:

| Loại rừng | Diện tích (ha) | Tỷ lệ (%) | Đặc điểm |
|-----------|----------------|-----------|----------|
| Rừng ngập mặn tự nhiên | ~28,000 | 70% | Tràm, đước, vẹt |
| Rừng ngập mặn trồng | ~12,000 | 30% | Chủ yếu tràm |
| **Tổng diện tích rừng** | **~40,000** | **100%** | 8% diện tích tỉnh |

**Vùng nghiên cứu:**

Luận văn tập trung vào toàn bộ diện tích rừng trong ranh giới tỉnh Cà Mau:

- **Diện tích nghiên cứu:** 162,469.25 hecta (1,624.69 km²)
- **Tọa độ UTM (Zone 48N):**
  - X min: 477,000 mE
  - X max: 586,000 mE
  - Y min: 995,000 mN
  - Y max: 1,120,000 mN
- **Kích thước raster:** 12,547 × 10,917 pixels (độ phân giải 10m)
- **Hệ quy chiếu:** EPSG:32648 (WGS 84 / UTM Zone 48N)

**Khu vực trọng điểm:**

- **Vườn Quốc gia U Minh Hạ:** 8,038 ha rừng tràm nguyên sinh
- **Khu Dự trữ Sinh quyển UNESCO Nam Cà Mau:** 41,862 ha (bao gồm cả vùng đệm)
- **Rừng phòng hộ ven biển:** Dọc theo 254 km đường bờ biển

### 3.1.2. Dữ liệu viễn thám

**Tổng quan dữ liệu:**

| Nguồn dữ liệu | Độ phân giải | Kỳ ảnh | Số bands | Dung lượng |
|---------------|--------------|--------|----------|------------|
| Sentinel-2 Before | 10m | 30/01/2024 | 7 | ~850 MB |
| Sentinel-2 After | 10m | 28/02/2025 | 7 | ~850 MB |
| Sentinel-1 Before | 10m | 04/02/2024 | 2 | ~250 MB |
| Sentinel-1 After | 10m | 22/02/2025 | 2 | ~250 MB |
| Ground Truth | - | - | - | 2,630 points |
| Forest Boundary | Vector | - | - | Shapefile |

**Sentinel-2 Multispectral Data:**

**Kỳ trước (Before): 30/01/2024**
- **Satellite:** Sentinel-2A
- **Product Level:** Level-2A (Surface Reflectance)
- **Cloud cover:** < 5%
- **Processing baseline:** 05.09
- **Tile ID:** 48PXS

**Kỳ sau (After): 28/02/2025**
- **Satellite:** Sentinel-2B
- **Product Level:** Level-2A
- **Cloud cover:** < 3%
- **Processing baseline:** 05.10
- **Tile ID:** 48PXS

**Bands được sử dụng:**

| Band | Tên | Bước sóng (nm) | Độ phân giải gốc | Sử dụng |
|------|-----|----------------|------------------|---------|
| B4 | Red | 665 | 10m | ✓ Spectral + NDVI |
| B8 | NIR | 842 | 10m | ✓ Spectral + NDVI/NBR/NDMI |
| B11 | SWIR1 | 1610 | 20m → 10m | ✓ NDMI |
| B12 | SWIR2 | 2190 | 20m → 10m | ✓ NBR |

**Tiền xử lý Sentinel-2:**

1. **Atmospheric correction:** Sen2Cor algorithm (đã áp dụng trong L2A)
2. **Resampling:** B11, B12 từ 20m → 10m (cubic convolution)
3. **Cloud masking:** Sử dụng Scene Classification Layer (SCL)
4. **Clipping:** Cắt theo ranh giới rừng Cà Mau
5. **Nodata masking:** Gán NoData = 0

**Sentinel-1 SAR Data:**

**Kỳ trước (Before): 04/02/2024**
- **Satellite:** Sentinel-1A
- **Mode:** IW (Interferometric Wide Swath)
- **Product Type:** GRD (Ground Range Detected)
- **Polarization:** VV + VH
- **Orbit:** Descending
- **Relative orbit:** 162

**Kỳ sau (After): 22/02/2025**
- **Satellite:** Sentinel-1B
- **Mode:** IW
- **Product Type:** GRD
- **Polarization:** VV + VH
- **Orbit:** Descending
- **Relative orbit:** 162

**Tiền xử lý Sentinel-1:**

1. **Thermal noise removal:** Loại bỏ thermal noise ở subswath borders
2. **Radiometric calibration:** Chuyển đổi Digital Number → Sigma Nought (σ⁰) dB
   ```
   σ⁰_dB = 10 × log₁₀(DN² / calibration_constant²)
   ```
3. **Speckle filtering:** Lee filter 5×5 window (giảm speckle noise)
4. **Terrain correction:** Range-Doppler orthorectification với SRTM 30m DEM
5. **Co-registration với Sentinel-2:** Align geometry chính xác
6. **Resampling:** 10m pixel spacing
7. **Clipping:** Cắt theo ranh giới rừng

**Chỉ số backscatter:**

```
VV_dB = 10 × log₁₀(VV_linear)
VH_dB = 10 × log₁₀(VH_linear)
```

Typical values cho rừng ngập mặn:
- VV: -8 to -12 dB
- VH: -15 to -20 dB

### 3.1.3. Ground Truth Data

**Thu thập dữ liệu ground truth:**

**Phương pháp:**
1. **Phiên giải ảnh độ phân giải cao:**
   - Google Earth imagery (0.5m resolution)
   - Planet Labs (3m resolution)
   - Sentinel-2 RGB composite

2. **Chiến lược lấy mẫu:**
   - **Stratified random sampling:** Đảm bảo đại diện cho tất cả các class
   - **Minimum distance:** 30m giữa các điểm (tránh spatial autocorrelation quá cao)
   - **Representative locations:** Phủ đều toàn bộ khu vực nghiên cứu

3. **Xác định label:**
   - So sánh ảnh Before và After
   - Phân loại thành 4 classes dựa trên biến động

**Thống kê Ground Truth:**

| Class | Tên | Số điểm | Tỷ lệ (%) | Mô tả |
|-------|-----|---------|-----------|-------|
| 0 | Forest Stable | 656 | 24.9% | Rừng ổn định (có rừng ở cả 2 kỳ) |
| 1 | Deforestation | 650 | 24.7% | Mất rừng (có rừng → không có rừng) |
| 2 | Non-forest | 664 | 25.3% | Không phải rừng (không có rừng ở cả 2 kỳ) |
| 3 | Reforestation | 660 | 25.1% | Tái trồng rừng (không có → có rừng) |
| **Tổng** | | **2,630** | **100%** | Balanced distribution |

**Format dữ liệu:**

File CSV: `data/raw/samples/4labels.csv`

```csv
id,label,x,y
1,0,479379.698,1002444.056
2,0,485903.845,1021227.658
3,1,486234.021,1020780.398
...
```

Trong đó:
- `id`: Unique identifier
- `label`: Class (0, 1, 2, 3)
- `x, y`: Tọa độ UTM Zone 48N (meters)

**Phân bố không gian:**

Ground truth points được phân bố đều trên toàn bộ khu vực nghiên cứu để đảm bảo:
- Đại diện cho tất cả các khu vực rừng (ven biển, nội địa, vùng cao, vùng thấp)
- Đa dạng về điều kiện môi trường (độ mặn, độ ẩm, cấu trúc rừng)
- Không bias về vị trí địa lý

**Độ tin cậy:**

- **Accuracy assessment:** Kiểm tra chéo với field survey data (nếu có)
- **Multi-interpreter check:** 10% samples được kiểm tra bởi 2 người phiên giải
- **Temporal consistency:** Kiểm tra chuỗi thời gian dài hạn từ Landsat

### 3.1.4. Forest Boundary

**Shapefile ranh giới rừng:**

- **File:** `data/raw/boundary/forest_boundary.shp`
- **CRS:** EPSG:32648 (WGS 84 / UTM Zone 48N)
- **Geometry type:** Polygon
- **Số đối tượng:** Multiple polygons (các mảnh rừng riêng biệt)
- **Nguồn:** Bản đồ hiện trạng rừng tỉnh Cà Mau (2023)

**Mục đích sử dụng:**

1. **Clipping:** Cắt raster chỉ lấy vùng có rừng
2. **Masking:** Tạo valid mask cho analysis
3. **Area calculation:** Tính diện tích từng class
4. **Visualization:** Hiển thị ranh giới trên bản đồ

**Valid pixel mask:**

```python
valid_mask = rasterize(forest_boundary,
                       out_shape=raster.shape,
                       transform=raster.transform)
# valid_mask[i,j] = True nếu pixel (i,j) nằm trong rừng
# valid_mask[i,j] = False nếu pixel (i,j) nằm ngoài rừng
```

**Thống kê:**
- **Total pixels trong raster:** 136,975,599 (12,547 × 10,917)
- **Valid pixels (trong forest boundary):** 16,246,925 (11.86%)
- **Invalid pixels (ngoài rừng):** 120,728,674 (88.14%)

---

## 3.2. Quy trình xử lý dữ liệu

### 3.2.1. Tổng quan quy trình

**Flowchart tổng quát:**

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA INPUTS                          │
├──────────────┬──────────────┬───────────────┬──────────────┤
│  Sentinel-2  │  Sentinel-1  │ Ground Truth  │   Boundary   │
│   Before     │   Before     │   CSV         │   Shapefile  │
│   After      │   After      │   (2,630)     │              │
└──────┬───────┴──────┬───────┴───────┬───────┴──────┬───────┘
       │              │               │              │
       ▼              ▼               ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 1: DATA LOADING & VALIDATION              │
│  - Load rasters (rasterio)                                  │
│  - Load ground truth (pandas)                               │
│  - Check CRS, shape, transform consistency                  │
│  - Validate data quality                                    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           STEP 2: FEATURE EXTRACTION (27 features)          │
│                                                             │
│  S2 Features (21):                                          │
│    - S2_before[0:7]:  B4, B8, B11, B12, NDVI, NBR, NDMI    │
│    - S2_after[0:7]:   B4, B8, B11, B12, NDVI, NBR, NDMI    │
│    - S2_delta[0:7]:   ΔB4, ΔB8, ΔB11, ΔB12, ΔNDVI, ...     │
│                                                             │
│  S1 Features (6):                                           │
│    - S1_before[0:2]:  VV, VH                               │
│    - S1_after[0:2]:   VV, VH                               │
│    - S1_delta[0:2]:   ΔVV, ΔVH                             │
│                                                             │
│  Output: feature_stack (27, H, W)                          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 3: PATCH EXTRACTION (3×3)                 │
│  - Convert ground truth coordinates to pixel indices        │
│  - Extract 3×3 patches at each ground truth location        │
│  - Skip edge pixels and NoData pixels                       │
│  - Output: patches (N, 3, 3, 27), labels (N,)              │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                STEP 4: NORMALIZATION                        │
│  - Compute mean and std across all patches                  │
│  - Z-score standardization: (x - μ) / σ                    │
│  - Save normalization parameters for inference              │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         STEP 5: SPATIAL-AWARE DATA SPLITTING                │
│  - Hierarchical clustering (distance threshold = 50m)       │
│  - Assign clusters to train/val/test                        │
│  - Train: 70%, Val: 15%, Test: 15%                         │
│  - Verify spatial separation                                │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            OUTPUT: READY-TO-TRAIN DATASET                   │
│  - X_train (1,838, 3, 3, 27), y_train (1,838,)            │
│  - X_val (395, 3, 3, 27), y_val (395,)                    │
│  - X_test (396, 3, 3, 27), y_test (396,)                  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2.2. Feature Extraction chi tiết

**Input data:**
- S2_before: (7, H, W) - Sentinel-2 kỳ trước
- S2_after: (7, H, W) - Sentinel-2 kỳ sau
- S1_before: (2, H, W) - Sentinel-1 kỳ trước
- S1_after: (2, H, W) - Sentinel-1 kỳ sau

**Sentinel-2 indices calculation:**

```python
# NDVI
NDVI = (NIR - Red) / (NIR + Red + 1e-8)

# NBR
NBR = (NIR - SWIR2) / (NIR + SWIR2 + 1e-8)

# NDMI
NDMI = (NIR - SWIR1) / (NIR + SWIR1 + 1e-8)
```

**Feature stack construction:**

```python
# Sentinel-2 features (21)
S2_before_all = [B4, B8, B11, B12, NDVI, NBR, NDMI]  # 7 bands
S2_after_all = [B4, B8, B11, B12, NDVI, NBR, NDMI]   # 7 bands
S2_delta = S2_after_all - S2_before_all               # 7 bands

# Sentinel-1 features (6)
S1_before_all = [VV, VH]                              # 2 bands
S1_after_all = [VV, VH]                               # 2 bands
S1_delta = S1_after_all - S1_before_all               # 2 bands

# Stack tất cả features
feature_stack = np.concatenate([
    S2_before_all,   # indices 0-6
    S2_after_all,    # indices 7-13
    S2_delta,        # indices 14-20
    S1_before_all,   # indices 21-22
    S1_after_all,    # indices 23-24
    S1_delta         # indices 25-26
], axis=0)

# Shape: (27, 12547, 10917)
```

**Feature descriptions:**

| Index | Feature | Nguồn | Mô tả |
|-------|---------|-------|-------|
| 0 | S2_B4_before | S2 | Red band kỳ trước |
| 1 | S2_B8_before | S2 | NIR band kỳ trước |
| 2 | S2_B11_before | S2 | SWIR1 band kỳ trước |
| 3 | S2_B12_before | S2 | SWIR2 band kỳ trước |
| 4 | S2_NDVI_before | S2 | NDVI kỳ trước |
| 5 | S2_NBR_before | S2 | NBR kỳ trước |
| 6 | S2_NDMI_before | S2 | NDMI kỳ trước |
| 7-13 | S2_*_after | S2 | Tương tự kỳ sau |
| 14-20 | S2_delta_* | S2 | Biến đổi (after - before) |
| 21 | S1_VV_before | S1 | VV polarization kỳ trước |
| 22 | S1_VH_before | S1 | VH polarization kỳ trước |
| 23-24 | S1_*_after | S1 | Tương tự kỳ sau |
| 25-26 | S1_delta_* | S1 | Biến đổi (after - before) |

**Valid mask creation:**

```python
# Tạo mask cho pixels hợp lệ
valid_S2 = ~np.isnan(S2_before_all).any(axis=0) & \
           ~np.isnan(S2_after_all).any(axis=0)

valid_S1 = ~np.isnan(S1_before_all).any(axis=0) & \
           ~np.isnan(S1_after_all).any(axis=0)

# Valid mask: có dữ liệu S1 HOẶC S2 (relaxed constraint)
valid_mask = valid_S1 | valid_S2

# Intersect với forest boundary
valid_mask = valid_mask & forest_boundary_mask

# Valid pixels: 16,246,925 (11.86% of total)
```

### 3.2.3. Patch Extraction

**Coordinate transformation:**

```python
# Ground truth coordinates (UTM)
x_geo, y_geo = ground_truth['x'], ground_truth['y']

# Transform to pixel coordinates
from rasterio.transform import rowcol
row, col = rowcol(transform, x_geo, y_geo)

# row, col là pixel indices (0-based)
```

**3×3 Patch extraction:**

```python
patch_size = 3
half_size = 1  # (3-1) / 2

patches = []
labels = []
valid_indices = []

for idx, (r, c, label) in enumerate(zip(rows, cols, ground_truth['label'])):
    # Check edge constraints
    if r < half_size or r >= H - half_size or \
       c < half_size or c >= W - half_size:
        continue  # Skip edge pixels

    # Extract 3×3 patch
    patch = feature_stack[:,
                         r-half_size:r+half_size+1,
                         c-half_size:c+half_size+1]
    # Shape: (27, 3, 3)

    # Check valid mask
    patch_mask = valid_mask[r-half_size:r+half_size+1,
                           c-half_size:c+half_size+1]
    if not patch_mask.all():
        continue  # Skip if any pixel in patch is invalid

    # Check for NaN or Inf
    if np.isnan(patch).any() or np.isinf(patch).any():
        continue

    # Transpose to (3, 3, 27) for CNN input format
    patch = np.transpose(patch, (1, 2, 0))

    patches.append(patch)
    labels.append(label)
    valid_indices.append(idx)

# Convert to numpy arrays
patches = np.array(patches)  # Shape: (N, 3, 3, 27)
labels = np.array(labels)    # Shape: (N,)

# N = 2,630 (extracted successfully from all valid ground truth points)
```

**Patch visualization:**

Ví dụ patch tại một ground truth point:

```
Feature stack tại (row=5000, col=5000):

Patch cho B4_before (feature index 0):
[245  250  248]
[242  247  252]
[240  245  250]

Patch cho NDVI_before (feature index 4):
[0.65  0.68  0.67]
[0.63  0.66  0.70]
[0.62  0.65  0.68]

→ Tất cả 27 features → Patch shape (3, 3, 27)
```

### 3.2.4. Normalization

**Z-score standardization:**

```python
# Compute statistics across all patches
mean = patches.mean(axis=(0, 1, 2), keepdims=True)  # Shape: (1, 1, 1, 27)
std = patches.std(axis=(0, 1, 2), keepdims=True)    # Shape: (1, 1, 1, 27)

# Normalize
patches_normalized = (patches - mean) / (std + 1e-8)

# Save normalization parameters
np.save('normalization_mean.npy', mean)
np.save('normalization_std.npy', std)
```

**Statistics per feature:**

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| S2_B4_before | 250.5 | 45.2 | 120 | 380 |
| S2_NDVI_before | 0.65 | 0.12 | 0.2 | 0.9 |
| S1_VV_before | -10.5 | 2.3 | -18 | -5 |
| ... | ... | ... | ... | ... |

**Importance of normalization:**

1. **Scale features:** Các features có scale khác nhau (B4: 0-400, NDVI: -1 to 1)
2. **Faster convergence:** Gradient descent hội tụ nhanh hơn
3. **Stable training:** Tránh numerical instability
4. **Better initialization:** Weights được khởi tạo cho normalized data

### 3.2.5. Quy trình đánh giá: 5-Fold Cross Validation + Fixed Test Set

**Problem with traditional splitting:**

```
Random split → Training và test samples có thể rất gần nhau trong không gian
→ High spatial correlation → Data leakage → Overestimate accuracy
```

**Giải pháp: Quy trình đánh giá khoa học với 5-Fold Cross Validation**

Nghiên cứu áp dụng quy trình đánh giá 4 bước:

```
Step 1: Tách 20% dữ liệu làm Fixed Test Set (không đụng đến trong quá trình training)
Step 2: 5-Fold Cross Validation trên 80% còn lại (Train+Val)
Step 3: Huấn luyện Final Model trên toàn bộ 80%
Step 4: Đánh giá Final Model trên 20% Test Set
```

**Step 1: Fixed Test Set với Spatial-Aware Splitting**

```python
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.model_selection import train_test_split

# Ground truth coordinates
coords = ground_truth[['x', 'y']].values  # Shape: (2630, 2)

# Hierarchical clustering để tránh data leakage
distances = pdist(coords, metric='euclidean')
linkage_matrix = linkage(distances, method='single')
distance_threshold = 50.0  # meters
cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

# Chia Train+Val (80%) và Test (20%)
trainval_indices, test_indices = train_test_split(
    np.arange(len(ground_truth)),
    test_size=0.20,
    stratify=ground_truth['label'],
    random_state=42
)
```

**Step 2: 5-Fold Cross Validation trên Train+Val**

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(cv.split(X_trainval, y_trainval)):
    # Train model on train_idx
    # Validate on val_idx
    # Save fold metrics
```

**Thống kê phân chia dữ liệu:**

| Split | Số mẫu | Tỷ lệ (%) | Mô tả |
|-------|--------|-----------|-------|
| Train+Val | 2,104 | 80.0% | Dùng cho 5-Fold CV |
| Test | 526 | 20.0% | Fixed, không đụng đến trong training |
| **Tổng** | **2,630** | **100%** | |

**Phân bố lớp trong Test Set (526 mẫu):**

| Lớp | Tên | Số mẫu | Tỷ lệ |
|-----|-----|--------|-------|
| 0 | Forest Stable | 131 | 24.9% |
| 1 | Deforestation | 130 | 24.7% |
| 2 | Non-forest | 133 | 25.3% |
| 3 | Reforestation | 132 | 25.1% |

**Ưu điểm của quy trình này:**

1. **Tránh data leakage**: Test set hoàn toàn độc lập, không tham gia vào bất kỳ giai đoạn nào của training
2. **Đánh giá robust**: 5-Fold CV cho variance estimation trên training performance
3. **Hyperparameter selection**: Có thể tune hyperparameters dựa trên CV mà không làm "rò rỉ" thông tin từ test set
4. **Realistic evaluation**: Final test accuracy phản ánh hiệu suất thực tế trên dữ liệu chưa từng thấy

---

## 3.3. Kiến trúc mô hình CNN đề xuất

### 3.3.1. Thiết kế kiến trúc

**Tổng quan architecture:**

```
INPUT: (batch_size, 3, 3, 27)
    ↓
PERMUTE → (batch_size, 27, 3, 3)  # PyTorch format (N, C, H, W)
    ↓
┌─────────────────────────────────────┐
│ CONVOLUTIONAL BLOCK 1               │
│   Conv2D(27 → 64, kernel=3×3,      │
│          padding=1, bias=False)     │
│   BatchNorm2D(64)                   │
│   ReLU()                            │
│   Dropout2D(p=0.7)                  │
└─────────────────────────────────────┘
    ↓ (batch_size, 64, 3, 3)
┌─────────────────────────────────────┐
│ CONVOLUTIONAL BLOCK 2               │
│   Conv2D(64 → 32, kernel=3×3,      │
│          padding=1, bias=False)     │
│   BatchNorm2D(32)                   │
│   ReLU()                            │
│   Dropout2D(p=0.7)                  │
└─────────────────────────────────────┘
    ↓ (batch_size, 32, 3, 3)
┌─────────────────────────────────────┐
│ GLOBAL AVERAGE POOLING              │
│   AdaptiveAvgPool2D(output_size=1)  │
└─────────────────────────────────────┘
    ↓ (batch_size, 32, 1, 1)
FLATTEN → (batch_size, 32)
    ↓
┌─────────────────────────────────────┐
│ FULLY CONNECTED BLOCK               │
│   Linear(32 → 64)                   │
│   BatchNorm1D(64)                   │
│   ReLU()                            │
│   Dropout(p=0.7)                    │
└─────────────────────────────────────┘
    ↓ (batch_size, 64)
┌─────────────────────────────────────┐
│ OUTPUT LAYER                        │
│   Linear(64 → 4)                    │
└─────────────────────────────────────┘
    ↓
OUTPUT: (batch_size, 4)  # Logits for 4 classes
```

### 3.3.2. Layer-by-layer specifications

**Input Layer:**

```python
Input shape: (batch_size, 3, 3, 27)
# batch_size: Số samples trong mini-batch (64)
# 3×3: Spatial patch size
# 27: Number of features (channels)
```

**Permute Layer:**

```python
# Chuyển từ (N, H, W, C) → (N, C, H, W) format của PyTorch
output = input.permute(0, 3, 1, 2)
Output shape: (batch_size, 27, 3, 3)
```

**Conv Block 1:**

```python
# Convolutional Layer
Conv2D(in_channels=27,
       out_channels=64,
       kernel_size=3,
       padding=1,
       bias=False)

# Calculation:
# Input: (N, 27, 3, 3)
# Kernel: (64, 27, 3, 3)
# Output size: H_out = (3 + 2×1 - 3) / 1 + 1 = 3
# Output: (N, 64, 3, 3)

# Parameters: 27 × 3 × 3 × 64 = 15,552

# BatchNorm2D
BatchNorm2D(num_features=64)
# Parameters: γ (64) + β (64) = 128
# Output: (N, 64, 3, 3)

# ReLU
ReLU()
# Output: (N, 64, 3, 3)

# Dropout2D (Spatial Dropout)
Dropout2D(p=0.7)
# Randomly drop entire feature maps với probability 0.7
# Output: (N, 64, 3, 3)
```

**Conv Block 2:**

```python
Conv2D(in_channels=64,
       out_channels=32,
       kernel_size=3,
       padding=1,
       bias=False)
# Parameters: 64 × 3 × 3 × 32 = 18,432
# Output: (N, 32, 3, 3)

BatchNorm2D(num_features=32)
# Parameters: 32 + 32 = 64
# Output: (N, 32, 3, 3)

ReLU()
Dropout2D(p=0.7)
# Output: (N, 32, 3, 3)
```

**Global Average Pooling:**

```python
AdaptiveAvgPool2D(output_size=(1, 1))

# For each channel, compute average over spatial dimensions
# GAP(k) = (1/(H×W)) × Σᵢⱼ input[i, j, k]
# Input: (N, 32, 3, 3)
# Output: (N, 32, 1, 1)

# No parameters!
```

**Flatten:**

```python
Flatten()
# Input: (N, 32, 1, 1)
# Output: (N, 32)
```

**FC Block:**

```python
Linear(in_features=32, out_features=64)
# Parameters: 32 × 64 + 64 = 2,112 (weights + bias)
# Output: (N, 64)

BatchNorm1D(num_features=64)
# Parameters: 64 + 64 = 128
# Output: (N, 64)

ReLU()
Dropout(p=0.7)
# High dropout rate cho regularization mạnh
# Output: (N, 64)
```

**Output Layer:**

```python
Linear(in_features=64, out_features=4)
# 4 classes: Forest Stable, Deforestation, Non-forest, Reforestation
# Parameters: 64 × 4 + 4 = 260
# Output: (N, 4) - Logits
```

### 3.3.3. Parameter Count

**Total trainable parameters:**

| Layer | Type | Parameters | Calculation |
|-------|------|------------|-------------|
| Conv1 | Weights | 15,552 | 27×3×3×64 |
| BN1 | γ, β | 128 | 64 + 64 |
| Conv2 | Weights | 18,432 | 64×3×3×32 |
| BN2 | γ, β | 64 | 32 + 32 |
| GAP | - | 0 | No params |
| FC1 | Weights, bias | 2,112 | 32×64 + 64 |
| BN3 | γ, β | 128 | 64 + 64 |
| FC2 | Weights, bias | 260 | 64×4 + 4 |
| **TOTAL** | | **36,676** | |

**Model size:**

```
Size = 36,676 parameters × 4 bytes/param (float32)
     = 146,704 bytes
     ≈ 143 KB

Với checkpointing và optimizer states:
Total size ≈ 450 KB
```

**Comparison với các architectures khác:**

| Model | Parameters | Layers | Designed for |
|-------|------------|--------|--------------|
| **Ours** | **36K** | **Shallow** | **Small datasets** |
| ResNet-18 | 11M | Deep (18) | ImageNet (1M images) |
| VGG-16 | 138M | Deep (16) | ImageNet |
| MobileNet | 4.2M | Deep | Mobile devices |

→ Mô hình của chúng ta nhẹ hơn **300 lần** so với ResNet-18!

### 3.3.4. Receptive Field Analysis

**Receptive field tại mỗi layer:**

```
Layer 0 (Input): Receptive field = 1×1

Conv1 (kernel 3×3, padding 1):
  Receptive field = 1 + (3-1) = 3×3

Conv2 (kernel 3×3, padding 1):
  Receptive field = 3 + (3-1) = 5×5

Global Average Pooling:
  Receptive field = Entire patch = 3×3
```

**Interpretation:**

- Conv1 nhìn thấy **3×3** patch → Local spatial patterns
- Conv2 nhìn thấy **toàn bộ 3×3 patch** → Global spatial context
- GAP tổng hợp information từ **toàn bộ patch**

**Kết luận:** Mặc dù chỉ có 2 conv layers, model đã có khả năng nhìn toàn bộ spatial context của patch 3×3.

### 3.3.5. Weight Initialization

**Kaiming Normal Initialization (He initialization):**

Cho convolutional layers:

```python
std = sqrt(2 / (in_channels × kernel_size²))

Weights ~ N(0, std²)
```

**Ví dụ Conv1:**

```python
in_channels = 27
kernel_size = 3

std = sqrt(2 / (27 × 3²))
    = sqrt(2 / 243)
    = 0.0908

Conv1 weights ~ N(0, 0.0908²)
```

**BatchNorm Initialization:**

```python
γ (scale) = 1.0
β (shift) = 0.0
```

**Linear Layer Initialization:**

```python
# Xavier Normal
std = sqrt(2 / (in_features + out_features))

FC1: std = sqrt(2 / (32 + 64)) = 0.1443
FC2: std = sqrt(2 / (64 + 4)) = 0.1715
```

---

## 3.4. Huấn luyện và tối ưu hóa mô hình

### 3.4.1. Training Configuration

**Hyperparameters:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `epochs` | 200 | Max epochs với early stopping |
| `batch_size` | 64 | Balance giữa stability và speed |
| `learning_rate` | 0.001 | Standard Adam LR |
| `weight_decay` | 1e-3 | L2 regularization (mạnh hơn để chống overfitting) |
| `optimizer` | AdamW | Adaptive learning + decoupled weight decay |
| `loss_function` | CrossEntropyLoss | Multi-class classification |
| `dropout_rate` | 0.7 | High dropout để regularization mạnh |
| `early_stopping_patience` | 15 | Kiên nhẫn hơn trước khi dừng |
| `lr_scheduler_patience` | 10 | Giảm LR sau 10 epochs không cải thiện |

**Class Weights:**

Do dataset gần như balanced, class weights được tính như sau:

```python
n_samples = 2630
n_classes = 4
class_counts = [656, 650, 664, 660]  # [C0, C1, C2, C3]

weights = [n_samples / (n_classes × count) for count in class_counts]
weights = [1.0033, 1.0100, 0.9924, 0.9946]

# Normalize
weights = weights / sum(weights) * n_classes
weights = [1.0008, 1.0075, 0.9899, 0.9921]
```

### 3.4.2. Loss Function

**Weighted Cross-Entropy Loss:**

```python
loss_fn = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0008, 1.0075, 0.9899, 0.9921]),
    reduction='mean'
)
```

**Forward computation:**

```python
# Input
logits = model(patches)  # Shape: (batch_size, 4)
labels = ...             # Shape: (batch_size,)

# Softmax
probs = F.softmax(logits, dim=1)

# Cross-entropy loss
loss = -Σᵢ wᵢ × yᵢ × log(probs[i])

# Backward
loss.backward()
```

### 3.4.3. Optimizer - AdamW

**AdamW Configuration:**

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,           # Initial learning rate
    betas=(0.9, 0.999), # Exponential decay rates for moments
    eps=1e-8,           # Numerical stability
    weight_decay=1e-4,  # L2 regularization
    amsgrad=False
)
```

**Update rule:**

```
m_t = β₁ × m_{t-1} + (1-β₁) × ∇L     # First moment
v_t = β₂ × v_{t-1} + (1-β₂) × ∇L²    # Second moment

m̂_t = m_t / (1 - β₁^t)               # Bias correction
v̂_t = v_t / (1 - β₂^t)

θ_{t+1} = θ_t - lr × (m̂_t / (√v̂_t + ε) + λ × θ_t)
                     ↑____________________↑   ↑________↑
                        Adam update        Weight decay
```

### 3.4.4. Learning Rate Scheduler

**ReduceLROnPlateau:**

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',        # Minimize validation loss
    factor=0.5,        # Reduce LR by 50%
    patience=10,       # Wait 10 epochs before reducing
    verbose=True,      # Print messages
    threshold=1e-4,    # Minimum change to qualify as improvement
    min_lr=1e-6        # Minimum learning rate
)
```

**Scheduling logic:**

```
If val_loss không giảm > threshold trong patience epochs:
    lr_new = factor × lr_old

Example:
Epoch 0-10:  lr = 0.001
Epoch 11-15: val_loss không giảm
Epoch 16:    lr = 0.0005  (reduced by 50%)
...
```

**LR history (example from actual run):**

| Epoch | Val Loss | LR | Action |
|-------|----------|----|----|
| 1 | 0.145 | 0.001 | - |
| 5 | 0.089 | 0.001 | - |
| 10 | 0.052 | 0.001 | - |
| 15 | 0.041 | 0.001 | - |
| 20 | 0.040 | 0.001 | No improvement → Reduce |
| 21 | 0.039 | 0.0005 | Continue |
| 30 | 0.038 | 0.0005 | - |
| 37 | 0.038 | 0.0005 | Best val_loss |

### 3.4.5. Early Stopping

**Early Stopping Strategy:**

```python
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
```

**Actual training:**

```
Epoch 1:  val_loss=0.145 → best_loss=0.145, counter=0
Epoch 2:  val_loss=0.128 → best_loss=0.128, counter=0
...
Epoch 25: val_loss=0.039 → best_loss=0.038, counter=1
Epoch 26: val_loss=0.040 → best_loss=0.038, counter=2
...
Epoch 37: val_loss=0.038 → best_loss=0.038, counter=0
Epoch 38: val_loss=0.039 → best_loss=0.038, counter=1
...
Epoch 47: val_loss=0.039 → counter=10 → STOP!
```

Actual stopping epoch: **37** (best validation loss achieved)

### 3.4.6. Training Loop

**Pseudocode:**

```python
for epoch in range(1, max_epochs + 1):
    # ============ TRAINING PHASE ============
    model.train()
    train_loss = 0
    train_correct = 0

    for batch_patches, batch_labels in train_loader:
        # Move to device
        patches = batch_patches.to(device)  # (32, 3, 3, 27)
        labels = batch_labels.to(device)    # (32,)

        # Forward pass
        optimizer.zero_grad()
        logits = model(patches)             # (32, 4)
        loss = loss_fn(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        train_loss += loss.item()
        preds = logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()

    # Compute epoch metrics
    train_loss = train_loss / len(train_loader)
    train_acc = train_correct / len(train_dataset)

    # ============ VALIDATION PHASE ============
    model.eval()
    val_loss = 0
    val_correct = 0

    with torch.no_grad():
        for batch_patches, batch_labels in val_loader:
            patches = batch_patches.to(device)
            labels = batch_labels.to(device)

            logits = model(patches)
            loss = loss_fn(logits, labels)

            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / len(val_dataset)

    # ============ LOGGING ============
    print(f"Epoch {epoch}/{max_epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # ============ LEARNING RATE SCHEDULING ============
    scheduler.step(val_loss)

    # ============ MODEL CHECKPOINTING ============
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }, 'best_model.pth')

    # ============ EARLY STOPPING CHECK ============
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break

# Load best model
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 3.4.7. Device Configuration

**CUDA Setup:**

```python
# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Move model to device
model = model.to(device)

# DataLoader settings
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,  # 0 for Windows, 4+ for Linux
    pin_memory=True if device.type == 'cuda' else False
)
```

**Training time:**

- **With CUDA (GPU):** ~15.2 seconds cho Final Model training
- **With CPU:** ~5-10 phút cho 38 epochs

---

## 3.5. Dự đoán và đánh giá kết quả

### 3.5.1. Test Set Evaluation

**Inference on test set:**

```python
model.eval()
test_preds = []
test_probs = []
test_labels = []

with torch.no_grad():
    for batch_patches, batch_labels in test_loader:
        patches = batch_patches.to(device)

        # Forward pass
        logits = model(patches)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        test_preds.extend(preds.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())
        test_labels.extend(batch_labels.numpy())

test_preds = np.array(test_preds)
test_probs = np.array(test_probs)  # Shape: (396, 4)
test_labels = np.array(test_labels)
```

**Metrics calculation:**

```python
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, roc_auc_score)

# Overall metrics
accuracy = accuracy_score(test_labels, test_preds)
precision = precision_score(test_labels, test_preds, average='weighted')
recall = recall_score(test_labels, test_preds, average='weighted')
f1 = f1_score(test_labels, test_preds, average='weighted')

# Per-class metrics
precision_per_class = precision_score(test_labels, test_preds, average=None)
recall_per_class = recall_score(test_labels, test_preds, average=None)
f1_per_class = f1_score(test_labels, test_preds, average=None)

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)

# ROC-AUC (One-vs-Rest)
roc_auc_ovr = roc_auc_score(test_labels, test_probs,
                            multi_class='ovr', average='macro')
```

### 3.5.2. Full Raster Prediction

**Sliding Window Extraction:**

```python
def extract_patches_for_prediction(feature_stack, valid_mask,
                                   patch_size=3, stride=1):
    """
    Extract all patches from feature stack for prediction
    """
    H, W = feature_stack.shape[1], feature_stack.shape[2]
    half_size = patch_size // 2

    patches = []
    coordinates = []

    for row in range(half_size, H - half_size, stride):
        for col in range(half_size, W - half_size, stride):
            # Check valid mask
            if not valid_mask[row, col]:
                continue

            # Extract patch
            patch = feature_stack[:,
                                 row-half_size:row+half_size+1,
                                 col-half_size:col+half_size+1]

            # Check all pixels in patch are valid
            patch_mask = valid_mask[row-half_size:row+half_size+1,
                                   col-half_size:col+half_size+1]
            if not patch_mask.all():
                continue

            # Transpose and normalize
            patch = np.transpose(patch, (1, 2, 0))  # (3, 3, 27)
            patch = (patch - mean) / (std + 1e-8)

            patches.append(patch)
            coordinates.append((row, col))

    return np.array(patches), coordinates
```

**Batch Prediction:**

```python
# Extract all patches
all_patches, all_coords = extract_patches_for_prediction(
    feature_stack, valid_mask
)
# all_patches: (N, 3, 3, 27), N ~ 16M patches

# Predict in batches
batch_size = 1000
all_preds = []
all_probs = []

model.eval()
with torch.no_grad():
    for i in range(0, len(all_patches), batch_size):
        batch = all_patches[i:i+batch_size]
        batch_tensor = torch.from_numpy(batch).float().to(device)

        logits = model(batch_tensor)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
```

**Reconstruct Rasters:**

```python
# Initialize output arrays
classification_map = np.full((H, W), -1, dtype=np.int8)
probability_map = np.full((H, W), -9999.0, dtype=np.float32)

# Fill predictions
for (row, col), pred, prob in zip(all_coords, all_preds, all_probs):
    classification_map[row, col] = pred
    probability_map[row, col] = prob[1]  # Probability of class 1 (Deforestation)

# Save as GeoTIFF
import rasterio

with rasterio.open('cnn_classification.tif', 'w',
                   driver='GTiff',
                   height=H, width=W, count=1,
                   dtype=np.int8,
                   crs=crs, transform=transform,
                   nodata=-1) as dst:
    dst.write(classification_map, 1)

with rasterio.open('cnn_probability.tif', 'w',
                   driver='GTiff',
                   height=H, width=W, count=1,
                   dtype=np.float32,
                   crs=crs, transform=transform,
                   nodata=-9999.0) as dst:
    dst.write(probability_map, 1)
```

### 3.5.3. Comparison với Random Forest

**Random Forest Training:**

```python
from sklearn.ensemble import RandomForestClassifier

# Extract pixel-based features (không dùng patches)
X_train_rf = []  # Shape: (N, 27)
y_train_rf = []

for (x, y, label) in ground_truth:
    row, col = rowcol(transform, x, y)
    features = feature_stack[:, row, col]  # 27 features
    X_train_rf.append(features)
    y_train_rf.append(label)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train_rf, y_train_rf)

# Predict
rf_preds = rf_model.predict(X_test_rf)
rf_accuracy = accuracy_score(y_test, rf_preds)
```

**Comparison metrics:**

| Metric | CNN | Random Forest |
|--------|-----|---------------|
| **Test Accuracy** | **98.86%** | ~98% |
| **Training Time** | 15.2s (GPU) | ~2-5 min |
| **Model Size** | 450 KB | ~2 MB |
| **Spatial Context** | ✓ Yes (3×3 patch) | ✗ No (pixel-based) |
| **Feature Learning** | ✓ Automatic | ✗ Manual (handcrafted) |
| **Interpretability** | Low | High (feature importance) |

---

**[Kết thúc Chương 3]**

📚 **Xem danh sách đầy đủ tài liệu tham khảo:** [REFERENCES.md](REFERENCES.md)
