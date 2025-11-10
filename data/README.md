# DATA DIRECTORY

Thư mục này chứa dữ liệu đầu vào cho dự án phát hiện phá rừng.

## 📁 Cấu trúc thư mục

```
data/
├── raw/                           # Dữ liệu gốc
│   ├── sentinel-1/                # Ảnh SAR Sentinel-1
│   │   ├── S1_2024_02_04_matched_S2_2024_01_30.tif
│   │   └── S1_2025_02_22_matched_S2_2025_02_28.tif
│   │
│   ├── sentinel-2/                # Ảnh quang học Sentinel-2
│   │   ├── S2_2024_01_30.tif
│   │   └── S2_2025_02_28.tif
│   │
│   ├── ground_truth/              # Ground truth points
│   │   └── ca_mau_points.csv
│   │
│   └── boundary/                  # Ranh giới khu vực nghiên cứu
│       ├── forest_boundary.shp
│       ├── forest_boundary.shx
│       ├── forest_boundary.dbf
│       ├── forest_boundary.prj
│       ├── forest_boundary.cpg
│       └── forest_boundary.qmd
│
├── processed/                     # Dữ liệu đã xử lý (tự động tạo)
└── patches/                       # CNN patches (tự động tạo)
```

## 📥 Hướng dẫn tải dữ liệu

### 1. Dữ liệu Sentinel-1 (SAR)
- **Nguồn:** [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
- **Product Type:** GRD (Ground Range Detected)
- **Polarization:** VV + VH
- **Resolution:** 10m
- **Dates:**
  - Before: 2024-02-04 (matched với S2 2024-01-30)
  - After: 2025-02-22 (matched với S2 2025-02-28)

### 2. Dữ liệu Sentinel-2 (Optical)
- **Nguồn:** [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
- **Product Type:** Level-2A (Surface Reflectance)
- **Bands:** B2, B3, B4, B5, B6, B7, B8A (7 bands)
- **Resolution:** 10m-20m (resample về 10m)
- **Dates:**
  - Before: 2024-01-30
  - After: 2025-02-28

### 3. Ground Truth Points
- **Format:** CSV file
- **Required columns:**
  - `longitude`: Kinh độ (decimal degrees)
  - `latitude`: Vĩ độ (decimal degrees)
  - `label`: Nhãn (0 = không phá rừng, 1 = phá rừng)
- **Hệ tọa độ:** WGS84 (EPSG:4326)

### 4. Boundary Shapefile
- **Format:** Shapefile (.shp + sidecar files)
- **Geometry:** Polygon
- **Hệ tọa độ:** WGS84 (EPSG:4326)
- **Mô tả:** Ranh giới khu vực nghiên cứu (rừng Cà Mau)

## ⚠️ Lưu ý quan trọng

1. **File size:** Các file ảnh vệ tinh (.tif) rất lớn (nhiều GB), do đó:
   - Không được commit lên GitHub
   - Đã được ignore trong `.gitignore`
   - Chỉ commit các file `.gitkeep` để giữ cấu trúc thư mục

2. **Boundary files:** Các file shapefile trong `boundary/` được commit lên GitHub vì:
   - Kích thước nhỏ (< 100MB)
   - Cần thiết để chạy code
   - Không thay đổi thường xuyên

3. **Ground truth CSV:**
   - File CSV được ignore vì có thể chứa thông tin nhạy cảm
   - Cần tạo file `.gitkeep` để giữ cấu trúc thư mục
   - Người dùng cần tự chuẩn bị file CSV theo format

## 🔧 Chuẩn bị dữ liệu

### Bước 1: Tạo cấu trúc thư mục
```bash
cd data/raw
mkdir -p sentinel-1 sentinel-2 ground_truth boundary
```

### Bước 2: Download Sentinel-1 & Sentinel-2
1. Truy cập [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
2. Tìm kiếm ảnh theo:
   - Khu vực: Cà Mau, Vietnam
   - Ngày tháng: Xem phần "Dates" ở trên
   - Product type: Xem phần "Product Type" ở trên
3. Download và đặt vào thư mục tương ứng

### Bước 3: Chuẩn bị Ground Truth
1. Tạo file CSV với format:
```csv
longitude,latitude,label
105.123456,8.654321,1
105.234567,8.765432,0
...
```
2. Lưu vào `data/raw/ground_truth/ca_mau_points.csv`

### Bước 4: Tạo Boundary Shapefile
1. Sử dụng QGIS để vẽ polygon ranh giới khu vực nghiên cứu
2. Export sang Shapefile format
3. Lưu vào `data/raw/boundary/forest_boundary.shp`

## ✅ Kiểm tra dữ liệu

Chạy script kiểm tra:
```bash
cd src
python -c "from common.config import verify_input_files; verify_input_files()"
```

Script sẽ kiểm tra:
- ✓ Tất cả file input tồn tại
- ✓ Format file đúng
- ✓ Hệ tọa độ phù hợp
- ✓ Kích thước và resolution

## 📊 Metadata

- **Study Area:** Cà Mau Province, Vietnam
- **Time Period:** 2024-01-30 to 2025-02-28 (~13 months)
- **Ground Truth Points:** ~1300 points (from AIO dataset)
- **Forest Type:** Mangrove forest
- **CRS:** EPSG:4326 (WGS84)

---

**Last Updated:** 2025-11-10
