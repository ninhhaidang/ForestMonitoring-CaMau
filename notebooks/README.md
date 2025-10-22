# 📓 Jupyter Notebooks

Thư mục này chứa các Jupyter notebooks để khám phá dữ liệu, phân tích và trực quan hóa kết quả.

---

## 📋 Danh Sách Notebooks

### ✅ `01_data_exploration.ipynb`
**Trạng thái:** Đã tạo
**Mục đích:** Kiểm tra metadata và khám phá dữ liệu Sentinel-1/2

**Nội dung:**
- Import libraries và setup
- Kiểm tra metadata (CRS, resolution, bounds, data type)
- Phân tích statistics (min, max, mean, std, NaN%)
- Visualization:
  - NaN percentage comparison
  - Band mean value comparison
  - Vegetation indices 2024 vs 2025
  - Sample band images
- Summary report

**Outputs:**
- `data/metadata_summary.csv`
- `figures/band_nan_comparison.png`
- `figures/band_mean_comparison.png`
- `figures/indices_2024_vs_2025.png`
- `figures/sample_band_images.png`

---

### ⬜ `02_training_analysis.ipynb` (Chưa tạo)
**Mục đích:** Phân tích quá trình training của 3 models

**Nội dung dự kiến:**
- Load training logs
- Plot loss curves (training vs validation)
- Plot accuracy/F1-score over epochs
- Compare 3 models side-by-side
- Analyze overfitting/underfitting
- Learning rate schedule visualization

---

### ⬜ `03_results_visualization.ipynb` (Chưa tạo)
**Mục đích:** Trực quan hóa kết quả dự đoán

**Nội dung dự kiến:**
- Load trained models
- Predict on test set
- Confusion matrices
- ROC curves
- Sample predictions visualization
- Error analysis
- Full-image probability maps

---

## 🚀 Quick Start

### 1. Kích hoạt môi trường Conda

```bash
conda activate dang
```

### 2. Khởi động JupyterLab

```bash
cd D:\HaiDang\25-26_HKI_DATN_21021411_DangNH
jupyter lab
```

### 3. Mở notebook

Trong JupyterLab, navigate đến `notebooks/` và mở `01_data_exploration.ipynb`

### 4. Run cells

- **Run all:** Kernel → Restart Kernel and Run All Cells
- **Run individual:** Shift + Enter

---

## 📦 Dependencies

Tất cả dependencies đã được cài đặt trong môi trường `dang`:

- `rasterio` - Đọc/ghi GeoTIFF
- `numpy` - Tính toán số học
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `pandas` - Data manipulation
- `jupyter` / `jupyterlab` - Notebook environment

---

## 💡 Tips

### Thay đổi Figure Size
```python
plt.figure(figsize=(12, 8))
```

### Save Figure với DPI cao
```python
plt.savefig('output.png', dpi=300, bbox_inches='tight')
```

### Reload Module khi đang phát triển
```python
import importlib
import my_module
importlib.reload(my_module)
```

### Memory Management với Large Rasters
```python
# Đọc theo window thay vì load toàn bộ
with rasterio.open(file) as src:
    window = rasterio.windows.Window(0, 0, 1000, 1000)
    data = src.read(1, window=window)
```

---

## ⚠️ Lưu Ý

1. **Memory:** Notebooks có thể dùng nhiều RAM khi load ảnh TIFF lớn. Khuyến nghị ≥16GB RAM.

2. **Paths:** Sử dụng relative paths từ thư mục notebooks:
   ```python
   data_path = Path("../data/raw")
   ```

3. **Kernels:** Đảm bảo chọn đúng kernel `dang` trong JupyterLab:
   - Kernel → Change Kernel → dang

4. **Git:** Notebooks không được commit vào git (đã ignore trong `.gitignore`). Chỉ commit code Python trong `src/`.

---

## 📊 Expected Outputs

Sau khi chạy `01_data_exploration.ipynb`:

```
ca-mau-deforestation/
├── data/
│   └── metadata_summary.csv          ← New
└── figures/
    ├── band_nan_comparison.png       ← New
    ├── band_mean_comparison.png      ← New
    ├── indices_2024_vs_2025.png      ← New
    └── sample_band_images.png        ← New
```

---

**Last updated:** 2025-10-22
**Author:** Ninh Hải Đăng
