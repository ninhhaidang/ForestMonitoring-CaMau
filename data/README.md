# Data Directory

## 📁 Cấu trúc

### raw/
Dữ liệu gốc, KHÔNG BAO GIỜ sửa đổi:
- `sentinel2/`: Ảnh Sentinel-2 (T1: 30/01/2024, T2: 28/02/2025)
- `sentinel1/`: Ảnh Sentinel-1 SAR 
- `ground_truth/`: 1285 điểm ground truth

### processed/
Dữ liệu sau xử lý:
- `phase1_s2only/`: 14 channels (S2 only)
- `phase2_s2s1/`: 18 channels (S2 + S1)

### samples/
Training patches (256x256):
- Split: 80% train / 10% val / 10% test
- Format: GeoTIFF với naming convention

## 🔒 Backup
Luôn giữ `raw/` nguyên vẹn. Chỉ xử lý từ `raw/` → `processed/`
