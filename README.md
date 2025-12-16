# Ứng dụng viễn thám và học sâu trong giám sát biến động rừng tỉnh Cà Mau

Giám sát biến động rừng Cà Mau sử dụng dữ liệu Sentinel-1 & Sentinel-2 kết hợp mô hình CNN. Phân loại 4 lớp: Rừng ổn định, Mất rừng, Phi rừng, Phục hồi rừng.

**[DEMO](https://ee-bonglantrungmuoi.projects.earthengine.app/view/giam-sat-bien-dong-rung-ca-mau)**


## 📂 Cấu trúc dự án

```
├── data/          # Dữ liệu thô & ground truth
├── notebooks/     # Jupyter notebooks
├── src/           # Mã nguồn mô hình
└── results/       # Kết quả
```

## 📊 Dữ liệu

### Nguồn dữ liệu
- **Sentinel-2 (Optical):** 7 bands (Red, NIR, SWIR, NDVI, NBR, NDMI)
- **Sentinel-1 (SAR):** VV, VH polarization
- **Thời kỳ:** 2 kỳ ảnh (1/2024 và 2/2025)
- **Samples:** 2,630 điểm

## 🧠 Mô hình CNN

### Kiến trúc
- **Input:** 3×3×27 patches
- **Conv layers:** 2 blocks với BatchNorm + ReLU + Dropout
- **Global Average Pooling**
- **Output:** 4 classes
- **Tổng tham số:** ~36k (lightweight model)

### Đặc điểm
- Dropout cao để chống overfitting
- BatchNorm cho training ổn định
- AdamW optimizer với weight decay
- Early stopping & learning rate scheduling


## 🎯 Kết quả

- Mô hình với Test Accuracy ~98.86%
- Bản đồ phân loại biến động rừng độ phân giải 10m


## 👤 Tác giả

**[Ninh Hải Đăng](https://ninhhaidang.github.io)**

*Viện Công nghệ Hàng không Vũ trụ - Trường Đại học Công nghệ - ĐHQG Hà Nội*

> Dự án phát triển cho mục đích nghiên cứu & học thuật.

**Last updated:** December 2025