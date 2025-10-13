# Project Overview

## 🌲 Phát Hiện Mất Rừng Cà Mau Sử Dụng Deep Learning

### Thông tin dự án
- **Sinh viên:** Ninh Hải Đăng (21021411)
- **Khóa:** 2021-2025
- **Đồ án:** Tốt nghiệp - Học kỳ I 2025-2026
- **Trường:** Viện Công nghệ Hàng không Vũ trụ - UET - VNU

### Mục tiêu
Sử dụng Deep Learning (SNUNet-CD) để phát hiện mất rừng ngập mặn tại Cà Mau:
- Dữ liệu: Sentinel-2 + Sentinel-1
- Chu kỳ: 2 lần/tháng
- Diện tích: 7,942.39 km²

### Phương pháp
- **Phase 1:** Chỉ dùng Sentinel-2 (14 channels)
- **Phase 2:** Kết hợp Sentinel-2 + Sentinel-1 (18 channels)
- **Model:** SNUNet-CD (Siamese Nested U-Net)
- **Framework:** Open-CD 1.1.0

### Dataset
- 1285 ground truth points (635 loss / 650 no change)
- Split: 80% train / 10% val / 10% test
