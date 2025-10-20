# 🌳 Forest Change Detection - Ca Mau Mangrove

**Phát hiện mất rừng ngập mặn Cà Mau sử dụng Deep Learning với dữ liệu đa nguồn vệ tinh**

**Sinh viên**: Ninh Hải Đăng (MSSV: 21021411)
**Trường**: Đại học Công nghệ - ĐHQGHN
**Năm học**: 2025-2026

---

## 📋 MỤC ĐÍCH DỰ ÁN

### Vấn đề:
Phát hiện và lập bản đồ **mất rừng ngập mặn** tại khu vực Cà Mau trong giai đoạn 2024-2025 bằng phương pháp học sâu (Deep Learning).

### Giải pháp:
Sử dụng **3 mô hình Deep Learning nhẹ** (PyTorch) để phân loại từng pixel trên ảnh vệ tinh:
- **Pixel = 0**: Không mất rừng (rừng còn nguyên vẹn)
- **Pixel = 1**: Mất rừng (phá rừng/chuyển đổi đất)

### Đầu vào (INPUT):
1. **4 ảnh vệ tinh GeoTIFF** (toàn bộ vùng Cà Mau):
   - Sentinel-1 Time 1 (2024-02-04): 2 bands (VH, VH/VV Ratio)
   - Sentinel-1 Time 2 (2025-02-22): 2 bands (VH, VH/VV Ratio)
   - Sentinel-2 Time 1 (2024-01-30): 7 bands (B4, B8, B11, B12, NDVI, NBR, NDMI)
   - Sentinel-2 Time 2 (2025-02-28): 7 bands (B4, B8, B11, B12, NDVI, NBR, NDMI)

2. **1 file CSV** với 1,285 điểm training (tọa độ x, y + nhãn):
   - 650 điểm "không mất rừng" (label = 0)
   - 635 điểm "mất rừng" (label = 1)

### Đầu ra (OUTPUT):
1. **probability_map.tif** - Bản đồ xác suất mất rừng (giá trị 0.0 → 1.0)
2. **binary_map.tif** - Bản đồ phân loại nhị phân (0 = không mất, 1 = mất rừng)
3. **visualization.png** - Bản đồ màu (Xanh lá = không mất, Đỏ = mất rừng)

---

## 🔬 PHƯƠNG PHÁP

### 1. Chuẩn bị dữ liệu:
- Extract patches 256×256 pixels tại tọa độ (x,y) từ CSV
- Mỗi patch chứa **18 bands tổng cộng**:
  - Time 1: 9 bands (2 S1 + 7 S2)
  - Time 2: 9 bands (2 S1 + 7 S2)
- Split: 80% train (1,028), 10% val (128), 10% test (129)

### 2. Training:
So sánh **3 mô hình Deep Learning nhẹ** từ thư viện `segmentation_models_pytorch`:

| Mô hình | Encoder | Params | Tốc độ | Đặc điểm |
|---------|---------|--------|--------|----------|
| **UNet-EfficientNet-B0** | EfficientNet-B0 | ~5M | Nhanh | ⭐ Cân bằng tốt |
| **UNet-MobileNetV2** | MobileNetV2 | ~2M | Rất nhanh | Nhẹ nhất, phù hợp mobile |
| **FPN-EfficientNet-B0** | EfficientNet-B0 | ~6M | Trung bình | Accuracy cao nhất |

**Training config:**
- Loss: CrossEntropyLoss (binary classification)
- Optimizer: AdamW
- Learning rate: 1e-4
- Batch size: 16
- Epochs: 50 (với early stopping)
- Augmentation: Random flip, rotation

### 3. Inference (Whole Scene):
- **Sliding window 256×256** với overlap trên toàn bộ 4 ảnh GeoTIFF gốc
- Merge predictions từ tất cả windows → Probability map (0.0-1.0)
- Apply threshold (0.5) → Binary map (0/1)

### 4. Tạo bản đồ cuối cùng:
- Save probability map dạng GeoTIFF (float32)
- Save binary map dạng GeoTIFF (uint8)
- Colorize và export PNG (visualization)

---

## 📁 CẤU TRÚC DỮ LIỆU

```
project/
│
├── 📂 data/
│   └── raw/                          # Data gốc
│       ├── sentinel1/
│       │   ├── S1_2024_02_04_matched_S2_2024_01_30.tif  (490MB)
│       │   └── S1_2025_02_22_matched_S2_2025_02_28.tif  (489MB)
│       ├── sentinel2/
│       │   ├── S2_2024_01_30.tif                        (1.5GB)
│       │   └── S2_2025_02_28.tif                        (1.5GB)
│       └── ground_truth/
│           └── Training_Points_CSV.csv                  (1,285 points)
│
├── 📂 notebooks/                     # Jupyter notebooks chính
│   ├── 1_train_models.ipynb         # Train 3 models
│   ├── 2_inference_wholescene.ipynb # Whole scene inference
│   └── 3_create_maps.ipynb          # Generate final outputs
│
├── 📂 src/                           # Source code modules
│   ├── dataset.py                    # PyTorch Dataset
│   ├── models.py                     # Model definitions
│   └── utils.py                      # Helper functions
│
├── 📂 models/                        # Saved models
│   ├── unet_efficientnet/
│   │   └── best_model.pth
│   ├── unet_mobilenet/
│   │   └── best_model.pth
│   └── fpn_efficientnet/
│       └── best_model.pth
│
└── 📂 results/                       # Outputs
    └── whole_scene/
        ├── probability_map.tif       # 🎯 Xác suất [0.0-1.0]
        ├── binary_map.tif            # 🎯 Nhị phân [0,1]
        └── visualization.png         # 🎯 Visualization (RGB)
```

---

## 🚀 HƯỚNG DẪN SỬ DỤNG

### Bước 0: Cài đặt môi trường

```bash
# Option 1: Sử dụng Conda (Recommended)
conda env create -f environment.yml
conda activate dang

# Option 2: Sử dụng pip
pip install -r requirements.txt
pip install segmentation-models-pytorch
```

**Yêu cầu:**
- Python 3.8+
- PyTorch 1.13+ (CUDA 11.7+)
- GPU NVIDIA (16GB VRAM khuyến nghị)
- RAM: 32GB
- Disk: ~10GB trống

---

### Bước 1: Train Models

```bash
jupyter notebook notebooks/1_train_models.ipynb
```

**Notebook này sẽ:**
1. Load patches từ CSV coordinates
2. Tạo PyTorch DataLoader (train/val split)
3. Train 3 models với real-time monitoring:
   - Loss/Accuracy curves (live update)
   - Sample predictions visualization
   - Progress bars
4. Save best model checkpoint vào `models/{model_name}/`

**Output:**
- `models/unet_efficientnet/best_model.pth`
- `models/unet_mobilenet/best_model.pth`
- `models/fpn_efficientnet/best_model.pth`
- Training history plots

**Thời gian**: ~30-60 phút/model (GPU)

---

### Bước 2: Inference Whole Scene

```bash
jupyter notebook notebooks/2_inference_wholescene.ipynb
```

**Notebook này sẽ:**
1. Load best model
2. Load 4 ảnh GeoTIFF gốc (toàn bộ vùng)
3. Sliding window 256×256 với overlap
4. Predict từng window
5. Merge predictions → Probability map (numpy array)
6. Visualize progress real-time

**Output:**
- Probability map (numpy array, sẽ save ở bước 3)
- Preview visualization

**Thời gian**: ~10-30 phút (tùy kích thước ảnh)

---

### Bước 3: Create Final Maps

```bash
jupyter notebook notebooks/3_create_maps.ipynb
```

**Notebook này sẽ:**
1. Load probability map từ bước 2
2. Apply threshold (0.5) → Binary map
3. Colorize (0 → Green, 1 → Red)
4. Save 3 outputs dạng GeoTIFF/PNG

**Output:**
- `results/whole_scene/probability_map.tif` (Float32, 0.0-1.0)
- `results/whole_scene/binary_map.tif` (UInt8, 0-1)
- `results/whole_scene/visualization.png` (RGB)

**Thời gian**: ~5 phút

---

## 📊 KẾT QUẢ KỲ VỌNG

### Metrics (Test set - 129 patches):
- **Accuracy**: 85-90%
- **F1-Score**: 0.85-0.90
- **IoU**: 0.75-0.85

### Bản đồ cuối cùng:
- Probability map: Xác suất mất rừng tại mỗi pixel
- Binary map: Phân loại rõ ràng (0/1)
- Visualization: Trực quan, dễ hiểu cho báo cáo

### Statistics ví dụ:
```
Tổng pixels: 50,000,000
Không mất rừng (0): 30,000,000 (60%)
Mất rừng (1): 20,000,000 (40%)
```

---

## 🔧 TECHNICAL DETAILS

### Multi-Sensor Data Fusion:
- **Sentinel-1 (SAR)**: Không bị ảnh hưởng mây, nhạy với cấu trúc thực vật
- **Sentinel-2 (Optical)**: Phổ phản xạ chi tiết, indices thực vật (NDVI, NBR, NDMI)
- **Fusion**: Concat 18 bands → Single input tensor

### Model Architecture:
```python
# UNet-EfficientNet Example
Input: (B, 18, 256, 256)  # 18 bands, 256x256 patch
  ↓
Encoder: EfficientNet-B0 (pretrained on ImageNet, adapted to 18 channels)
  ↓
Decoder: UNet decoder with skip connections
  ↓
Output: (B, 2, 256, 256)  # 2 classes (no change, change)
  ↓
Softmax → Probability map: (B, 256, 256) values in [0.0, 1.0]
```

### Sliding Window Strategy:
```
Window size: 256×256
Overlap: 32 pixels
Step: 224 pixels
Total windows: ~5,000-10,000 (depends on scene size)
```

---

## 📚 THƯ VIỆN SỬ DỤNG

### Core Libraries:
- **PyTorch** (1.13+): Deep learning framework
- **segmentation_models_pytorch**: Pre-built segmentation models
- **rasterio**: Read/write GeoTIFF
- **albumentations**: Data augmentation
- **pandas**: CSV processing
- **matplotlib/seaborn**: Visualization

### Model Library:
```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name='efficientnet-b0',
    encoder_weights='imagenet',
    in_channels=18,
    classes=2
)
```

---

## 🎯 SO SÁNH 3 MODELS

| Tiêu chí | UNet-EfficientNet | UNet-MobileNet | FPN-EfficientNet |
|----------|-------------------|----------------|------------------|
| **Params** | ~5M | ~2M | ~6M |
| **Inference Speed** | ⚡⚡⚡ | ⚡⚡⚡⚡ | ⚡⚡ |
| **Accuracy** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Memory (VRAM)** | ~4GB | ~2GB | ~6GB |
| **Training Time** | ~45 min | ~30 min | ~60 min |
| **Best For** | Cân bằng | Production, Mobile | Highest Accuracy |

---

## ⚠️ LƯU Ý

### 1. Data Location:
- Đảm bảo 4 ảnh TIFF + CSV trong `data/raw/`
- Kiểm tra tọa độ CSV khớp với coordinate system của ảnh

### 2. GPU Memory:
- UNet-MobileNet: OK với GPU 8GB
- UNet-EfficientNet: Cần GPU 12GB
- FPN-EfficientNet: Cần GPU 16GB
- Giảm batch_size nếu bị OOM

### 3. Whole Scene Inference:
- Có thể mất 10-30 phút tùy kích thước ảnh
- Progress bar sẽ hiển thị tiến độ
- Nếu quá lâu, có thể chỉ inference một phần ảnh

---

## 📝 CITATION

```bibtex
@thesis{dang2025forest,
  title={Forest Change Detection in Ca Mau using Multi-Sensor Deep Learning},
  author={Ninh Hải Đăng},
  school={VNU University of Engineering and Technology},
  year={2025},
  type={Bachelor's Thesis}
}
```

---

## 📧 LIÊN HỆ

**Sinh viên**: Ninh Hải Đăng
**MSSV**: 21021411
**Email**: ninhhaidangg@gmail.com
**Trường**: Đại học Công nghệ - ĐHQGHN

---

## 📄 LICENSE

MIT License - Xem file LICENSE

---

## 🙏 ACKNOWLEDGMENTS

- **segmentation_models_pytorch**: https://github.com/qubvel/segmentation_models.pytorch
- **PyTorch**: https://pytorch.org/
- **Sentinel Hub**: Dữ liệu vệ tinh Sentinel-1/2
- **VNU-UET**: Hỗ trợ tài nguyên và hướng dẫn

---

**Last Updated**: 2025-10-18
**Status**: ✅ Ready for development
