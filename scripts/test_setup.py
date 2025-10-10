import torch
import sys
import os

print("="*70)
print("🎯 KIỂM TRA MÔI TRƯỜNG - DỰ ÁN PHÁT HIỆN MẤT RỪNG CÀ MAU")
print("="*70)

# 1. Python & System
print(f"\n📌 Python: {sys.version.split()[0]}")
print(f"📌 Working Directory: {os.getcwd()}")

# 2. PyTorch & GPU
print(f"\n🔥 PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✅ VRAM: {vram:.1f} GB")
    
    # Test tensor 18 channels
    x = torch.randn(2, 18, 128, 128).cuda()
    print(f"✅ Test tensor (2, 18, 128, 128) on GPU: OK")
    del x
    torch.cuda.empty_cache()

# 3. Core Libraries
print(f"\n📦 Core Libraries:")
import numpy as np
print(f"   ✅ NumPy: {np.__version__}")

import cv2
print(f"   ✅ OpenCV: {cv2.__version__}")

# 4. OpenMMLab Ecosystem
print(f"\n🌐 OpenMMLab Ecosystem:")
import mmengine
print(f"   ✅ mmengine: {mmengine.__version__}")

import mmcv
print(f"   ✅ mmcv: {mmcv.__version__}")

import mmdet
print(f"   ✅ mmdet: {mmdet.__version__}")

import mmseg
print(f"   ✅ mmseg: {mmseg.__version__}")

import mmpretrain
print(f"   ✅ mmpretrain: {mmpretrain.__version__}")

# 5. Open-CD
print(f"\n🎯 Open-CD Framework:")
import opencd
print(f"   ✅ Open-CD: {opencd.__version__}")

from opencd.models import SNUNet_ECAM
print(f"   ✅ SNUNet_ECAM model: Available")

# 6. Check directories
print(f"\n📁 Project Structure:")
dirs = ['data', 'configs', 'scripts', 'work_dirs', 'notebooks', 'results', 'open-cd']
for d in dirs:
    exists = "✅" if os.path.exists(d) else "❌ MISSING"
    print(f"   {exists} {d}/")

print("\n" + "="*70)
print("🎉 OPEN-CD FRAMEWORK - SETUP HOÀN TẤT!")
print("="*70)
print("\n📋 Bước tiếp theo:")
print("   1. Cài GDAL + Rasterio (xử lý GeoTIFF)")
print("   2. Cài các thư viện Sentinel (sentinelsat)")
print("   3. Download dữ liệu Sentinel-2 & Sentinel-1")
print("="*70)