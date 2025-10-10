import sys
import torch
import numpy as np
import os

print("="*70)
print("🎯 VERIFICATION - DỰ ÁN PHÁT HIỆN MẤT RỪNG CÀ MAU")
print("="*70)

# 1. Python & System
print(f"\n📌 Python: {sys.version.split()[0]}")
if not sys.version.startswith('3.8'):
    print("⚠️  WARNING: Python should be 3.8.x!")

# 2. PyTorch & GPU
print(f"\n🔥 PyTorch: {torch.__version__}")
if not torch.__version__.startswith('1.13.1'):
    print("⚠️  WARNING: PyTorch should be 1.13.1!")
    
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✅ VRAM: {vram:.1f} GB")

# 3. NumPy
print(f"\n📦 NumPy: {np.__version__}")
if np.__version__.startswith('2.'):
    print("⚠️  WARNING: NumPy should be <2.0!")

# 4. Test GPU tensor
try:
    x = torch.randn(2, 18, 128, 128).cuda()
    print(f"✅ Test tensor (2, 18, 128, 128) on GPU: OK")
    del x
    torch.cuda.empty_cache()
except Exception as e:
    print(f"❌ GPU test failed: {e}")

# 5. OpenMMLab Ecosystem
print(f"\n🌐 OpenMMLab Ecosystem:")
try:
    import mmengine
    print(f"   ✅ mmengine: {mmengine.__version__}")
    if mmengine.__version__ != '0.10.4':
        print(f"   ⚠️  Expected: 0.10.4")
except Exception as e:
    print(f"   ❌ mmengine: {e}")

try:
    import mmcv
    print(f"   ✅ mmcv: {mmcv.__version__}")
    if mmcv.__version__ != '2.1.0':
        print(f"   ⚠️  Expected: 2.1.0")
except Exception as e:
    print(f"   ❌ mmcv: {e}")

try:
    import mmdet
    print(f"   ✅ mmdet: {mmdet.__version__}")
    if mmdet.__version__ != '3.3.0':
        print(f"   ⚠️  Expected: 3.3.0")
except Exception as e:
    print(f"   ❌ mmdet: {e}")

try:
    import mmseg
    print(f"   ✅ mmseg: {mmseg.__version__}")
    if mmseg.__version__ != '1.2.2':
        print(f"   ⚠️  Expected: 1.2.2")
except Exception as e:
    print(f"   ❌ mmseg: {e}")

try:
    import mmpretrain
    print(f"   ✅ mmpretrain: {mmpretrain.__version__}")
    if mmpretrain.__version__ != '1.2.0':
        print(f"   ⚠️  Expected: 1.2.0")
except Exception as e:
    print(f"   ❌ mmpretrain: {e}")

# 6. Open-CD
print(f"\n🎯 Open-CD:")
try:
    import opencd
    print(f"   ✅ Open-CD: {opencd.__version__}")
    
    from opencd.models import SNUNet_ECAM
    print(f"   ✅ SNUNet model: Available")
except Exception as e:
    print(f"   ❌ Open-CD failed: {e}")

# 7. Data Processing
print(f"\n📊 Data Processing:")
try:
    import cv2
    print(f"   ✅ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"   ⚠️  OpenCV: Not installed yet")

try:
    import rasterio
    print(f"   ✅ Rasterio: {rasterio.__version__}")
except Exception as e:
    print(f"   ⚠️  Rasterio: Not installed yet")

try:
    from osgeo import gdal
    print(f"   ✅ GDAL: {gdal.__version__}")
except Exception as e:
    print(f"   ⚠️  GDAL: Not installed yet")

# 8. Jupyter
print(f"\n📓 Jupyter:")
try:
    import jupyter
    import notebook
    import jupyterlab
    print(f"   ✅ Jupyter: Installed")
    print(f"   ✅ Notebook: {notebook.__version__}")
    print(f"   ✅ JupyterLab: {jupyterlab.__version__}")
except Exception as e:
    print(f"   ❌ Jupyter: {e}")

# 9. Directories
print(f"\n📁 Project Structure:")
dirs = ['data', 'data/sentinel2', 'data/sentinel1', 'data/labels', 
        'configs', 'scripts', 'work_dirs', 'notebooks', 'results', 'open-cd']
for d in dirs:
    exists = "✅" if os.path.exists(d) else "❌"
    print(f"   {exists} {d}/")

print("\n" + "="*70)
print("🎉 VERIFICATION COMPLETE!")
print("="*70)
print("\n📋 Next Steps:")
print("   1. Cài GDAL + Rasterio (cho xử lý Sentinel data)")
print("   2. Tạo cấu trúc thư mục data")
print("   3. Download Sentinel-2 data (30/1/2024 & 28/2/2025)")
print("   4. Download Sentinel-1 data (cùng thời điểm)")
print("="*70)