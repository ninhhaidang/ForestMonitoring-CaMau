"""
Quick check for Open-CD installation
"""
import sys
import os

print("=" * 80)
print("CHECKING OPEN-CD INSTALLATION")
print("=" * 80)

# Step 1: Check Python path
print("\n[1/6] Checking Python environment...")
print(f"  Python: {sys.version}")
print(f"  Python path: {sys.executable}")

# Step 2: Check if open-cd in path
print("\n[2/6] Checking if open-cd is in Python path...")
sys.path.insert(0, 'open-cd')
print(f"  Added open-cd to path: OK")

# Step 3: Try importing Open-CD
print("\n[3/6] Importing Open-CD modules...")
try:
    import opencd
    print(f"  [OK] opencd imported")

    from opencd.registry import DATASETS, MODELS, TRANSFORMS
    print(f"  [OK] Registries imported")

    # Import to register all modules
    import opencd.datasets
    import opencd.datasets.transforms
    import opencd.models

    print(f"  [OK] All modules imported")
except ImportError as e:
    print(f"  [ERROR] {e}")
    print("\n  Open-CD not installed properly!")
    print("  Please run: cd open-cd && pip install -v -e .")
    sys.exit(1)

# Step 4: Check registered components
print("\n[4/6] Checking registered components...")
print(f"  Datasets: {len(DATASETS)} registered")
print(f"  Models: {len(MODELS)} registered")
print(f"  Transforms: {len(TRANSFORMS)} registered")

# Step 5: Check required transforms
print("\n[5/6] Checking required transforms...")
required_transforms = [
    'MultiImgLoadImageFromFile',
    'MultiImgLoadAnnotations',
    'MultiImgRandomRotate',
    'MultiImgRandomFlip',
    'MultiImgPhotoMetricDistortion',
    'MultiImgPackSegInputs'
]

all_ok = True
for transform in required_transforms:
    if transform in TRANSFORMS:
        print(f"  [OK] {transform}")
    else:
        print(f"  [MISSING] {transform}")
        all_ok = False

# Step 6: Check required datasets
print("\n[6/6] Checking required datasets...")
required_datasets = ['LEVIR_CD_Dataset']

for dataset in required_datasets:
    if dataset in DATASETS:
        print(f"  [OK] {dataset}")
    else:
        print(f"  [MISSING] {dataset}")
        all_ok = False

print("\n" + "=" * 80)
if all_ok:
    print("[SUCCESS] Open-CD installation is complete and ready!")
    print("=" * 80)
    print("\nYou can now run training with:")
    print("  python open-cd/tools/train.py configs/tinycdv2_camau.py")
else:
    print("[WARNING] Some components are missing!")
    print("=" * 80)
    print("\nPlease reinstall Open-CD:")
    print("  cd open-cd && pip install -v -e .")
    sys.exit(1)
