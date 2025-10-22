# 📦 Source Code Modules

This directory contains modular Python code that can be imported into Jupyter notebooks or used in scripts.

---

## 📋 Modules Overview

### 1. `utils.py` - Data Loading & Metadata

**Key Functions:**
- `load_tiff(filepath, bands, window)` - Load TIFF files
- `check_tiff_metadata(filepath, verbose)` - Check file metadata
- `get_tiff_stats(filepath, sample_size)` - Get band statistics
- `load_ground_truth(csv_path)` - Load labels CSV
- `coord_to_pixel(geo_x, geo_y, transform)` - Coordinate conversion
- `pixel_to_coord(pixel_x, pixel_y, transform)` - Inverse conversion

**Usage in Notebook:**
```python
from src import load_tiff, check_tiff_metadata, get_tiff_stats

# Load TIFF
data = load_tiff('data/raw/sentinel1/S1_2024.tif', bands=[1, 2])

# Check metadata
meta = check_tiff_metadata('data/raw/sentinel1/S1_2024.tif', verbose=True)

# Get statistics
stats = get_tiff_stats('data/raw/sentinel2/S2_2024.tif', sample_size=1000)
```

---

### 2. `preprocessing.py` - Data Processing

**Key Functions:**
- `normalize_band(data, method, clip_range)` - Normalize bands
  - Methods: 'standardize', 'minmax', 'clip'
- `handle_nan(data, method, fill_value)` - Handle NaN values
  - Methods: 'interpolate', 'fill', 'mean', 'median'
- `extract_patch(data, center_y, center_x, patch_size)` - Extract patches
- `create_patches_dataset(...)` - **Main function** to create full dataset
- `augment_patch(patch, augmentation)` - Data augmentation

**Usage in Notebook:**
```python
from src import normalize_band, handle_nan, extract_patch

# Normalize
norm_data = normalize_band(data, method='standardize')

# Handle NaN
clean_data = handle_nan(data, method='interpolate')

# Extract patch
patch = extract_patch(data, center_y=1000, center_x=2000, patch_size=128)
```

**Create Dataset:**
```python
from src.preprocessing import create_patches_dataset

counts = create_patches_dataset(
    s1_2024_path='data/raw/sentinel1/S1_2024.tif',
    s1_2025_path='data/raw/sentinel1/S1_2025.tif',
    s2_2024_path='data/raw/sentinel2/S2_2024.tif',
    s2_2025_path='data/raw/sentinel2/S2_2025.tif',
    ground_truth_csv='data/raw/ground_truth/Training_Points_CSV.csv',
    output_dir='data/patches',
    patch_size=128,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15
)
```

---

### 3. `visualization.py` - Plotting Functions

**Key Functions:**
- `plot_band(data, title, cmap, save_path)` - Plot single band
- `plot_band_comparison(data_list, titles)` - Compare multiple bands
- `plot_statistics(stats_dict, metric)` - Plot statistics comparison
- `plot_indices_comparison(stats_2024, stats_2025)` - Compare vegetation indices
- `plot_confusion_matrix(y_true, y_pred)` - Confusion matrix
- `plot_training_history(history, metrics)` - Training curves
- `plot_patch_sample(patch, label, band_indices)` - Visualize patch

**Usage in Notebook:**
```python
from src import plot_band, plot_band_comparison, plot_statistics

# Plot single band
plot_band(data, title='S1 VH 2024', cmap='viridis', save_path='figures/s1_vh.png')

# Compare bands
plot_band_comparison([data_2024, data_2025],
                     titles=['2024', '2025'],
                     suptitle='Comparison')

# Plot statistics
plot_statistics({'2024': stats_2024, '2025': stats_2025},
                metric='mean')
```

---

### 4. `models.py` - CNN Architectures

**3 Models Implemented:**

#### Model 1: `SpatialContextCNN` (~30K params)
- Simple 3-layer CNN
- Receptive field: 5×5 pixels (50m)
- Best for: Baseline, fast inference

#### Model 2: `MultiScaleCNN` (~80K params)
- Multi-scale branches (3×3 and 5×5)
- Receptive field: 7×7 and 9×9 pixels
- Best for: **Production** (recommended)

#### Model 3: `ShallowUNet` (~120K params)
- Encoder-decoder with skip connections
- Receptive field: 13×13 pixels (130m)
- Best for: Highest quality, smooth predictions

**Usage in Notebook:**
```python
import torch
from src.models import get_model, count_parameters, print_model_summary

# Get model
model = get_model('shallow_unet', in_channels=18)

# Count parameters
n_params = count_parameters(model)
print(f"Parameters: {n_params:,}")

# Print summary
print_model_summary(model)

# Forward pass
x = torch.randn(1, 18, 128, 128)  # (B, C, H, W)
y = model(x)  # (B, 1, H, W) - probability map
```

---

## 🚀 Quick Start

### Step 1: Add src/ to Python path (in notebook)

```python
import sys
from pathlib import Path

# Add src to path
src_path = Path('../src').resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Now you can import
from src import load_tiff, plot_band, get_model
```

**Or use relative imports:**
```python
# From notebooks/ directory
from src import load_tiff
from src.preprocessing import normalize_band
from src.visualization import plot_band
from src.models import get_model
```

---

### Step 2: Import what you need

```python
# Import specific functions
from src import (
    # Utils
    load_tiff,
    check_tiff_metadata,
    get_tiff_stats,
    load_ground_truth,

    # Preprocessing
    normalize_band,
    handle_nan,
    extract_patch,

    # Visualization
    plot_band,
    plot_band_comparison,
    plot_statistics,

    # Models
    get_model,
    count_parameters
)
```

**Or import entire modules:**
```python
from src import utils
from src import preprocessing
from src import visualization
from src import models

# Use with module prefix
data = utils.load_tiff('file.tif')
model = models.get_model('shallow_unet')
```

---

## 📝 Example Workflow

### Data Exploration (Notebook)

```python
# 1. Load data
from src import load_tiff, check_tiff_metadata, get_tiff_stats

meta = check_tiff_metadata('S1_2024.tif', verbose=True)
stats = get_tiff_stats('S1_2024.tif', sample_size=1000)
data = load_tiff('S1_2024.tif', bands=[1])

# 2. Visualize
from src import plot_band, plot_statistics

plot_band(data, title='S1 VH 2024', cmap='viridis')
plot_statistics({'2024': stats}, metric='mean')
```

### Create Dataset (Notebook or Script)

```python
from src.preprocessing import create_patches_dataset

counts = create_patches_dataset(
    s1_2024_path='data/raw/sentinel1/S1_2024.tif',
    s1_2025_path='data/raw/sentinel1/S1_2025.tif',
    s2_2024_path='data/raw/sentinel2/S2_2024.tif',
    s2_2025_path='data/raw/sentinel2/S2_2025.tif',
    ground_truth_csv='data/raw/ground_truth/Training_Points_CSV.csv',
    output_dir='data/patches',
    patch_size=128
)

print(f"Created {counts['train']} train, {counts['val']} val, {counts['test']} test patches")
```

### Model Testing (Notebook)

```python
import torch
from src.models import get_model, print_model_summary
import numpy as np

# Create model
model = get_model('shallow_unet', in_channels=18)
print_model_summary(model)

# Load a patch
patch = np.load('data/patches/train/train_0001_label1.npy')

# Convert to tensor and add batch dimension
x = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0)  # (1, 18, 128, 128)

# Forward pass
with torch.no_grad():
    y = model(x)  # (1, 1, 128, 128)

# Visualize result
from src import plot_band
plot_band(y[0, 0].numpy(), title='Prediction Probability', cmap='RdYlGn')
```

---

## 🧪 Testing Modules

Each module has a `__main__` block for testing:

```bash
# Test utils
python src/utils.py

# Test preprocessing
python src/preprocessing.py

# Test visualization
python src/visualization.py

# Test models
python src/models.py
```

---

## 📖 Documentation

Each function has detailed docstrings:

```python
from src import load_tiff

# View docstring
help(load_tiff)

# In Jupyter:
load_tiff?
```

---

## 💡 Best Practices

### 1. Import at notebook top
```python
# Cell 1: Imports
from src import (
    load_tiff,
    check_tiff_metadata,
    plot_band,
    get_model
)
```

### 2. Use relative paths
```python
from pathlib import Path

base_dir = Path('../data/raw')
s1_file = base_dir / 'sentinel1' / 'S1_2024.tif'
data = load_tiff(s1_file)
```

### 3. Reload module during development
```python
import importlib
from src import utils

# After modifying utils.py
importlib.reload(utils)
```

---

## 📦 Module Structure

```
src/
├── __init__.py              # Package initialization, exports key functions
├── utils.py                 # Data loading & utilities (~300 lines)
├── preprocessing.py         # Data preprocessing (~400 lines)
├── visualization.py         # Plotting functions (~500 lines)
├── models.py                # CNN architectures (~400 lines)
├── README.md                # This file
│
├── dataset.py              # ⬜ TODO: PyTorch Dataset class
├── train.py                # ⬜ TODO: Training script
├── evaluate.py             # ⬜ TODO: Evaluation metrics
└── predict.py              # ⬜ TODO: Inference script
```

---

## ✅ Checklist

- [x] `utils.py` - Data loading & metadata
- [x] `preprocessing.py` - Normalization & patch extraction
- [x] `visualization.py` - Plotting functions
- [x] `models.py` - 3 CNN architectures
- [ ] `dataset.py` - PyTorch Dataset
- [ ] `train.py` - Training pipeline
- [ ] `evaluate.py` - Metrics & evaluation
- [ ] `predict.py` - Full-image inference

---

**Last updated:** 2025-10-22
**Author:** Ninh Hải Đăng
