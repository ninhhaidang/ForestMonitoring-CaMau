# 🌲 Workflow - Phát Hiện Mất Rừng Cà Mau với SNUNet-CD

**Dự án:** Đồ án tốt nghiệp K66 - Ninh Hải Đăng (21021411)  
**Framework:** Open-CD + SNUNet-CD  
**Phase 1:** Sentinel-2 only (7 bands)  
**Last Updated:** October 14, 2025

---

## 🎯 MỤC TIÊU PHASE 1

Train SNUNet-CD model để phát hiện forest loss detection với:
- **Input:** 7 bands Sentinel-2 (B4, B8, B11, B12, NDVI, NBR, NDMI)
- **Tile size:** 256×256 pixels
- **Task:** Binary change detection (0=no change, 1=forest loss)
- **Dataset:** 1,285 ground truth points từ khu vực Cà Mau
- **Time period:** 2024-01-30 → 2025-02-28

---

## 📋 TỔNG QUAN WORKFLOW

```
Phase 0: Environment Setup (15 phút)
    ↓
Phase I: Data Understanding (1 giờ)
    ↓
Phase II: Data Preprocessing (4-5 giờ)
    ↓
Phase III: Organize for Open-CD (1 giờ)
    ↓
Phase IV: Modify Open-CD for 7 Channels (2 giờ)
    ↓
Phase V: Training Setup (30 phút)
    ↓
Phase VI: Training & Monitoring (6-8 giờ)
    ↓
Phase VII: Evaluation & Analysis (2 giờ)
    ↓
Phase VIII: Documentation (1 giờ)
```

**Total Time:** ~18-21 giờ (12-13 giờ hands-on + 6-8 giờ training)

---

# PHASE 0: CHUẨN BỊ BAN ĐẦU

## ✅ BƯỚC 0: Kiểm tra Environment (15 phút)

### 0.1. Verify Open-CD Installation
- [ ] Activate conda environment: `conda activate opencd`
- [ ] Test Python imports: `import opencd, torch, mmcv`
- [ ] Check PyTorch + CUDA: `torch.cuda.is_available()` → True
- [ ] Verify GPU: `nvidia-smi` → RTX A4000 visible

### 0.2. Kiểm tra Raw Data
**Location:** `data/raw/`

- [ ] `sentinel2/S2_2024_01_30.tif` - exists, readable
- [ ] `sentinel2/S2_2025_02_28.tif` - exists, readable  
- [ ] `ground_truth/Training_Points_CSV.csv` - 1,285 rows
- [ ] Verify không có file corrupt

### 0.3. Backup Data
- [ ] Copy `data/raw/` → external drive/backup location
- [ ] Document backup location

**Output:** Environment ready, data verified ✓

---

# PHASE I: DATA UNDERSTANDING

## 🔍 BƯỚC 1: Phân tích Sentinel-2 Files (30 phút)

### 1.1. Create Analysis Notebook
**File:** `notebooks/01_exploration/01_inspect_sentinel2.ipynb`

### 1.2. Analyze S2_2024_01_30.tif
- [ ] Load file với rasterio
- [ ] Đếm số bands → verify = 7
- [ ] Đọc dimensions: width × height pixels
- [ ] Print CRS (coordinate reference system)
- [ ] Print bounds: (minx, miny, maxx, maxy)
- [ ] Check pixel resolution (meters per pixel)
- [ ] Print data type cho mỗi band (uint8/uint16/float32?)

### 1.3. Check Pixel Value Ranges
**Cho từng band:**
- [ ] Band 1 (B4): min/max values
- [ ] Band 2 (B8): min/max values
- [ ] Band 3 (B11): min/max values
- [ ] Band 4 (B12): min/max values
- [ ] Band 5 (NDVI): min/max values
- [ ] Band 6 (NBR): min/max values
- [ ] Band 7 (NDMI): min/max values

### 1.4. Analyze S2_2025_02_28.tif
- [ ] Repeat steps 1.2-1.3
- [ ] **Verify:** Same dimensions as 2024 file?
- [ ] **Verify:** Same CRS as 2024 file?
- [ ] **Verify:** Same bounds (aligned)?

### 1.5. Visualize Bands
- [ ] Plot each band separately
- [ ] Create false color composite (B8-B4-B11)
- [ ] Create difference images (2025 - 2024) per band
- [ ] Note observations: nodata values, outliers, clouds

### 1.6. Document Findings
**Record in notebook:**
- Exact image dimensions: _____ × _____ pixels
- CRS: _____
- Pixel resolution: _____ meters
- Value ranges per band
- Data type: _____
- Nodata value: _____
- Issues found: _____

**Output:** Complete understanding of Sentinel-2 data ✓

---

## 📍 BƯỚC 2: Phân tích Ground Truth (20 phút)

### 2.1. Create Analysis Notebook
**File:** `notebooks/01_exploration/02_inspect_groundtruth.ipynb`

### 2.2. Load and Inspect CSV
- [ ] Load `Training_Points_CSV.csv`
- [ ] Verify columns: `id, label, x, y`
- [ ] Verify 1,285 rows
- [ ] Check for missing values
- [ ] Check for duplicate points

### 2.3. Class Distribution Analysis
- [ ] Count class 0 (no change): _____ points
- [ ] Count class 1 (forest loss): _____ points
- [ ] Calculate imbalance ratio: _____
- [ ] **Document:** If >80% one class → need FocalLoss

### 2.4. Spatial Analysis
- [ ] Plot all points on scatter plot
- [ ] Color by class: green (0), red (1)
- [ ] Check for spatial clustering
- [ ] Verify points within Sentinel bounds

### 2.5. Coordinate System Check
- [ ] Identify CRS of points (EPSG:4326? WGS84?)
- [ ] Compare with Sentinel CRS
- [ ] **Decision:** Need coordinate transformation? Yes/No
- [ ] If yes, document transformation parameters

### 2.6. Document Findings
**Record in notebook:**
- Total points: 1,285
- Class 0: _____ (___%)
- Class 1: _____ (___%)
- Imbalance ratio: _____
- Points CRS: _____
- Transformation needed: Yes/No
- Outliers: _____

**Output:** Complete understanding of ground truth ✓

---

## 📊 BƯỚC 3: Calculate Tile Statistics (10 phút)

### 3.1. Create Calculation Notebook
**File:** `notebooks/01_exploration/03_calculate_tiles.ipynb`

### 3.2. Calculate Number of Tiles
**Given:**
- Image size: _____ × _____ pixels (from Step 1)
- Tile size: 256 × 256 pixels
- Overlap: 0 pixels (no overlap)

**Calculate:**
- [ ] Tiles per row: image_width / 256 = _____
- [ ] Tiles per column: image_height / 256 = _____
- [ ] Total tiles: rows × columns = _____
- [ ] Document: ~_____ tiles expected

### 3.3. Storage Estimation
- [ ] Estimate tile file size: ~100-200 KB per .npy file
- [ ] Total storage needed:
  - Tiles 2024: _____ tiles × 200 KB = _____ MB
  - Tiles 2025: _____ tiles × 200 KB = _____ MB
  - Masks: _____ tiles × 50 KB = _____ MB
  - **Total: _____ MB**
- [ ] Verify sufficient disk space available

### 3.4. Ground Truth Coverage
- [ ] Average points per tile: 1,285 / _____ tiles = _____ points/tile
- [ ] Estimate tiles with changes: ~_____
- [ ] Estimate tiles no change: ~_____
- [ ] **Note:** Many tiles will have no ground truth points (all-zero masks)

### 3.5. Dataset Split Estimation
**80/10/10 split:**
- [ ] Train tiles: _____ × 0.80 = ~_____
- [ ] Val tiles: _____ × 0.10 = ~_____
- [ ] Test tiles: _____ × 0.10 = ~_____

**Output:** Tile statistics documented ✓

---

# PHASE II: DATA PREPROCESSING

## 🛠️ BƯỚC 4: Process Sentinel-2 into Tiles (2 giờ)

### 4.1. Create Processing Script
**File:** `src/preprocessing/01_process_sentinel2_tiles.py`

### 4.2. Implement Workflow
**For each Sentinel file (2024, 2025):**

#### Step 4.2.1: Load Image
- [ ] Load all 7 bands into memory or chunks
- [ ] Handle large files efficiently

#### Step 4.2.2: Normalize Each Band
**For each of 7 bands:**
- [ ] Calculate percentiles: p2, p98
- [ ] Clip outliers: `clip(band, p2, p98)`
- [ ] Normalize to 0-255: `(band - p2) / (p98 - p2) × 255`
- [ ] Convert to uint8
- [ ] Handle nodata values appropriately

#### Step 4.2.3: Stack Bands
- [ ] Stack into single array: shape (height, width, 7)
- [ ] Verify shape correct
- [ ] Check memory usage

#### Step 4.2.4: Create Tiles
- [ ] Loop through all positions (256px stride)
- [ ] For each position (y, x):
  - Extract tile: `image[y:y+256, x:x+256, :]`
  - Verify shape = (256, 256, 7)
  - Store metadata: tile_id, x_offset, y_offset
  - Calculate geographic bounds for tile
  - Generate filename: `tile_{y:05d}_{x:05d}.npy`

#### Step 4.2.5: Save Tiles
- [ ] Save as .npy format (supports 7 channels)
- [ ] **Not PNG** - PNG only supports 3-4 channels
- [ ] Save to: `data/processed/phase1_s2only/tiles_2024/`
- [ ] Create metadata.json with geo info

#### Step 4.2.6: Repeat for 2025
- [ ] Process S2_2025_02_28.tif
- [ ] Save to: `data/processed/phase1_s2only/tiles_2025/`
- [ ] Verify same tile IDs as 2024

### 4.3. Verify Tile Creation
- [ ] Count files in tiles_2024/: _____ files
- [ ] Count files in tiles_2025/: _____ files
- [ ] Verify counts match expected
- [ ] Load random 3 tiles, check shape = (256, 256, 7)
- [ ] Check value ranges: 0-255
- [ ] Check no corrupt files

### 4.4. Output Structure
```
data/processed/phase1_s2only/
├── tiles_2024/
│   ├── tile_00000_00000.npy  # Shape: (256, 256, 7), dtype: uint8
│   ├── tile_00000_00256.npy
│   ├── tile_00256_00000.npy
│   └── ... (~360-400 files)
├── tiles_2025/
│   ├── tile_00000_00000.npy
│   └── ... (same count as 2024)
└── metadata.json  # Geo bounds for each tile
```

**Output:** Sentinel tiles created ✓

---

## 🎯 BƯỚC 5: Create Ground Truth Masks (2 giờ)

### 5.1. Create Processing Script
**File:** `src/preprocessing/02_create_groundtruth_masks.py`

### 5.2. Implement Workflow

#### Step 5.2.1: Load Ground Truth
- [ ] Load `Training_Points_CSV.csv`
- [ ] Create GeoDataFrame with points
- [ ] Set CRS appropriately

#### Step 5.2.2: Transform Coordinates (if needed)
- [ ] Transform points CRS → Sentinel CRS
- [ ] Verify transformation correct
- [ ] Test on sample points

#### Step 5.2.3: Create Tile → Points Lookup
- [ ] For each tile in metadata.json:
  - Get tile geographic bounds
  - Find all points within bounds
  - Store in dictionary: `{tile_id: [points]}`
- [ ] Document: _____ tiles have ground truth points
- [ ] Document: _____ tiles have no points (will be all-zero)

#### Step 5.2.4: Generate Mask for Each Tile
**For each tile:**

- [ ] Create blank mask: shape (256, 256), dtype uint8, all zeros
- [ ] Get points for this tile
- [ ] If no points → save all-zero mask
- [ ] If has points:
  - For each point:
    - Convert world coords → tile pixel coords (0-255 range)
    - Get point label (0 or 1)
    - Apply buffer around point (3×3 or 5×5 pixels)
      - **Why buffer:** Single pixel too small, model can't learn
    - Fill buffer area with label value
- [ ] Verify mask values only 0 or 1
- [ ] Save as PNG: `data/processed/phase1_s2only/masks/tile_xxxxx_xxxxx.png`

#### Step 5.2.5: Handle Edge Cases
- [ ] Points at tile edges: clip buffer to tile bounds
- [ ] Multiple overlapping points: take max label
- [ ] Invalid pixel coordinates: skip point

### 5.3. Quality Check Masks
- [ ] Count total masks created: _____ files
- [ ] Count masks with change (has 1s): _____ files
- [ ] Count masks no change (all 0s): _____ files
- [ ] Load random 5 masks, verify shape (256, 256)
- [ ] Check value range: only 0 and 1
- [ ] Calculate pixel statistics:
  - Total pixels: _____ tiles × 256 × 256
  - Pixels with label=1: _____
  - Imbalance ratio: _____

### 5.4. Visualize Sample Tiles
**Create visualization notebook:**
**File:** `notebooks/01_exploration/04_visualize_tiles_masks.ipynb`

- [ ] Select 5 tiles with ground truth changes
- [ ] Select 5 tiles no changes
- [ ] For each tile, display:
  - 2024 image (false color)
  - 2025 image (false color)
  - Ground truth mask
  - Overlay mask on images
- [ ] Visual inspection: Does it look reasonable?

### 5.5. Output Structure
```
data/processed/phase1_s2only/
└── masks/
    ├── tile_00000_00000.png  # 256×256, values: 0 or 1
    ├── tile_00000_00256.png
    └── ... (~360-400 files, matches tile count)
```

**Output:** Ground truth masks created ✓

---

## ✅ BƯỚC 6: Data Quality Check (30 phút)

### 6.1. Create Quality Check Notebook
**File:** `notebooks/01_exploration/05_data_quality_check.ipynb`

### 6.2. Verify File Counts
- [ ] Count: `ls data/processed/phase1_s2only/tiles_2024/*.npy | wc -l` = _____
- [ ] Count: `ls data/processed/phase1_s2only/tiles_2025/*.npy | wc -l` = _____
- [ ] Count: `ls data/processed/phase1_s2only/masks/*.png | wc -l` = _____
- [ ] **Verify:** All three counts identical? Yes/No

### 6.3. Verify File Name Consistency
- [ ] List tile names from tiles_2024/
- [ ] List tile names from tiles_2025/
- [ ] List tile names from masks/
- [ ] **Verify:** Every tile in 2024 has match in 2025 and masks? Yes/No
- [ ] Document any missing files: _____

### 6.4. Check Data Integrity
**Load random 10 tiles from each set:**

- [ ] Tiles 2024:
  - Shape = (256, 256, 7)? ✓
  - Dtype = uint8? ✓
  - Value range 0-255? ✓
- [ ] Tiles 2025:
  - Shape = (256, 256, 7)? ✓
  - Dtype = uint8? ✓
  - Value range 0-255? ✓
- [ ] Masks:
  - Shape = (256, 256)? ✓
  - Dtype = uint8? ✓
  - Values only 0 and 1? ✓

### 6.5. Visual Quality Check
**Display comparison grid:**
- [ ] 5 tiles with changes:
  - Display: [2024] [2025] [Mask] [Overlay]
  - Visual check: Change makes sense? Yes/No
  - Document observations: _____
- [ ] 5 tiles no changes:
  - Display: [2024] [2025] [Mask]
  - Visual check: Mask all zeros? Yes/No

### 6.6. Statistical Summary
**Document in notebook:**

```
Total tiles created: _____
Tiles with changes (mask has 1s): _____ (___%)
Tiles no change (mask all 0s): _____ (___%)
Total change pixels: _____
Total no-change pixels: _____
Pixel imbalance ratio: _____:1
```

### 6.7. Sign-Off
- [ ] All checks passed? Yes/No
- [ ] Data ready for next phase? Yes/No
- [ ] Issues to address: _____

**Output:** Data quality verified ✓

---

# PHASE III: ORGANIZE FOR OPEN-CD

## 📁 BƯỚC 7: Split Dataset (20 phút)

### 7.1. Create Split Script
**File:** `src/preprocessing/03_split_dataset.py`

### 7.2. Implement Split Logic

#### Step 7.2.1: List All Tiles
- [ ] Get list of all tile IDs from masks/ folder
- [ ] Total tiles: _____

#### Step 7.2.2: Separate by Class (Optional)
- [ ] Load each mask
- [ ] Identify tiles with changes (has 1s): _____
- [ ] Identify tiles no change (all 0s): _____

#### Step 7.2.3: Stratified Split
**If enough positive samples:**
- [ ] Split tiles with changes: 80/10/10
- [ ] Split tiles no change: 80/10/10
- [ ] Combine splits

**If too few positive samples:**
- [ ] Use random split: 80/10/10
- [ ] Set random_seed = 42 for reproducibility

#### Step 7.2.4: Verify Split Balance
- [ ] Train set: _____ tiles
  - With changes: _____ tiles
  - No change: _____ tiles
- [ ] Val set: _____ tiles
  - With changes: _____ tiles
  - No change: _____ tiles
- [ ] Test set: _____ tiles
  - With changes: _____ tiles
  - No change: _____ tiles

#### Step 7.2.5: Save Split Lists
- [ ] Save `train_files.txt` with tile IDs
- [ ] Save `val_files.txt` with tile IDs
- [ ] Save `test_files.txt` with tile IDs
- [ ] Document split statistics in README

### 7.3. Output Files
```
data/processed/phase1_s2only/
├── train_files.txt  # ~320 tile IDs
├── val_files.txt    # ~40 tile IDs
└── test_files.txt   # ~40 tile IDs
```

**Output:** Dataset split defined ✓

---

## 🗂️ BƯỚC 8: Organize Open-CD Structure (40 phút)

### 8.1. Create Organization Script
**File:** `src/preprocessing/04_organize_opencd_structure.py`

### 8.2. Create Directory Structure
- [ ] Create folders:
```
data/samples/phase1_s2only/
├── train/
│   ├── A/
│   ├── B/
│   └── label/
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

### 8.3. Copy Train Files
- [ ] Load train_files.txt
- [ ] For each tile ID:
  - Copy `tiles_2024/{tile_id}.npy` → `train/A/{tile_id}.npy`
  - Copy `tiles_2025/{tile_id}.npy` → `train/B/{tile_id}.npy`
  - Copy `masks/{tile_id}.png` → `train/label/{tile_id}.png`
- [ ] Progress: _____ files copied

### 8.4. Copy Val Files
- [ ] Load val_files.txt
- [ ] For each tile ID:
  - Copy to `val/A/`, `val/B/`, `val/label/`
- [ ] Progress: _____ files copied

### 8.5. Copy Test Files
- [ ] Load test_files.txt
- [ ] For each tile ID:
  - Copy to `test/A/`, `test/B/`, `test/label/`
- [ ] Progress: _____ files copied

### 8.6. Verify Copy Results
- [ ] Count files in `train/A/`: _____
- [ ] Count files in `train/B/`: _____
- [ ] Count files in `train/label/`: _____
- [ ] **Verify:** All three counts equal? Yes/No
- [ ] Repeat for val/ and test/
- [ ] Total files copied: _____ (should be tiles × 3 × 3 splits)

### 8.7. Final Structure Check
```
data/samples/phase1_s2only/
├── train/
│   ├── A/          # ~320 .npy files (2024 images)
│   ├── B/          # ~320 .npy files (2025 images)
│   └── label/      # ~320 .png files (masks)
├── val/
│   ├── A/          # ~40 .npy files
│   ├── B/          # ~40 .npy files
│   └── label/      # ~40 .png files
└── test/
    ├── A/          # ~40 .npy files
    ├── B/          # ~40 .npy files
    └── label/      # ~40 .png files
```

**Output:** Data organized for Open-CD ✓

---

# PHASE IV: MODIFY OPEN-CD FOR 7 CHANNELS

## ⚙️ BƯỚC 9: Create Custom Dataset Loader (1 giờ)

### 9.1. Create Dataset Class
**File:** `open-cd/opencd/datasets/camau_dataset.py`

### 9.2. Implementation Checklist

#### 9.2.1: Class Definition
- [ ] Import BaseCDDataset from opencd.datasets
- [ ] Create class CaMauDataset inheriting BaseCDDataset
- [ ] Add @DATASETS.register_module() decorator
- [ ] Define CLASSES = ('background', 'change')
- [ ] Define PALETTE = [[0,0,0], [255,0,0]]

#### 9.2.2: Override __init__
- [ ] Call super().__init__()
- [ ] Handle 7-channel specific parameters

#### 9.2.3: Override load_annotations
- [ ] List all .npy files in img_dir
- [ ] For each file:
  - Parse filename to get tile_id
  - Find corresponding label .png
  - Create annotation dict
  - Add to img_infos list
- [ ] Return img_infos

#### 9.2.4: Override __getitem__ or prepare_data
- [ ] Load .npy file (7 channels)
- [ ] Load .png label (1 channel)
- [ ] Apply transforms from pipeline
- [ ] Return dict with img and gt_semantic_seg

#### 9.2.5: Add Helper Methods
- [ ] Method to load 7-channel .npy
- [ ] Method to verify data integrity
- [ ] Method to get class statistics

### 9.3. Register Dataset
**Edit:** `open-cd/opencd/datasets/__init__.py`

- [ ] Add: `from .camau_dataset import CaMauDataset`
- [ ] Add to __all__: `'CaMauDataset'`

### 9.4. Test Dataset Class
**Create test notebook:** `notebooks/02_preprocessing/01_test_dataset_loader.ipynb`

- [ ] Import CaMauDataset
- [ ] Initialize dataset with train path
- [ ] Load sample: `dataset[0]`
- [ ] Verify output format:
  - img shape = (7, 256, 256)?
  - label shape = (256, 256)?
- [ ] Visualize sample
- [ ] Test augmentation pipeline

**Output:** Custom dataset class working ✓

---

## 🔧 BƯỚC 10: Modify Model Config (30 phút)

### 10.1. Create Model Config
**File:** `open-cd/configs/_base_/models/snunet_c32_7ch.py`

### 10.2. Copy Base Config
- [ ] Copy `snunet_c32.py` content
- [ ] Rename to `snunet_c32_7ch.py`

### 10.3. Modify for 7 Channels

#### 10.3.1: Update Backbone
```python
backbone=dict(
    type='SNUNet_ECAM',
    in_channels=7,        # ← Changed from 3
    base_channel=32,      # Keep
    num_stages=5,         # Keep
    enc_num_convs=[2, 2, 2, 2, 2],  # Keep
    dec_num_convs=[2, 2, 2, 2]      # Keep
)
```

#### 10.3.2: Update Data Preprocessor
```python
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[127.5] * 7,    # ← 7 values instead of 3
    std=[50.0] * 7,      # ← 7 values instead of 3
    bgr_to_rgb=False,    # ← Not applicable
    size=(256, 256),     # Keep
    pad_val=0,
    seg_pad_val=255
)
```

**Note:** Mean/std values are temporary. Should calculate from actual data later.

#### 10.3.3: Keep Other Settings
- [ ] decode_head config unchanged
- [ ] loss config unchanged
- [ ] train_cfg unchanged
- [ ] test_cfg unchanged

### 10.4. Calculate Actual Mean/Std (Optional but Recommended)
**Create script:** `src/preprocessing/05_calculate_normalization.py`

- [ ] Load all training tiles
- [ ] For each of 7 channels:
  - Calculate mean across all training data
  - Calculate std across all training data
- [ ] Update config with actual values

**Output:** Model config for 7 channels ✓

---

## 📊 BƯỚC 11: Create Dataset Config (30 phút)

### 11.1. Create Dataset Config
**File:** `open-cd/configs/_base_/datasets/camau_forest_7ch.py`

### 11.2. Configuration Structure

#### 11.2.1: Basic Settings
- [ ] `dataset_type = 'CaMauDataset'`
- [ ] `data_root = 'data/samples/phase1_s2only'`

#### 11.2.2: Normalization Config
```python
img_norm_cfg = dict(
    mean=[127.5] * 7,  # Or actual calculated values
    std=[50.0] * 7,    # Or actual calculated values
    to_rgb=False       # Not applicable for 7 channels
)
```

#### 11.2.3: Pipeline Config
**Train pipeline:**
- [ ] MultiImgLoadImageFromFile (needs support .npy)
- [ ] MultiImgLoadAnnotations (load .png)
- [ ] MultiImgRandomRotate (prob=0.5, degree=180)
- [ ] MultiImgRandomCrop (crop_size=(256,256))
- [ ] MultiImgRandomFlip (horizontal, prob=0.5)
- [ ] MultiImgRandomFlip (vertical, prob=0.5)
- [ ] MultiImgExchangeTime (prob=0.5) - swap A/B
- [ ] MultiImgPhotoMetricDistortion
- [ ] MultiImgNormalize (use img_norm_cfg)
- [ ] MultiImgDefaultFormatBundle
- [ ] Collect (keys=['img', 'gt_semantic_seg'])

**Test pipeline:**
- [ ] MultiImgLoadImageFromFile
- [ ] MultiImgMultiScaleFlipAug
- [ ] Simpler transforms (no augmentation)

#### 11.2.4: Dataloader Config
**Train dataloader:**
- [ ] batch_size = 8 (can reduce to 4 if OOM)
- [ ] num_workers = 4
- [ ] persistent_workers = True
- [ ] sampler = InfiniteSampler
- [ ] Dataset with train paths

**Val dataloader:**
- [ ] batch_size = 1
- [ ] num_workers = 4
- [ ] sampler = DefaultSampler
- [ ] Dataset with val paths

**Test dataloader:**
- [ ] Same as val

#### 11.2.5: Evaluator Config
- [ ] type = 'IoUMetric'
- [ ] iou_metrics = ['mIoU', 'mFscore']

### 11.3. Verify Config Syntax
- [ ] No Python syntax errors
- [ ] All paths correct
- [ ] All imported types exist in Open-CD

**Output:** Dataset config created ✓

---

# PHASE V: TRAINING SETUP

## 🎓 BƯỚC 12: Create Main Training Config (30 phút)

### 12.1. Create Training Config
**File:** `open-cd/configs/snunet/snunet_c32_256x256_40k_camau_7ch.py`

### 12.2. Configuration Structure

#### 12.2.1: Inherit Base Configs
```python
_base_ = [
    '../_base_/models/snunet_c32_7ch.py',
    '../_base_/datasets/camau_forest_7ch.py',
    '../_base_/schedules/schedule_40k.py',
    '../_base_/default_runtime.py'
]
```

#### 12.2.2: Model Overrides
- [ ] crop_size = (256, 256)
- [ ] data_preprocessor size = (256, 256)
- [ ] base_channel = 32
- [ ] in_channels = 7
- [ ] num_classes = 2

#### 12.2.3: Loss Function
**Choose based on class imbalance:**

**Option A: FocalLoss (if highly imbalanced)**
```python
loss_decode=dict(
    type='FocalLoss',
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.25,
    loss_weight=1.0
)
```

**Option B: CrossEntropyLoss + DiceLoss**
```python
loss_decode=[
    dict(type='CrossEntropyLoss', loss_weight=1.0),
    dict(type='DiceLoss', loss_weight=0.5)
]
```

#### 12.2.4: Optimizer
- [ ] type = 'AdamW'
- [ ] lr = 0.001
- [ ] weight_decay = 0.0005
- [ ] betas = (0.9, 0.999)

#### 12.2.5: Learning Rate Schedule
**Warmup:**
- [ ] type = 'LinearLR'
- [ ] start_factor = 1e-6
- [ ] begin = 0
- [ ] end = 1500 iterations

**Main schedule:**
- [ ] type = 'PolyLR'
- [ ] power = 1.0
- [ ] begin = 1500
- [ ] end = 40000

#### 12.2.6: Training Loop
- [ ] type = 'IterBasedTrainLoop'
- [ ] max_iters = 40000
- [ ] val_interval = 2000 (validate every 2k iters)

#### 12.2.7: Checkpointing
- [ ] interval = 2000
- [ ] save_best = 'mIoU'
- [ ] max_keep_ckpts = 3

#### 12.2.8: Work Directory
- [ ] work_dir = '../experiments/phase1_s2only'

### 12.3. Verify Config
- [ ] Run: `python tools/misc/print_config.py configs/snunet/snunet_c32_256x256_40k_camau_7ch.py`
- [ ] Check output for errors
- [ ] Verify all paths resolved correctly

**Output:** Training config complete ✓

---

# PHASE VI: TRAINING & MONITORING

## 🚀 BƯỚC 13: Pre-training Validation (30 phút)

### 13.1. Test Configuration
**In open-cd/ directory:**

#### 13.1.1: Print Config
- [ ] Run: `python tools/misc/print_config.py configs/snunet/snunet_c32_256x256_40k_camau_7ch.py`
- [ ] Review output: no errors? ✓
- [ ] Verify paths correct: ✓

#### 13.1.2: Test Dataloader
**Create test script:** `src/testing/01_test_dataloader.py`

- [ ] Import Open-CD dataset builder
- [ ] Build train dataloader from config
- [ ] Load one batch: `batch = next(iter(dataloader))`
- [ ] Check batch structure:
  - `batch['img'].shape` = (batch_size, 7, 256, 256)? ✓
  - `batch['gt_semantic_seg'].shape` = (batch_size, 1, 256, 256)? ✓
  - Values in correct range? ✓
- [ ] Visualize batch samples

#### 13.1.3: Test Forward Pass
- [ ] Build model from config
- [ ] Move model to GPU: `model.cuda()`
- [ ] Run forward: `output = model(batch)`
- [ ] Check output shape correct
- [ ] Check no CUDA errors
- [ ] Check no NaN values

### 13.2. Fix Any Errors
**Common issues:**
- [ ] Path errors → fix config paths
- [ ] CUDA OOM → reduce batch size
- [ ] Shape mismatch → check data loading
- [ ] Import errors → verify custom dataset registered

### 13.3. Final Pre-flight Check
- [ ] Environment activated: ✓
- [ ] GPU available: ✓
- [ ] Config valid: ✓
- [ ] Dataloader working: ✓
- [ ] Model forward pass OK: ✓
- [ ] Sufficient disk space for checkpoints: ✓

**Output:** Ready to train ✓

---

## 🏃 BƯỚC 14: Start Training (5-6 giờ unattended)

### 14.1. Setup Training Environment

#### 14.1.1: Open Two Terminals
**Terminal 1: Training**
- [ ] `cd open-cd`
- [ ] `conda activate opencd`

**Terminal 2: Monitoring**
- [ ] `cd project_root`
- [ ] `conda activate opencd`

### 14.2. Launch TensorBoard (Terminal 2)
- [ ] Run: `tensorboard --logdir experiments/phase1_s2only/vis_data`
- [ ] Open browser: http://localhost:6006
- [ ] Verify TensorBoard loads

### 14.3. Start Training (Terminal 1)
**Command:**
```bash
python tools/train.py \
    configs/snunet/snunet_c32_256x256_40k_camau_7ch.py \
    --work-dir ../experiments/phase1_s2only \
    --amp
```

**Flags:**
- `--amp`: Mixed precision (faster, less memory)
- `--work-dir`: Output directory
- Can add `--resume` to resume from checkpoint

### 14.4. Initial Monitoring (First 15 minutes)
**Watch console output:**
- [ ] Training starts without errors? ✓
- [ ] Iterations running? ✓
- [ ] Loss value shown (initial ~0.6-0.8)? ✓
- [ ] Iteration speed: ~0.3-0.5 sec/iter? ✓
- [ ] GPU memory usage: <16GB? ✓
- [ ] No warnings or errors? ✓

**If issues:**
- NaN loss → stop, reduce LR to 0.0001, restart
- CUDA OOM → stop, reduce batch_size to 4, restart
- Very slow → check data loading bottleneck

### 14.5. Training Progress Expectations

**Timeline (40,000 iterations @ 0.4 sec/iter = ~4.5 hours):**

| Iteration | Time | Expected Loss | Expected mIoU |
|-----------|------|---------------|---------------|
| 0 | Start | 0.6-0.8 | - |
| 2,000 | 13 min | 0.4-0.5 | ~0.50-0.60 |
| 5,000 | 33 min | 0.3-0.4 | ~0.65-0.70 |
| 10,000 | 1h 7min | 0.25-0.35 | ~0.70-0.75 |
| 20,000 | 2h 13min | 0.20-0.30 | ~0.75-0.80 |
| 30,000 | 3h 20min | 0.15-0.25 | ~0.78-0.83 |
| 40,000 | 4h 27min | 0.15-0.20 | ~0.80-0.85 |

### 14.6. Let Training Run
- [ ] Training running smoothly
- [ ] Leave overnight or work on other tasks
- [ ] Check occasionally via TensorBoard

**Output:** Training in progress...

---

## 📊 BƯỚC 15: Monitor Training (Ongoing)

### 15.1. TensorBoard Monitoring
**Check every 1-2 hours:**

#### 15.1.1: Loss Curves
- [ ] Training loss decreasing? ✓
- [ ] Smooth curve or noisy? (smooth better)
- [ ] No sudden spikes? ✓
- [ ] Validation loss also decreasing? ✓

#### 15.1.2: Metrics
- [ ] mIoU increasing over time? ✓
- [ ] F1-Score increasing? ✓
- [ ] Gap between train/val metrics reasonable? ✓

#### 15.1.3: Learning Rate
- [ ] LR schedule correct? ✓
- [ ] Warmup phase visible (0-1500 iters)? ✓
- [ ] Decay phase working? ✓

### 15.2. Console Monitoring

#### 15.2.1: Iteration Speed
- [ ] Speed stable: ~0.3-0.5 sec/iter ✓
- [ ] Not slowing down over time? ✓

#### 15.2.2: GPU Memory
- [ ] Memory usage stable: ~10-14GB ✓
- [ ] No memory leaks? ✓

#### 15.2.3: ETA (Estimated Time)
- [ ] Check remaining time
- [ ] Plan return time accordingly

### 15.3. Checkpoint Monitoring
**Every 2000 iterations, verify:**
- [ ] New checkpoint saved: `iter_XXXX.pth`
- [ ] Best model updated if mIoU improved
- [ ] Checkpoint file size reasonable (~100-200MB)

### 15.4. Problem Detection

**Stop training if:**
- Loss becomes NaN → reduce LR, restart
- Loss stuck (no change for 5000 iters) → check data
- GPU memory error → reduce batch size
- Loss exploding (>10.0) → reduce LR significantly

**Normal patterns:**
- Loss oscillates slightly → OK
- Val loss higher than train → OK (expected)
- Slow convergence near end → OK

### 15.5. Document Progress
**Create monitoring log:** `experiments/phase1_s2only/training_log.md`

**Record hourly:**
- Current iteration: _____
- Training loss: _____
- Validation mIoU: _____
- Time elapsed: _____
- Observations: _____

**Output:** Training monitored ✓

---

# PHASE VII: EVALUATION & ANALYSIS

## 🎯 BƯỚC 16: Test Set Evaluation (30 phút)

### 16.1. Wait for Training Completion
- [ ] Training finished: 40,000 iterations ✓
- [ ] Final checkpoint saved ✓
- [ ] Best model identified: `best_mIoU_iter_XXXXX.pth`

### 16.2. Run Test Evaluation
**Command:**
```bash
cd open-cd
python tools/test.py \
    configs/snunet/snunet_c32_256x256_40k_camau_7ch.py \
    ../experiments/phase1_s2only/best_mIoU_iter_XXXXX.pth
```

### 16.3. Collect Metrics
**From console output, record:**

- [ ] Overall Accuracy: _____
- [ ] mIoU: _____
- [ ] mFscore (F1-Score): _____

**Per-class metrics:**
- [ ] Class 0 (no change):
  - Precision: _____
  - Recall: _____
  - F1-Score: _____
- [ ] Class 1 (forest loss):
  - Precision: _____
  - Recall: _____
  - F1-Score: _____

**Confusion matrix:**
```
              Predicted
              0       1
Actual  0    ___    ___
        1    ___    ___
```

### 16.4. Compare with Baselines

**Expected performance:**
- Random classifier: F1 ~0.0, mIoU ~0.50
- Simple threshold: F1 ~0.40-0.60
- SNUNet-CD target: F1 ~0.70-0.85, mIoU ~0.75-0.85

**Your results:**
- F1-Score: _____ (above 0.70 = good)
- mIoU: _____ (above 0.75 = good)
- Status: _____ (Excellent/Good/Needs improvement)

### 16.5. Save Results
**Create results file:** `experiments/phase1_s2only/test_results.txt`

- [ ] Copy all metrics
- [ ] Add timestamp
- [ ] Note best checkpoint used

**Output:** Test evaluation complete ✓

---

## 📸 BƯỚC 17: Generate Visualizations (30 phút)

### 17.1. Generate Prediction Images
**Command:**
```bash
python tools/test.py \
    configs/snunet/snunet_c32_256x256_40k_camau_7ch.py \
    ../experiments/phase1_s2only/best_mIoU_iter_XXXXX.pth \
    --show-dir ../results/phase1_visualizations
```

### 17.2. Organize Visualizations
**Check output folder:** `results/phase1_visualizations/`

- [ ] Contains prediction overlay images
- [ ] Count: should match test set size (~40 images)

### 17.3. Select Example Cases
**Create selection notebook:** `notebooks/03_phase1_s2only/01_select_examples.ipynb`

#### 17.3.1: Best Predictions
- [ ] Find 10 tiles với:
  - High confidence (>0.9)
  - Correct predictions
  - Clear changes visible
- [ ] Copy to: `results/figures/phase1_best/`

#### 17.3.2: Failed Predictions
- [ ] Find 10 tiles với:
  - Incorrect predictions
  - False positives (predicted change, but no change)
  - False negatives (missed actual changes)
- [ ] Copy to: `results/figures/phase1_failures/`

#### 17.3.3: Edge Cases
- [ ] Find 5 tiles với:
  - Ambiguous cases
  - Partial changes
  - Boundary errors
- [ ] Copy to: `results/figures/phase1_edge_cases/`

### 17.4. Create Comparison Grid
**Script:** `src/evaluation_utils.py` (create function)

**For each example:**
- [ ] Load 2024 image (A)
- [ ] Load 2025 image (B)
- [ ] Load ground truth mask
- [ ] Load prediction
- [ ] Create 4-panel figure:
  - Panel 1: 2024 image
  - Panel 2: 2025 image
  - Panel 3: Ground truth (green=no change, red=change)
  - Panel 4: Prediction (same coloring)
- [ ] Add metrics overlay (Precision, Recall, F1)
- [ ] Save high-resolution version

### 17.5. Create Summary Figure
**Combine multiple examples:**
- [ ] 3 best cases (top row)
- [ ] 3 failed cases (middle row)
- [ ] 3 edge cases (bottom row)
- [ ] Add labels and annotations
- [ ] Save: `results/figures/phase1_summary_grid.png`

**Output:** Visualizations created ✓

---

## 📈 BƯỚC 18: Analysis & Documentation (1 giờ)

### 18.1. Create Analysis Notebook
**File:** `notebooks/03_phase1_s2only/02_results_analysis.ipynb`

### 18.2. Performance Analysis

#### 18.2.1: Class-wise Performance
- [ ] Plot confusion matrix heatmap
- [ ] Calculate per-class metrics table
- [ ] Analyze which class harder to predict
- [ ] Document: _____

#### 18.2.2: Spatial Analysis
- [ ] Map predictions back to geographic space
- [ ] Identify regions of high accuracy
- [ ] Identify regions of low accuracy
- [ ] Check for spatial patterns in errors
- [ ] Document observations: _____

#### 18.2.3: Error Analysis
**False Positives:**
- [ ] Count FP cases: _____
- [ ] Visualize FP examples
- [ ] Identify patterns:
  - Seasonal vegetation change?
  - Shadows/clouds?
  - Agricultural areas?
- [ ] Document causes: _____

**False Negatives:**
- [ ] Count FN cases: _____
- [ ] Visualize FN examples
- [ ] Identify patterns:
  - Small clearings missed?
  - Gradual deforestation?
  - Buffer too small in ground truth?
- [ ] Document causes: _____

### 18.3. Learning Curves Analysis

#### 18.3.1: Training Progress
- [ ] Plot training loss over time
- [ ] Plot validation loss over time
- [ ] Identify when model converged
- [ ] Check for overfitting (val loss increases)
- [ ] Document convergence point: iter _____

#### 18.3.2: Metrics Over Time
- [ ] Plot mIoU progression
- [ ] Plot F1-Score progression
- [ ] Identify best iteration
- [ ] Check if training long enough
- [ ] Document: Training sufficient? Yes/No

### 18.4. Band Importance Analysis (Optional)
**If time permits:**
- [ ] Test model with different band subsets
- [ ] Rank bands by importance
- [ ] Identify most informative bands
- [ ] Document for Phase 2 planning

### 18.5. Key Findings Summary

**Document in notebook:**

**Strengths:**
- Model performs well on _____
- High accuracy for _____
- Good at detecting _____

**Weaknesses:**
- Model struggles with _____
- Low accuracy for _____
- Misses _____

**Insights:**
- _____
- _____
- _____

**Improvement Ideas:**
- _____
- _____
- _____

### 18.6. Create Summary Report
**File:** `experiments/phase1_s2only/summary_report.md`

**Include:**
- [ ] Project overview
- [ ] Dataset statistics
- [ ] Model configuration
- [ ] Training details
- [ ] Test results (all metrics)
- [ ] Key findings
- [ ] Visualizations
- [ ] Next steps

**Output:** Analysis complete ✓

---

# PHASE VIII: DOCUMENTATION & NEXT STEPS

## 📝 BƯỚC 19: Update Documentation (30 phút)

### 19.1. Update Project README
**File:** `README.md`

#### 19.1.1: Project Status
- [ ] Add: "Phase 1: COMPLETED ✓"
- [ ] Add completion date
- [ ] Add link to results summary

#### 19.1.2: Phase 1 Results Section
**Add new section:**
```markdown
## Phase 1 Results (Sentinel-2 Only)

**Model:** SNUNet-CD (32 channels, 7-band input)
**Dataset:** 1,285 ground truth points, ~400 tiles (256×256)
**Training:** 40,000 iterations (~5 hours on RTX A4000)

### Metrics
- **mIoU:** _____
- **F1-Score:** _____
- **Precision:** _____
- **Recall:** _____
- **Overall Accuracy:** _____

### Key Findings
- [Finding 1]
- [Finding 2]
- [Finding 3]

See detailed results: [experiments/phase1_s2only/summary_report.md](experiments/phase1_s2only/summary_report.md)
```

#### 19.1.3: Dataset Statistics
- [ ] Update with final numbers
- [ ] Add train/val/test split
- [ ] Add class distribution

#### 19.1.4: Add Visualizations
- [ ] Embed summary figure
- [ ] Link to full visualization folder

### 19.2. Update Training Guide
**File:** `docs/02_training_guide.md`

- [ ] Document actual training time
- [ ] Add lessons learned
- [ ] Document issues encountered
- [ ] Add solutions/workarounds
- [ ] Include tips for future training

### 19.3. Update Data Guide
**File:** `docs/01_data_guide.md`

- [ ] Document actual preprocessing steps used
- [ ] Add statistics about data quality
- [ ] Note any data issues discovered
- [ ] Document ground truth coverage

### 19.4. Create Checklist for Reproducibility
**File:** `docs/reproducibility_checklist.md`

**Document:**
- [ ] Exact Python environment (conda list > environment_freeze.txt)
- [ ] Random seeds used
- [ ] Data preprocessing parameters
- [ ] Model hyperparameters
- [ ] Training commands
- [ ] Evaluation commands

**Output:** Documentation updated ✓

---

## 🔜 BƯỚC 20: Plan Phase 2 (30 phút)

### 20.1. Create Phase 2 Plan
**File:** `docs/phase2_plan.md`

### 20.2. Phase 2 Scope

#### 20.2.1: Data Preparation
- [ ] Process Sentinel-1 data (VH, R bands)
  - Similar workflow as Phase 1 Step 4
  - Create tiles 256×256
  - Save as .npy (now 9 channels: 7 S2 + 2 S1)
- [ ] Verify S1 aligned with S2
- [ ] Combine into single 9-channel tiles
- [ ] Use same ground truth masks
- [ ] Organize same train/val/test split

#### 20.2.2: Model Modifications
- [ ] Modify `snunet_c32_9ch.py`
- [ ] Change `in_channels=7` → `in_channels=9`
- [ ] Update normalization (mean/std for 9 channels)
- [ ] Consider: Start from Phase 1 weights (transfer learning)?

#### 20.2.3: Training Strategy
**Decision needed:**
- Option A: Train from scratch (fair comparison)
- Option B: Fine-tune from Phase 1 model (faster)
- **Recommend:** Option A for Phase 2, Option B if time limited

#### 20.2.4: Comparison Plan
- [ ] Use exact same test set as Phase 1
- [ ] Calculate same metrics
- [ ] Direct comparison: Phase 1 vs Phase 2
- [ ] Ablation study: Which bands contribute most?
  - S2 only: [Phase 1 results]
  - S2 + S1: [Phase 2 results]
  - S1 only: [Optional additional experiment]

### 20.3. Timeline Estimation

| Task | Estimated Time |
|------|----------------|
| Process S1 data | 2-3 giờ |
| Combine with S2 | 1 giờ |
| Update configs | 1 giờ |
| Training | 5-6 giờ |
| Evaluation | 1 giờ |
| Analysis | 2 giờ |
| **Total** | **12-14 giờ** |

### 20.4. Success Criteria for Phase 2

**Minimum success:**
- Phase 2 mIoU ≥ Phase 1 mIoU

**Good success:**
- Phase 2 mIoU > Phase 1 mIoU + 0.02
- Improvement in small object detection

**Excellent success:**
- Phase 2 mIoU > Phase 1 mIoU + 0.05
- Better recall for forest loss class
- Fewer false negatives

### 20.5. Research Questions for Phase 2
- [ ] Does S1 SAR improve detection accuracy?
- [ ] Which band combination optimal?
- [ ] Does S1 help with cloud-covered areas?
- [ ] Does S1 reduce seasonal confusion?
- [ ] What's the computational cost tradeoff?

### 20.6. Risk Mitigation
**Potential issues:**
- S1 and S2 misalignment → Pre-check registration
- S1 adds noise → Try with/without S1
- Training longer → Monitor carefully for overfitting
- No improvement → Analyze why, still valuable result

**Output:** Phase 2 planned ✓

---

# ✅ COMPLETION CHECKLIST

## Phase 1 Complete When:

### Data Preparation
- [x] Sentinel-2 data processed into tiles
- [x] Ground truth masks created
- [x] Data split into train/val/test
- [x] Data organized in Open-CD format
- [x] Data quality verified

### Model Development
- [x] Custom 7-channel dataset loader created
- [x] Model config modified for 7 channels
- [x] Training config complete
- [x] Pre-training tests passed

### Training
- [x] Training completed (40,000 iterations)
- [x] Best model saved
- [x] Training curves look good
- [x] No major issues encountered

### Evaluation
- [x] Test set evaluation done
- [x] All metrics calculated
- [x] Metrics documented
- [x] Meets minimum success criteria (F1 > 0.70)

### Analysis
- [x] Error analysis completed
- [x] Visualizations created
- [x] Key findings identified
- [x] Improvement ideas documented

### Documentation
- [x] README updated
- [x] Summary report written
- [x] Code documented
- [x] Lessons learned recorded
- [x] Phase 2 plan created

### Repository
- [x] All code committed to git
- [x] Results saved
- [x] Environment documented
- [x] Reproducibility ensured

---

# 📊 TIME TRACKING

| Phase | Planned | Actual | Notes |
|-------|---------|--------|-------|
| 0. Environment | 15 min | _____ | _____ |
| I. Data Understanding | 1 hr | _____ | _____ |
| II. Preprocessing | 4-5 hr | _____ | _____ |
| III. Organization | 1 hr | _____ | _____ |
| IV. Modify Open-CD | 2 hr | _____ | _____ |
| V. Training Setup | 30 min | _____ | _____ |
| VI. Training | 6-8 hr | _____ | _____ |
| VII. Evaluation | 2 hr | _____ | _____ |
| VIII. Documentation | 1 hr | _____ | _____ |
| **TOTAL** | **18-21 hr** | **_____** | _____ |

---

# 🚨 TROUBLESHOOTING GUIDE

## Common Issues & Solutions

### During Data Preparation
**Issue:** Tiles have wrong dimensions
- **Check:** Verify image size divisible by 256
- **Fix:** Adjust tiling logic to handle edges

**Issue:** Ground truth masks all zeros
- **Check:** Coordinate transformation correct?
- **Fix:** Verify CRS matching between points and images

### During Training
**Issue:** CUDA out of memory
- **Fix:** Reduce batch_size to 4
- **Fix:** Enable gradient checkpointing
- **Fix:** Use --amp flag

**Issue:** Loss becomes NaN
- **Fix:** Reduce learning rate to 0.0001
- **Fix:** Check data for NaN values
- **Fix:** Use gradient clipping

**Issue:** Loss not decreasing
- **Fix:** Verify data loading correctly
- **Fix:** Check label format (0/1 not 0/255)
- **Fix:** Try different learning rate

### During Evaluation
**Issue:** Very low scores
- **Check:** Using correct checkpoint?
- **Check:** Test data format correct?
- **Check:** Class imbalance extreme?

---

# 📞 SUPPORT & RESOURCES

## Documentation
- Open-CD GitHub: https://github.com/likyoo/open-cd
- MMSegmentation Docs: https://mmsegmentation.readthedocs.io
- This workflow: `docs/workflow.md`

## Project Files
- Main README: `README.md`
- Data guide: `docs/01_data_guide.md`
- Training guide: `docs/02_training_guide.md`
- Summary report: `experiments/phase1_s2only/summary_report.md`

## Key Notebooks
- Data inspection: `notebooks/01_exploration/`
- Training analysis: `notebooks/03_phase1_s2only/`

---

**Document Version:** 1.0  
**Created:** October 14, 2025  
**Last Updated:** October 14, 2025  
**Author:** Ninh Hải Đăng (21021411)