# Ứng dụng Viễn thám và Học sâu trong Giám sát Biến động Rừng tỉnh Cà Mau

**Đồ án tốt nghiệp - Công nghệ Hàng không Vũ trụ**

Sinh viên: **Ninh Hải Đăng** (MSSV: 21021411)
Năm học: 2025 - 2026, Học kỳ I

---

## 📋 Tổng quan

Dự án này phát triển một hệ thống tự động giám sát biến động rừng tại tỉnh Cà Mau sử dụng kết hợp dữ liệu viễn thám đa nguồn (Sentinel-1 SAR và Sentinel-2 Optical) với hai phương pháp tiếp cận: Machine Learning truyền thống (Random Forest) và Deep Learning (CNN). Hệ thống có khả năng phát hiện và phân loại các khu vực mất rừng dựa trên phân tích chuỗi thời gian ảnh vệ tinh, với độ chính xác > 98%.

---

## 📊 Dữ liệu

### Ground Truth Points
- **Tổng số điểm:** 1,300 điểm training
- **Phân bố:**
  - Label 0 (Không mất rừng): 650 điểm (50.0%)
  - Label 1 (Mất rừng): 650 điểm (50.0%)
- **Format:** CSV file với các trường: `id`, `label`, `x`, `y` (tọa độ UTM Zone 48N)
- **File:** `data/raw/ground_truth/Training_Points_CSV.csv`

### Sentinel-2 (Optical)
- **7 bands** gồm spectral bands và spectral indices:
  - **Spectral bands:** B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2)
  - **Spectral indices:** NDVI, NBR, NDMI
- **Độ phân giải không gian:** 10m
- **Kỳ ảnh:**
  - Trước: 30/01/2024 (`S2_2024_01_30.tif`)
  - Sau: 28/02/2025 (`S2_2025_02_28.tif`)
- **Đã xử lý:** Cắt theo ranh giới rừng tỉnh Cà Mau, masked NoData

### Sentinel-1 (SAR)
- **2 bands:** VV và VH polarization
- **Độ phân giải không gian:** 10m (matched với Sentinel-2)
- **Kỳ ảnh:**
  - Trước: 04/02/2024 (`S1_2024_02_04_matched_S2_2024_01_30.tif`)
  - Sau: 22/02/2025 (`S1_2025_02_22_matched_S2_2025_02_28.tif`)
- **Đã xử lý:** Co-registered với Sentinel-2, cắt theo ranh giới rừng

### Boundary Shapefile
- **File:** `data/raw/boundary/forest_boundary.shp`
- **Mục đích:** Giới hạn khu vực phân tích chỉ trong vùng rừng

---

## 📦 Output Files

Sau khi chạy xong, kết quả được lưu trong folder `results/`:

**Random Forest Outputs:**
```
results/
├── rasters/
│   ├── rf_classification.tif               # Binary classification map (0/1)
│   └── rf_probability.tif                  # Probability map (0.0-1.0)
├── models/
│   └── rf_model.pkl                        # Trained Random Forest (277 KB)
├── data/
│   ├── training_data.csv                   # Training features (1,300 samples)
│   ├── rf_feature_importance.csv           # Feature importance rankings
│   └── rf_evaluation_metrics.json          # Performance metrics
└── plots/
    ├── rf_confusion_matrices.png           # Confusion matrices
    ├── rf_roc_curve.png                    # ROC curve
    ├── rf_feature_importance.png           # Top 20 features
    ├── rf_classification_maps.png          # Binary & probability maps
    └── rf_cv_scores.png                    # 5-fold CV scores
```

**CNN Outputs:**
```
results/
├── rasters/
│   ├── cnn_classification.tif              # Binary classification map
│   └── cnn_probability.tif                 # Probability map
├── models/
│   └── cnn_model.pth                       # Trained CNN (448 KB)
├── data/
│   ├── cnn_training_patches.npz            # Saved patches data
│   ├── cnn_evaluation_metrics.json         # Performance metrics
│   └── cnn_training_history.json           # Training curves (loss, acc)
└── plots/
    ├── cnn_confusion_matrices.png          # Confusion matrices
    ├── cnn_roc_curve.png                   # ROC curve
    ├── cnn_training_curves.png             # Loss & accuracy curves
    └── cnn_classification_maps.png         # Binary & probability maps
```

---

---

## 🔄 Pipeline Xử Lý

### Pipeline Random Forest (Pixel-based Classification)

Pipeline Random Forest xử lý dữ liệu ở mức **pixel-level**, sử dụng các feature được trích xuất từ chuỗi thời gian ảnh vệ tinh để phân loại từng pixel độc lập.

#### **Bước 1: Load Dữ liệu (Data Loading)**
- **Input:**
  - Sentinel-2 Before/After: 7 bands mỗi kỳ (B4, B8, B11, B12, NDVI, NBR, NDMI)
  - Sentinel-1 Before/After: 2 bands mỗi kỳ (VV, VH)
  - Ground truth points: CSV với 1,300 điểm (x, y, label)
  - Forest boundary: Shapefile ranh giới rừng

- **Xử lý:**
  - Load tất cả dữ liệu raster với `rasterio`
  - Đọc ground truth từ CSV với `pandas`
  - Kiểm tra kích thước, CRS, độ phân giải

- **Output:** Dictionary chứa arrays và metadata

#### **Bước 2: Feature Extraction**
- **Input:** S2 before/after (7×H×W), S1 before/after (2×H×W)

- **Xử lý:**
  ```
  1. Sentinel-2 Features (21 features):
     - S2_before[0:7]  → 7 features (B4, B8, B11, B12, NDVI, NBR, NDMI)
     - S2_after[0:7]   → 7 features
     - S2_delta = S2_after - S2_before → 7 features (temporal change)

  2. Sentinel-1 Features (6 features):
     - S1_before[0:2]  → 2 features (VV, VH)
     - S1_after[0:2]   → 2 features
     - S1_delta = S1_after - S1_before → 2 features (temporal change)

  3. Valid Mask Creation:
     - Loại bỏ pixels có NoData/NaN ở bất kỳ band/thời điểm nào
     - Đảm bảo tất cả 27 features hợp lệ cho mỗi pixel
  ```

- **Output:** Feature stack (27×H×W), Valid mask (H×W)

#### **Bước 3: Extract Training Data**
- **Input:** Feature stack, Ground truth points, Transform

- **Xử lý:**
  ```
  1. Coordinate Conversion:
     - Convert ground truth (x,y) từ UTM → pixel coordinates
     - Sử dụng rasterio transform

  2. Feature Extraction:
     - Với mỗi ground truth point:
       - Tìm pixel tương ứng (row, col)
       - Trích xuất 27 feature values tại pixel đó
       - Gán label từ ground truth
       - Skip nếu pixel nằm ngoài bounds hoặc có NoData

  3. Data Quality Check:
     - Kiểm tra missing values, infinite values
     - Kiểm tra features có zero variance
     - Kiểm tra class balance

  4. Train/Val/Test Split:
     - Train: 70% (stratified)
     - Validation: 15% (stratified)
     - Test: 15% (stratified)
     - Random state = 42 để reproducible
  ```

- **Output:**
  - Training DataFrame (n_samples × 28): 27 features + 1 label
  - Split arrays: X_train, X_val, X_test, y_train, y_val, y_test

#### **Bước 4: Train Random Forest Model**
- **Input:** X_train (n_train × 27), y_train (n_train,)

- **Hyperparameters:**
  ```python
  n_estimators = 100          # Số decision trees
  max_depth = 20              # Độ sâu tối đa của tree
  min_samples_split = 10      # Số samples tối thiểu để split
  min_samples_leaf = 4        # Số samples tối thiểu ở leaf node
  max_features = 'sqrt'       # Số features cho mỗi split
  class_weight = 'balanced'   # Cân bằng class weights
  oob_score = True            # Out-of-Bag score để đánh giá
  n_jobs = -1                 # Parallel processing
  random_state = 42
  ```

- **Training Process:**
  ```
  1. Model Creation:
     - Khởi tạo RandomForestClassifier với hyperparameters
     - Sử dụng sklearn.ensemble

  2. Model Fitting:
     - Fit model với X_train, y_train
     - Mỗi tree được train trên random subset của data
     - Bootstrap sampling với replacement
     - Random feature selection tại mỗi split

  3. Validation:
     - Đánh giá trên validation set
     - Tính OOB score (Out-of-Bag)
     - Log training/validation accuracy

  4. Feature Importance:
     - Tính Gini importance cho mỗi feature
     - Rank features theo importance
     - Lưu top 20 features quan trọng nhất
  ```

- **Output:**
  - Trained model (pickle file ~277 KB)
  - Feature importance rankings

#### **Bước 5: Predict Full Raster**
- **Input:** Feature stack (27×H×W), Valid mask, Trained model

- **Xử lý:**
  ```
  1. Reshape Features:
     - Reshape từ (27, H, W) → (H×W, 27)
     - Tạo 2D feature matrix cho prediction

  2. Batch Prediction:
     - Lọc chỉ valid pixels theo mask
     - Chia thành batches (10,000 pixels/batch) để tiết kiệm memory
     - Với mỗi batch:
       - predictions = model.predict(batch_features)
       - probabilities = model.predict_proba(batch_features)[:, 1]

  3. Reconstruct Rasters:
     - Tạo classification map: shape (H, W), dtype int8
       - 0 = No deforestation
       - 1 = Deforestation
       - -1 = NoData
     - Tạo probability map: shape (H, W), dtype float32
       - Range [0.0, 1.0] = xác suất mất rừng
       - -9999.0 = NoData
  ```

- **Output:**
  - Classification raster (GeoTIFF)
  - Probability raster (GeoTIFF)

#### **Bước 6: Evaluation & Visualization**
- **Input:** y_test, predictions, probabilities

- **Metrics:**
  ```
  1. Classification Metrics:
     - Accuracy, Precision, Recall, F1-Score
     - Confusion Matrix (train/val/test)
     - ROC Curve & AUC Score

  2. Cross-Validation:
     - 5-fold stratified CV
     - CV scores distribution plot

  3. Feature Analysis:
     - Feature importance plot (top 20)
     - Feature importance CSV export
  ```

- **Output:**
  - Confusion matrices plot
  - ROC curve plot
  - Feature importance plot
  - Classification maps visualization
  - Metrics JSON file

---

### Pipeline CNN (Patch-based Classification)

Pipeline CNN xử lý dữ liệu ở mức **patch-level**, sử dụng kiến trúc mạng neural để học spatial patterns từ các patches 3×3 pixels.

#### **Bước 1: Load Dữ liệu (Data Loading)**
- Giống với Random Forest Pipeline
- **Output:** Dictionary chứa arrays và metadata

#### **Bước 2: Feature Extraction**
- Giống với Random Forest Pipeline
- **Output:** Feature stack (27×H×W), Valid mask (H×W)

#### **Bước 3: Spatial Patch Extraction**
- **Input:** Feature stack (27×H×W), Ground truth points, Valid mask

- **Patch Configuration:**
  ```python
  patch_size = 3              # 3×3 spatial window
  half_size = 1               # Padding around center pixel
  ```

- **Extraction Process:**
  ```
  1. Coordinate Conversion:
     - Convert ground truth (x,y) → pixel coordinates (row, col)

  2. Patch Extraction:
     Với mỗi ground truth point tại (row, col):
     - Kiểm tra edge constraints:
       if row < 1 or row >= H-1 or col < 1 or col >= W-1: skip

     - Extract 3×3 window:
       patch = feature_stack[:, row-1:row+2, col-1:col+2]
       # Shape: (27, 3, 3)

     - Transpose để phù hợp CNN input:
       patch = transpose(patch, (1, 2, 0))
       # Shape: (3, 3, 27)

     - Validate patch:
       - Kiểm tra valid_mask[row-1:row+2, col-1:col+2].all()
       - Kiểm tra NaN/Inf values
       - Skip nếu patch không hợp lệ

  3. Quality Control:
     - Loại bỏ patches ở edge (không đủ padding)
     - Loại bỏ patches có NoData
     - Đảm bảo class balance
  ```

- **Output:**
  - Patches array: (n_samples, 3, 3, 27)
  - Labels array: (n_samples,)
  - Valid indices list

#### **Bước 4: Patch Normalization**
- **Input:** Raw patches (n_samples, 3, 3, 27)

- **Standardization Method:**
  ```python
  # Z-score normalization per feature channel
  mean = patches.mean(axis=(0, 1, 2), keepdims=True)  # Shape: (1, 1, 1, 27)
  std = patches.std(axis=(0, 1, 2), keepdims=True)    # Shape: (1, 1, 1, 27)

  normalized_patches = (patches - mean) / (std + 1e-8)
  ```

- **Output:**
  - Normalized patches
  - Normalization statistics (mean, std) để dùng cho inference

#### **Bước 5: Spatial Data Split**
- **Input:** Patches, Labels, Ground truth coordinates

- **Spatial Split Strategy:**
  ```
  1. Calculate Spatial Median:
     - median_x = median(ground_truth['x'])
     - median_y = median(ground_truth['y'])

  2. Spatial Quadrant Assignment:
     - NW quadrant (x < median_x, y >= median_y) → Train
     - NE quadrant (x >= median_x, y >= median_y) → Train
     - SW quadrant (x < median_x, y < median_y) → Test
     - SE quadrant (x >= median_x, y < median_y) → Validation

  3. Prevent Data Leakage:
     - Train/Val/Test không có overlap về không gian
     - Đảm bảo model không học từ vùng lân cận test areas
  ```

- **Output:**
  - X_train, y_train (spatial NW + NE)
  - X_val, y_val (spatial SE)
  - X_test, y_test (spatial SW)

#### **Bước 6: Build CNN Architecture**

- **Model Architecture:**
  ```
  Input: (batch, 3, 3, 27)
  ↓
  Permute → (batch, 27, 3, 3)  # PyTorch format: (N, C, H, W)
  ↓
  ┌─────────────────────────────────────┐
  │ Conv Block 1                        │
  │  - Conv2d: 27 → 64 channels (3×3)  │
  │  - BatchNorm2d(64)                  │
  │  - ReLU activation                  │
  │  - Dropout2d(p=0.3)                 │
  └─────────────────────────────────────┘
  ↓
  ┌─────────────────────────────────────┐
  │ Conv Block 2                        │
  │  - Conv2d: 64 → 32 channels (3×3)  │
  │  - BatchNorm2d(32)                  │
  │  - ReLU activation                  │
  │  - Dropout2d(p=0.3)                 │
  └─────────────────────────────────────┘
  ↓
  Global Average Pooling → (batch, 32, 1, 1)
  ↓
  Flatten → (batch, 32)
  ↓
  ┌─────────────────────────────────────┐
  │ FC Block                            │
  │  - Linear: 32 → 64                  │
  │  - BatchNorm1d(64)                  │
  │  - ReLU activation                  │
  │  - Dropout(p=0.5)                   │
  └─────────────────────────────────────┘
  ↓
  Linear: 64 → 2 (logits)
  ↓
  Output: (batch, 2)
  ```

- **Model Parameters:**
  ```
  - Total parameters: ~50,000 (trainable)
  - Model size: ~448 KB
  ```

- **Regularization Techniques:**
  ```
  - Batch Normalization: Ổn định training, giảm internal covariate shift
  - Dropout (0.3 conv, 0.5 fc): Prevent overfitting
  - Weight Decay (L2): 1e-4
  ```

#### **Bước 7: Train CNN Model**
- **Training Configuration:**
  ```python
  optimizer = AdamW(lr=0.001, weight_decay=1e-4)
  loss_fn = CrossEntropyLoss(weight=[1.0, 1.0])  # Balanced classes
  scheduler = ReduceLROnPlateau(factor=0.5, patience=5)

  batch_size = 32
  epochs = 50
  early_stopping_patience = 10
  ```

- **Training Loop:**
  ```
  For each epoch (1 to 50):
    1. Training Phase:
       - model.train()
       - For each batch in train_loader:
         - Forward pass: logits = model(patches)
         - Compute loss: loss = CrossEntropyLoss(logits, labels)
         - Backward pass: loss.backward()
         - Update weights: optimizer.step()
         - Track: train_loss, train_accuracy

    2. Validation Phase:
       - model.eval()
       - With torch.no_grad():
         - Forward pass trên validation set
         - Compute: val_loss, val_accuracy

    3. Learning Rate Scheduling:
       - scheduler.step(val_loss)
       - Giảm LR nếu val_loss không cải thiện sau 5 epochs

    4. Model Checkpointing:
       - If val_loss < best_val_loss:
         - Save model state_dict
         - Update best_val_loss, best_val_acc
         - Reset early_stopping_counter = 0
       - Else:
         - early_stopping_counter += 1

    5. Early Stopping:
       - If early_stopping_counter >= 10:
         - Stop training
         - Load best model checkpoint
  ```

- **Output:**
  - Best model checkpoint (cnn_model.pth)
  - Training history: train_loss, val_loss, train_acc, val_acc per epoch
  - Learning rate schedule

#### **Bước 8: Evaluate CNN Model**
- **Input:** Trained model, Test set (X_test, y_test)

- **Evaluation Process:**
  ```
  1. Test Inference:
     - model.eval()
     - With torch.no_grad():
       - logits = model(X_test)
       - probs = softmax(logits, dim=1)
       - preds = argmax(probs, dim=1)

  2. Metrics Calculation:
     - Accuracy = correct / total
     - Precision = TP / (TP + FP)
     - Recall = TP / (TP + FN)
     - F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
     - ROC-AUC = area under ROC curve

  3. Confusion Matrix:
     - Train set confusion matrix
     - Validation set confusion matrix
     - Test set confusion matrix
  ```

- **Output:**
  - Test metrics JSON
  - Confusion matrices plot
  - ROC curve plot
  - Training curves (loss/accuracy over epochs)

#### **Bước 9: Full Raster Prediction (Sliding Window)**
- **Input:** Feature stack (27×H×W), Valid mask, Trained model

- **Sliding Window Extraction:**
  ```
  1. Patch Grid Generation:
     - stride = 1 (sliding window với bước 1 pixel)
     - For row in range(1, H-1):
         For col in range(1, W-1):
           - Check valid_mask[row, col]
           - Extract patch tại (row, col)
           - Append to patches_list
           - Save coordinates (row, col)

  2. Batch Prediction:
     - Chia patches thành batches (1000 patches/batch)
     - For each batch:
       - Normalize batch using training mean/std
       - Forward pass: logits = model(batch)
       - probs = softmax(logits, dim=1)[:, 1]  # Prob of class 1
       - preds = argmax(logits, dim=1)

  3. Reconstruct Rasters:
     - Initialize classification_map (H, W) với NoData = -1
     - Initialize probability_map (H, W) với NoData = -9999
     - For each (row, col, pred, prob):
       - classification_map[row, col] = pred
       - probability_map[row, col] = prob
  ```

- **Output:**
  - CNN classification raster (GeoTIFF)
  - CNN probability raster (GeoTIFF)

#### **Bước 10: Probability Calibration**
- **Input:** Model predictions, True labels

- **Calibration Method:**
  ```python
  from sklearn.calibration import CalibratedClassifierCV

  # Isotonic Regression Calibration
  calibrator = CalibratedClassifierCV(
      base_estimator=None,  # Sử dụng CNN predictions
      method='isotonic',     # Isotonic regression
      cv='prefit'           # Model đã được train
  )

  calibrated_probs = calibrator.predict_proba(val_probs)
  ```

- **Calibration Metrics:**
  ```
  - Expected Calibration Error (ECE)
  - Reliability Diagram
  - Brier Score: measure of probability accuracy
  ```

- **Output:**
  - Calibrated probability raster
  - Calibration curve plot

#### **Bước 11: Post-processing & Visualization**
- Giống Random Forest Pipeline
- **Additional CNN-specific visualizations:**
  - Training curves (loss & accuracy)
  - Learning rate schedule
  - Calibration curves

---

### So sánh 2 Pipeline

| Aspect | Random Forest | CNN |
|--------|--------------|-----|
| **Input Unit** | Single pixel (27 features) | 3×3 patch (3×3×27) |
| **Spatial Context** | Không sử dụng spatial info | Học spatial patterns từ patches |
| **Feature Extraction** | Manual feature extraction | Automatic feature learning |
| **Training Time** | ~2-5 phút (100 trees) | ~10-20 phút (50 epochs) |
| **Model Size** | ~277 KB (pickle) | ~448 KB (PyTorch) |
| **Inference Speed** | Nhanh (~10k pixels/s) | Chậm hơn (~1k patches/s) |
| **Interpretability** | Cao (feature importance) | Thấp (black-box) |
| **Data Requirements** | Ít data, robust với noise | Cần nhiều data hơn |
| **Overfitting Risk** | Thấp với ensemble | Cao hơn (cần regularization) |
| **Edge Handling** | Predict tất cả valid pixels | Bỏ qua edge pixels (padding) |
| **Accuracy** | >98% | >98% |

---

## 📧 Liên hệ

- **Sinh viên:** Ninh Hải Đăng
- **Email:** ninhhaidangg@gmail.com
- **GitHub:** [ninhhaidang](https://github.com/ninhhaidang)
- **Đơn vị:** Trường Đại học Công nghệ - ĐHQGHN