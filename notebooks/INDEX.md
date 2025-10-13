# Notebooks Index

## 📖 Workflow: Từ Data → Results

### 01_exploration/
Khám phá dữ liệu ban đầu:
- `1.1_explore_sentinel2.ipynb`: Visualize S2 bands + indices
- `1.2_explore_sentinel1.ipynb`: Visualize SAR data
- `1.3_ground_truth_analysis.ipynb`: Phân tích 1285 điểm

### 02_preprocessing/
Chuẩn bị training data:
- `2.1_prepare_phase1.ipynb`: Tạo 14-channel input
- `2.2_prepare_phase2.ipynb`: Merge S2+S1 → 18 channels
- `2.3_create_samples.ipynb`: Extract patches từ ground truth

### 03_phase1_s2only/
Thí nghiệm Phase 1 (S2 only):
- `3.1_train_phase1.ipynb`: Training SNUNet-CD
- `3.2_evaluate_phase1.ipynb`: Metrics & confusion matrix
- `3.3_visualize_phase1.ipynb`: Prediction samples

### 04_phase2_s2s1/
Thí nghiệm Phase 2 (S2 + S1):
- `4.1_train_phase2.ipynb`: Training với 18 channels
- `4.2_evaluate_phase2.ipynb`: Metrics comparison
- `4.3_visualize_phase2.ipynb`: Prediction samples

### 05_comparison/
So sánh & inference cuối cùng:
- `5.1_compare_phases.ipynb`: Phase 1 vs Phase 2
- `5.2_inference_full_area.ipynb`: Change detection map toàn tỉnh

## 🎯 Thứ tự chạy
01 → 02 → 03 → 04 → 05
