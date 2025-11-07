# GEE-Style Workflow

Script Python duy nhất áp dụng workflow Google Earth Engine với 27 features.

## Tổng quan

- **27 features**: 7 bands S2 + 2 bands S1, mỗi band × 3 (before, after, delta)
- **Random Forest 100 trees**: Giống configuration GEE
- **Morphological operations**: Làm mượt kết quả (erosion + dilation)
- **1 script duy nhất**: Chạy toàn bộ workflow từ train đến inference

## Cách chạy

```bash
cd gee_approach
python gee_workflow.py
```

Script sẽ tự động:
1. ✓ Train Random Forest (100 trees, 27 features)
2. ✓ Evaluate trên validation set
3. ✓ Chạy inference trên full area (sliding window)
4. ✓ Apply morphological operations
5. ✓ Tạo visualization (4 panels)
6. ✓ Lưu tất cả kết quả

## Kết quả

Sau khi chạy xong, bạn sẽ có:

```
gee_approach/
├── models/
│   └── rf_gee_100trees.pkl          # Model đã train
└── results/
    ├── probability_map.npy           # Probability map [0, 1]
    ├── binary_map.npy                # Binary map (threshold=0.5)
    ├── binary_map_smooth.npy         # Sau morphology
    └── gee_workflow_results.png      # Visualization (4 panels)
```

## Visualization

Script tạo 1 figure với 4 panels:

1. **Probability Map**: Xác suất mất rừng cho mỗi pixel
2. **Binary Map**: Phân loại nhị phân (threshold=0.5)
3. **Smoothed Map**: Sau morphological operations
4. **Difference Map**: So sánh smoothed vs original

## Configuration

Bạn có thể chỉnh config trong script:

```python
CONFIG = {
    'n_estimators': 100,        # Số trees
    'patch_size': 64,           # Kích thước patch
    'stride': 32,               # Stride cho sliding window
    'batch_size': 32,           # Batch size
    'apply_morphology': True,   # Bật/tắt morphology
    'morphology_kernel': 3,     # Kích thước kernel
    'threshold': 0.5            # Threshold cho binary
}
```

## Dependencies

```bash
pip install numpy scikit-learn matplotlib opencv-python tqdm
```

## So sánh với approach cũ

| Aspect | Approach cũ | GEE-style |
|--------|------------|-----------|
| Features | 144 hoặc 27 | 27 |
| Trees | 500 | 100 |
| Morphology | Không | Có |
| Scripts | Nhiều notebooks | 1 script |
| Thời gian | ~30-40 phút | ~20-25 phút |
