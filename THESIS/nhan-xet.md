# BÁO CÁO PHẢN BIỆN ĐỒ ÁN TỐT NGHIỆP

## Đề tài: "Ứng dụng viễn thám và học sâu trong giám sát biến động rừng tỉnh Cà Mau"

**Sinh viên:** Đặng Ninh Hải
**Cơ sở đào tạo:** Trường Đại học Công nghệ - ĐHQGHN
**Người phản biện:** [Giáo sư chuyên ngành Viễn thám & GIS]

---

## I. NHẬN XÉT TỔNG QUAN

### 1.1. Tính khoa học và logic

Đồ án có **cấu trúc logic rõ ràng**, tuân thủ khuôn khổ nghiên cứu khoa học cơ bản: đặt vấn đề → tổng quan → phương pháp → kết quả → thảo luận → kết luận. Tuy nhiên, **tính khoa học còn một số điểm cần cải thiện**:

- Thiếu khung lý thuyết (theoretical framework) làm nền tảng cho nghiên cứu
- Câu hỏi nghiên cứu được đặt ra nhưng không được trả lời một cách hệ thống trong phần kết quả
- Thiếu giả thuyết nghiên cứu (research hypotheses) để kiểm định

### 1.2. Chất lượng học thuật

**Điểm mạnh:**
- Sử dụng phương pháp Cross Validation 5-Fold - đây là thực hành tốt
- Có Ablation Studies để phân tích ảnh hưởng của các thành phần
- Kết quả định lượng rõ ràng với các metrics chuẩn

**Điểm yếu nghiêm trọng:**
- **Thiếu validation thực địa (field validation)** - đây là lỗi phương pháp luận nghiêm trọng
- Ground truth data nguồn gốc không rõ ràng, không mô tả quy trình thu thập
- Thiếu phân tích uncertainty và confidence intervals

---

## II. ĐÁNH GIÁ CHI TIẾT TỪNG PHẦN

### 2.1. MỤC TIÊU NGHIÊN CỨU

| Tiêu chí | Đánh giá | Nhận xét |
|----------|----------|----------|
| Rõ ràng | 7/10 | Mục tiêu được nêu nhưng thiếu chỉ số đo lường cụ thể (SMART) |
| Khả thi | 8/10 | Phù hợp với thời gian và nguồn lực đồ án |
| Đóng góp | 6/10 | Chủ yếu là ứng dụng, đóng góp mới về phương pháp còn hạn chế |

**Vấn đề cụ thể:**
- Mục tiêu "phát triển mô hình CNN" nhưng thực chất là thiết kế kiến trúc CNN đơn giản, không phải đóng góp về architectural innovation
- Không đặt mục tiêu định lượng trước (ví dụ: đạt accuracy >95%)

### 2.2. PHƯƠNG PHÁP NGHIÊN CỨU

#### 2.2.1. Thu thập dữ liệu

**Vấn đề nghiêm trọng #1: Nguồn gốc Ground Truth không minh bạch**

Đồ án nêu có 2,630 điểm ground truth nhưng:
- Không mô tả **ai** thu thập, **khi nào**, **bằng phương pháp gì**
- Không có thông tin về **độ chính xác định vị** (GPS accuracy)
- Không nêu rõ ground truth được lấy từ **giải đoán ảnh** hay **khảo sát thực địa**
- Không có **sampling design** (random, stratified, systematic?)

> **Câu hỏi phản biện:** Ground truth này được thu thập như thế nào? Nếu chỉ giải đoán từ ảnh vệ tinh khác thì đây là **circular validation** - một sai lầm phương pháp luận cơ bản.

**Vấn đề #2: Thời điểm dữ liệu**

- Sentinel-2 Before: 30/01/2024
- Sentinel-2 After: 28/02/2025

Khoảng cách ~13 tháng là hợp lý, nhưng:
- Chỉ sử dụng **1 ảnh mỗi thời điểm** thay vì composite → nhạy với nhiễu khí quyển
- Không giải thích tiêu chí chọn ảnh (cloud cover threshold?)
- Không sử dụng **atmospheric correction** (SR vs TOA?) - không được đề cập

#### 2.2.2. Tiền xử lý dữ liệu

**Thiếu sót nghiêm trọng:**
- Không mô tả **quy trình tiền xử lý ảnh SAR** (speckle filtering?)
- Không nêu **geometric correction** giữa S1 và S2
- Không có **co-registration accuracy assessment**
- Không xử lý **temporal mismatch** giữa S1 (04/02/2024) và S2 (30/01/2024)

**Normalization:**
> Đồ án sử dụng Z-score normalization nhưng:
> - Không nêu mean/std được tính trên toàn bộ dữ liệu hay training set only
> - Nếu tính trên toàn bộ → **data leakage** nghiêm trọng

#### 2.2.3. Kiến trúc mô hình

**Phân tích kỹ thuật:**

Kiến trúc CNN với ~36,676 tham số là **quá đơn giản** cho bài toán này:

```
Input (3×3×27) → Conv(64) → Conv(32) → GAP → FC(64) → Output(4)
```

**Vấn đề:**
1. **Patch size 3×3 quá nhỏ**: Với độ phân giải 10m, patch 3×3 chỉ bao phủ 30×30m - nhỏ hơn diện tích một cây rừng trưởng thành. Điều này làm giảm khả năng capture spatial context
2. **Không có skip connections** như ResNet hoặc U-Net
3. **Dropout 0.7 là cực kỳ cao** - thường chỉ dùng 0.2-0.5
4. Không so sánh với **baseline models** (Random Forest, SVM, XGBoost)

#### 2.2.4. Đánh giá mô hình

**Vấn đề về data splitting:**

```
80% Train+Val → 5-Fold CV
20% Test (fixed)
```

- Không có **spatial stratification** - các điểm gần nhau có thể rơi vào cả train và test → **spatial autocorrelation** gây overestimate accuracy
- Không sử dụng **spatial blocking** hoặc **leave-one-region-out** cross validation

**Kết quả 98.86% accuracy là đáng ngờ:**

Độ chính xác này cao bất thường vì:
1. Các nghiên cứu tương tự chỉ đạt 85-96%
2. Có thể do **spatial autocorrelation** trong data (Tobler's First Law)
3. Có thể do **data leakage** trong normalization
4. Ground truth có thể không độc lập với dữ liệu training

### 2.3. PHÂN TÍCH KẾT QUẢ

#### 2.3.1. Kết quả phân loại

**Vấn đề thống kê:**
- Không có **confidence intervals** cho các metrics
- Không có **statistical significance tests** so sánh các configurations
- Ablation studies thiếu **p-values** hoặc **effect sizes**

**Vấn đề diễn giải:**

Đồ án kết luận phát hiện 7,282 ha mất rừng (4.48%) nhưng:
- Không có **error propagation analysis** từ model accuracy sang area estimation
- Không áp dụng **confusion matrix adjustment** cho area estimates (theo Olofsson et al., 2014)
- Không có **uncertainty bounds** cho diện tích ước tính

> **Công thức cần áp dụng:** Diện tích hiệu chỉnh = Diện tích thô × (User's Accuracy / Producer's Accuracy)

#### 2.3.2. Bản đồ sản phẩm

**Thiếu sót nghiêm trọng:**
- **Không có bản đồ thực sự** trong đồ án - chỉ có placeholders
- Không có **independent validation** với dữ liệu thực địa
- Không có **visual inspection** của kết quả ở các vùng điển hình
- Không so sánh với các sản phẩm có sẵn (Global Forest Watch, JAXA Forest Map)

### 2.4. THẢO LUẬN VÀ HẠN CHẾ

**Điểm tốt:**
- Tác giả nhận ra một số hạn chế (thời gian dự đoán, black-box, quy mô ground truth)

**Thiếu sót trong thảo luận:**
- Không phân tích **nguyên nhân lỗi phân loại** giữa Forest Stable và Deforestation
- Không thảo luận về **edge effects** tại ranh giới rừng
- Không đề cập đến **mixed pixels** và spectral confusion
- Không so sánh kết quả với **số liệu thống kê chính thức** về mất rừng Cà Mau

---

## III. CÁC SAI SÓT VÀ KHOẢNG TRỐNG HỌC THUẬT

### 3.1. Sai sót phương pháp luận

| # | Vấn đề | Mức độ | Khuyến nghị |
|---|--------|--------|-------------|
| 1 | Ground truth không có validation thực địa | Nghiêm trọng | Cần khảo sát thực địa ít nhất 100 điểm |
| 2 | Không có spatial stratification trong data splitting | Nghiêm trọng | Áp dụng spatial blocking CV |
| 3 | Thiếu atmospheric correction cho S2 | Trung bình | Sử dụng Sen2Cor hoặc FORCE |
| 4 | Thiếu speckle filtering cho S1 | Trung bình | Áp dụng Lee filter hoặc Refined Lee |
| 5 | Không có area estimation with uncertainty | Nghiêm trọng | Theo Olofsson et al. (2014) |

### 3.2. Khoảng trống học thuật

1. **Thiếu so sánh với baseline models:** Random Forest, SVM, Gradient Boosting thường là benchmarks bắt buộc
2. **Thiếu interpretability analysis:** Không có Grad-CAM, SHAP, hoặc feature importance
3. **Thiếu transferability test:** Model có generalize được sang vùng khác không?
4. **Thiếu temporal analysis:** Chỉ bi-temporal, chưa khai thác time series

---

## IV. ĐỀ XUẤT CẢI THIỆN CHUYÊN SÂU

### 4.1. Về phương pháp

**1. Spatial Cross-Validation:**
```python
# Sử dụng spatial blocking thay vì random split
from sklearn.model_selection import GroupKFold
# Chia theo grid cells hoặc administrative units
```

**2. Area Estimation theo chuẩn quốc tế:**
Áp dụng methodology của Olofsson et al. (2014) "Good practices for estimating area and assessing accuracy of land change" - Remote Sensing of Environment.

**3. Ensemble approach:**
Thay vì chỉ dùng CNN đơn lẻ:
- Random Forest + CNN ensemble
- Multi-temporal stacking
- Model uncertainty quantification

### 4.2. Về dữ liệu

**1. Image compositing:**
Thay vì 1 ảnh/thời điểm, sử dụng median composite của nhiều ảnh trong khoảng ±30 ngày

**2. Thêm features:**
- Texture metrics (GLCM)
- Topographic variables (DEM-derived)
- Distance to roads/settlements

**3. Ground truth:**
- Cần protocol thu thập rõ ràng
- Field validation ít nhất 5% total points
- Photointerpretation guidelines documented

### 4.3. Về mô hình

**1. So sánh architectures:**

| Model | Đề xuất thử nghiệm |
|-------|-------------------|
| ResNet-18 | Transfer learning từ ImageNet |
| U-Net | Semantic segmentation |
| Random Forest | Baseline comparison |
| XGBoost | Gradient boosting baseline |

**2. Patch size optimization:**
Thử nghiệm: 5×5, 7×7, 11×11, 15×15 với proper validation

**3. Uncertainty quantification:**
- Monte Carlo Dropout
- Deep Ensembles
- Bayesian Neural Networks

---

## V. ĐÁNH GIÁ CHẤT LƯỢNG TRÌNH BÀY

### 5.1. Bố cục và văn phong

| Tiêu chí | Điểm | Nhận xét |
|----------|------|----------|
| Cấu trúc logic | 8/10 | Rõ ràng, tuân thủ quy định |
| Văn phong học thuật | 7/10 | Khá tốt, đôi chỗ còn informal |
| Chính tả/ngữ pháp | 8/10 | Ít lỗi |
| Thuật ngữ | 6/10 | Trộn lẫn tiếng Anh-Việt không nhất quán |

### 5.2. Hình ảnh và bản đồ

**Vấn đề nghiêm trọng:**
- **Hầu hết hình ảnh là PLACEHOLDER** - chưa hoàn thiện
- Không có bản đồ kết quả thực sự
- Thiếu hình vẽ kiến trúc mô hình
- Thiếu hình minh họa quy trình

> Đây là thiếu sót không thể chấp nhận được cho một đồ án hoàn chỉnh.

### 5.3. Tài liệu tham khảo

**Điểm mạnh:**
- Có phân tách tiếng Anh/tiếng Việt
- Format nhất quán theo chuẩn VNU

**Điểm yếu:**
- Chỉ 27 tài liệu tiếng Anh, 3 tiếng Việt - **quá ít**
- Thiếu các công trình quan trọng:
  - Olofsson et al. (2014) - area estimation
  - Potapov et al. (2022) - latest global forest mapping
  - Stehman & Foody (2019) - accuracy assessment
- Một số citations không được trích dẫn trong text

---

## VI. PHÊ BÌNH TRỰC DIỆN

### 6.1. Những điểm không chấp nhận được

1. **Ground truth không minh bạch:** Đây là nền tảng của toàn bộ nghiên cứu. Không biết nguồn gốc ground truth thì không thể tin kết quả.

2. **Không có validation thực địa:** Một nghiên cứu về giám sát rừng mà không có một lần khảo sát thực địa là thiếu sót nghiêm trọng về phương pháp.

3. **Accuracy 98.86% không đáng tin:** Con số này cao bất thường và nhiều khả năng do:
   - Data leakage
   - Spatial autocorrelation
   - Overfitting

4. **Bản đồ kết quả là placeholder:** Đồ án về viễn thám mà không có bản đồ hoàn chỉnh là không đạt yêu cầu.

### 6.2. Những câu hỏi cần trả lời

1. Ground truth 2,630 điểm được thu thập như thế nào? Ai đã validate?
2. Tại sao không có một điểm khảo sát thực địa nào?
3. Normalization được thực hiện trước hay sau khi split data?
4. Các điểm ground truth có phân bố đều không gian không hay clustered?
5. Kết quả 7,282 ha mất rừng có khớp với số liệu thống kê chính thức không?

---

## VII. ĐIỂM ĐÁNH GIÁ SƠ BỘ

### Thang điểm 10 theo tiêu chuẩn hàn lâm nghiêm ngặt:

| Tiêu chí | Trọng số | Điểm | Điểm × Trọng số |
|----------|----------|------|-----------------|
| Tính mới và đóng góp | 15% | 5.5/10 | 0.825 |
| Phương pháp luận | 25% | 5.0/10 | 1.250 |
| Thu thập và xử lý dữ liệu | 15% | 5.5/10 | 0.825 |
| Phân tích và diễn giải | 15% | 6.0/10 | 0.900 |
| Độ tin cậy kết quả | 15% | 4.5/10 | 0.675 |
| Trình bày và hình ảnh | 10% | 4.0/10 | 0.400 |
| Tài liệu tham khảo | 5% | 6.5/10 | 0.325 |

### **TỔNG ĐIỂM: 5.2/10**

### Xếp loại: **TRUNG BÌNH** (theo tiêu chuẩn hàn lâm nghiêm ngặt)

---

## VIII. KẾT LUẬN VÀ KHUYẾN NGHỊ

### 8.1. Kết luận

Đồ án cho thấy sinh viên có **nền tảng kỹ thuật tốt** về deep learning và xử lý dữ liệu viễn thám. Tuy nhiên, còn **thiếu sót đáng kể về phương pháp luận khoa học**, đặc biệt là:
- Validation thực địa
- Minh bạch về nguồn dữ liệu
- Uncertainty quantification
- Hoàn thiện bản đồ sản phẩm

### 8.2. Điều kiện để được bảo vệ

Để đạt yêu cầu bảo vệ, sinh viên cần:

1. **Bắt buộc:**
   - Hoàn thiện tất cả hình ảnh và bản đồ (thay placeholder)
   - Giải thích rõ nguồn gốc ground truth
   - Thực hiện spatial cross-validation và so sánh kết quả

2. **Khuyến nghị mạnh:**
   - Thực hiện field validation ít nhất 50-100 điểm
   - So sánh kết quả với sản phẩm Global Forest Watch
   - Thêm baseline models (Random Forest)
   - Tính uncertainty cho area estimates

---

**Người phản biện**

*[Chữ ký]*

*Giáo sư chuyên ngành Viễn thám và GIS*

---

**Ghi chú cuối cùng cho sinh viên:**

Đồ án của bạn có tiềm năng và nền tảng kỹ thuật tốt. Tuy nhiên, trong nghiên cứu khoa học về viễn thám, **validation thực địa** và **minh bạch phương pháp** là không thể thiếu. Kết quả accuracy cao không có ý nghĩa nếu không có cơ sở khoa học vững chắc để đảm bảo độ tin cậy. Hãy xem những nhận xét này như cơ hội để hoàn thiện công trình của mình trước khi bảo vệ.
