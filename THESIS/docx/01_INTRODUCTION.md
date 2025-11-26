# ĐỒ ÁN TỐT NGHIỆP

## ỨNG DỤNG VIỄN THÁM VÀ HỌC SÂU TRONG GIÁM SÁT BIẾN ĐỘNG RỪNG TỈNH CÀ MAU
---

**Sinh viên thực hiện:** Ninh Hải Đăng
**Mã số sinh viên:** 21021411
**Lớp:** Công nghệ Hàng không Vũ trụ K66

**Giảng viên hướng dẫn:**
- TS. Hà Minh Cường
- ThS. Hoàng Tích Phúc

**Đơn vị:** Viện Công nghệ Hàng không Vũ trụ
Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội

**Thời gian thực hiện:** Học kỳ I, năm học 2025-2026

---

# MỤC LỤC

## PHẦN MỞ ĐẦU

## CHƯƠNG 1: TỔNG QUAN VỀ VẤN ĐỀ NGHIÊN CỨU
1.1. Bối cảnh và tình hình mất rừng
1.2. Công nghệ viễn thám trong giám sát rừng
1.3. Tổng quan các nghiên cứu liên quan
1.4. Khoảng trống nghiên cứu và định hướng đồ án

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT
2.1. Công nghệ viễn thám và ảnh vệ tinh
2.2. Mạng Neural Tích chập (Convolutional Neural Networks)
2.3. Phương pháp phân loại ảnh viễn thám
2.4. Đánh giá hiệu suất mô hình

## CHƯƠNG 3: PHƯƠNG PHÁP NGHIÊN CỨU
3.1. Khu vực và dữ liệu nghiên cứu
3.2. Quy trình xử lý dữ liệu
3.3. Kiến trúc mô hình CNN đề xuất
3.4. Huấn luyện và tối ưu hóa mô hình
3.5. Dự đoán và đánh giá kết quả

## CHƯƠNG 4: KẾT QUẢ VÀ THẢO LUẬN
4.1. Kết quả huấn luyện mô hình
4.2. Đánh giá hiệu suất trên tập kiểm tra
4.3. Kết quả phân loại toàn vùng nghiên cứu
4.4. So sánh với phương pháp Random Forest
4.5. Phân tích lỗi và thảo luận
4.6. Trực quan hóa kết quả

## CHƯƠNG 5: KẾT LUẬN VÀ KIẾN NGHỊ
5.1. Kết luận
5.2. Đóng góp khoa học
5.3. Hạn chế của nghiên cứu
5.4. Hướng phát triển

## TÀI LIỆU THAM KHẢO

## PHỤ LỤC

---
---

# PHẦN MỞ ĐẦU

## 1. Lý do chọn đề tài

Rừng đóng vai trò quan trọng trong việc duy trì cân bằng sinh thái, điều hòa khí hậu, lưu giữ carbon và bảo vệ đa dạng sinh học. Tuy nhiên, tình trạng mất rừng đang diễn ra nghiêm trọng trên toàn cầu, đặc biệt tại các quốc gia đang phát triển. Theo báo cáo "Global Forest Resources Assessment 2020" của Tổ chức Lương thực và Nông nghiệp Liên hợp quốc [1], thế giới đã mất ròng (net loss) khoảng 178 triệu hecta rừng trong giai đoạn 1990-2020, tương đương diện tích của Libya.

Tại Việt Nam, mặc dù độ che phủ rừng đã tăng từ 37% (năm 2000) lên 42% (năm 2020) nhờ các chương trình trồng rừng, nhưng tình trạng suy thoái và mất rừng tự nhiên vẫn đáng báo động, đặc biệt tại các tỉnh ven biển và đồng bằng sông Cửu Long. Tỉnh Cà Mau, nằm ở cực Nam Tổ Quốc, sở hữu hệ sinh thái rừng ngập mặn quan trọng nhưng đang phải đối mặt với áp lực từ nuôi trồng thủy sản, xâm nhập mặn, và biến đổi khí hậu.

Phương pháp giám sát rừng truyền thống dựa trên điều tra thực địa tốn kém thời gian, chi phí và khó áp dụng cho diện tích rộng. Công nghệ viễn thám vệ tinh cung cấp giải pháp hiệu quả, cho phép giám sát liên tục, diện rộng với chi phí hợp lý. Chương trình Copernicus của Liên minh Châu Âu (EU) cung cấp dữ liệu miễn phí từ các vệ tinh Sentinel-1 (SAR) và Sentinel-2 (Optical) với độ phân giải không gian 10m và chu kỳ quay trở lại ngắn (5-6 ngày), phù hợp cho giám sát rừng nhiệt đới.

Trong những năm gần đây, trí tuệ nhân tạo (AI) và học sâu (Deep Learning) đã đạt được những bước tiến vượt bậc trong xử lý ảnh và nhận dạng mẫu. Mạng Neural Tích chập (Convolutional Neural Networks - CNN) đặc biệt hiệu quả trong phân loại ảnh nhờ khả năng tự động học đặc trưng không gian (spatial features) từ dữ liệu thô. Khác với các phương pháp học máy truyền thống như Random Forest hay SVM đòi hỏi trích xuất đặc trưng thủ công, CNN có thể học các mẫu phức tạp và bất biến đối với phép tịnh tiến, xoay.

Xuất phát từ nhu cầu thực tiễn về giám sát rừng hiệu quả và xu hướng ứng dụng công nghệ AI tiên tiến, đồ án này lựa chọn đề tài **"Ứng dụng mạng Neural Tích chập sâu trong giám sát biến động rừng từ ảnh vệ tinh đa nguồn: Nghiên cứu điển hình tại tỉnh Cà Mau"** nhằm phát triển hệ thống tự động phát hiện mất rừng với độ chính xác cao.

## 2. Mục tiêu nghiên cứu

### 2.1. Mục tiêu tổng quát

Phát triển mô hình học sâu dựa trên kiến trúc CNN để phát hiện và phân loại tự động các khu vực biến động rừng từ ảnh vệ tinh đa nguồn (Sentinel-1 SAR và Sentinel-2 Optical) tại tỉnh Cà Mau.

### 2.2. Mục tiêu cụ thể

1. **Xây dựng bộ dữ liệu huấn luyện:** Thu thập và xử lý dữ liệu ảnh vệ tinh Sentinel-1/2 đa thời gian, kết hợp với ground truth points để tạo bộ dữ liệu huấn luyện chất lượng cao.

2. **Thiết kế kiến trúc CNN tối ưu:** Đề xuất kiến trúc CNN nhẹ (lightweight) phù hợp với bộ dữ liệu có quy mô vừa phải (~2,600 mẫu), tích hợp các kỹ thuật regularization (Batch Normalization, Dropout) để tránh overfitting.

3. **Phân chia dữ liệu khoa học:** Triển khai phương pháp stratified random split để đảm bảo phân bố lớp đồng đều giữa tập huấn luyện và test, kết hợp 5-Fold Cross Validation để đánh giá độ ổn định của mô hình.

4. **Huấn luyện và tối ưu hóa mô hình:** Áp dụng các kỹ thuật huấn luyện tiên tiến như early stopping, learning rate scheduling, class weighting để đạt được mô hình có hiệu suất cao và ổn định.

5. **Đánh giá và so sánh:** Đánh giá chi tiết hiệu suất mô hình CNN trên các chỉ số Accuracy, Precision, Recall, F1-Score, ROC-AUC. So sánh với phương pháp Random Forest để chứng minh ưu thế của Deep Learning so với Machine Learning truyền thống.

6. **Ứng dụng thực tế:** Áp dụng mô hình đã huấn luyện để phân loại toàn bộ khu vực rừng Cà Mau, ước tính diện tích mất rừng, và trực quan hóa kết quả dưới dạng bản đồ phân loại.

## 3. Đối tượng và phạm vi nghiên cứu

### 3.1. Đối tượng nghiên cứu

- **Đối tượng chính:** Các khu vực rừng tự nhiên và rừng trồng tại tỉnh Cà Mau, bao gồm rừng ngập mặn, rừng phòng hộ ven biển.

- **Biến động rừng:** Các trạng thái biến động được phân loại thành 4 nhóm:
  - **Forest Stable (Rừng ổn định):** Vùng rừng không có biến đổi trong giai đoạn nghiên cứu.
  - **Deforestation (Mất rừng):** Vùng rừng bị chuyển đổi sang đất trống, đất canh tác hoặc nuôi trồng thủy sản.
  - **Non-forest (Không phải rừng):** Vùng không có rừng trong cả hai thời điểm (đất trống, mặt nước, khu dân cư).
  - **Reforestation (Tái trồng rừng):** Vùng không có rừng trở thành rừng trong giai đoạn nghiên cứu.

- **Dữ liệu viễn thám:** Ảnh vệ tinh đa nguồn từ Sentinel-1 (SAR) và Sentinel-2 (Optical), kỳ trước (tháng 1-2/2024) và kỳ sau (tháng 2/2025).

### 3.2. Phạm vi nghiên cứu

- **Không gian:** Toàn bộ khu vực có rừng trong ranh giới hành chính tỉnh Cà Mau, diện tích khoảng **162,469 hecta** (tương đương 1,624.69 km²).

- **Thời gian:** Giai đoạn từ tháng 01/2024 đến tháng 02/2025 (khoảng 13 tháng).

- **Độ phân giải không gian:** 10 mét/pixel (độ phân giải gốc của Sentinel-1/2).

- **Hệ tọa độ:** EPSG:32648 (WGS 84 / UTM Zone 48N).

### 3.3. Giới hạn nghiên cứu

- Nghiên cứu sử dụng dữ liệu tại hai thời điểm (bi-temporal), chưa khai thác đầy đủ chuỗi thời gian liên tục.
- Ground truth được thu thập từ phiên giải ảnh và dữ liệu có sẵn, chưa có khảo sát thực địa đầy đủ.
- Mô hình được đào tạo và đánh giá trên dữ liệu Cà Mau, khả năng tổng quát hóa sang các khu vực khác cần được kiểm chứng thêm.

## 4. Phương pháp nghiên cứu

Đồ án áp dụng phương pháp nghiên cứu thực nghiệm kết hợp với phương pháp phân tích định lượng:

### 4.1. Thu thập và xử lý dữ liệu

- Thu thập ảnh Sentinel-1 SAR và Sentinel-2 Optical từ Copernicus Open Access Hub.
- Tiền xử lý: hiệu chỉnh khí quyển, co-registration, cắt theo ranh giới nghiên cứu.
- Tính toán các chỉ số thực vật (NDVI, NBR, NDMI) từ dữ liệu Sentinel-2.
- Thu thập ground truth points (2,630 điểm) qua phiên giải ảnh độ phân giải cao.

### 4.2. Trích xuất đặc trưng

- Xây dựng feature stack 27 chiều từ dữ liệu đa nguồn:
  - Sentinel-2: 21 features (7 before + 7 after + 7 delta)
  - Sentinel-1: 6 features (2 before + 2 after + 2 delta)
- Trích xuất patches không gian 3×3 pixels tại vị trí ground truth.
- Chuẩn hóa dữ liệu bằng phương pháp z-score standardization.

### 4.3. Phát triển mô hình CNN

- Thiết kế kiến trúc CNN nhẹ với 2 convolutional blocks + fully connected layers.
- Tổng số parameters: ~36,000 (phù hợp với bộ dữ liệu nhỏ).
- Áp dụng Batch Normalization, Dropout, Weight Decay để regularization.

### 4.4. Chia dữ liệu

- Sử dụng stratified random split để đảm bảo phân bố lớp đồng đều.
- Tỷ lệ: 80% Train+Val, 20% Fixed Test.
- 5-Fold Stratified Cross Validation trên tập Train+Val.

### 4.5. Huấn luyện và đánh giá

- Optimizer: AdamW với learning rate 0.001, weight decay 1e-4.
- Loss function: CrossEntropyLoss với class weights.
- Early stopping với patience 10 epochs.
- Đánh giá trên các chỉ số: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

### 4.6. So sánh với Random Forest

- Huấn luyện mô hình Random Forest (pixel-based) trên cùng dữ liệu.
- So sánh về accuracy, thời gian, khả năng sử dụng spatial context.

## 5. Ý nghĩa khoa học và thực tiễn

### 5.1. Ý nghĩa khoa học

- **Đóng góp về phương pháp:** Đề xuất kiến trúc CNN nhẹ và hiệu quả cho bài toán phân loại ảnh viễn thám với bộ dữ liệu nhỏ.

- **Quy trình đánh giá khoa học:** Áp dụng stratified split kết hợp 5-Fold Cross Validation, đảm bảo đánh giá mô hình một cách toàn diện và đáng tin cậy.

- **Tích hợp đa nguồn:** Kết hợp hiệu quả dữ liệu SAR (Sentinel-1) và Optical (Sentinel-2), tận dụng ưu thế của từng loại dữ liệu (SAR xuyên qua mây, Optical cung cấp thông tin quang phổ).

- **So sánh định lượng:** Cung cấp bằng chứng thực nghiệm về ưu thế của Deep Learning (CNN) so với Machine Learning truyền thống (Random Forest) trong phân loại ảnh viễn thám có tính không gian.

### 5.2. Ý nghĩa thực tiễn

- **Giám sát rừng hiệu quả:** Cung cấp công cụ tự động phát hiện mất rừng với độ chính xác cao (>99%), giảm đáng kể thời gian và chi phí so với phương pháp truyền thống.

- **Hỗ trợ quản lý tài nguyên:** Kết quả nghiên cứu giúp các cơ quan quản lý rừng tại Cà Mau và các tỉnh khác có cơ sở dữ liệu để ra quyết định bảo vệ và phát triển rừng bền vững.

- **Cảnh báo sớm:** Hệ thống có thể triển khai để giám sát liên tục, phát hiện kịp thời các hoạt động phá rừng trái phép.

- **Mở rộng ứng dụng:** Phương pháp có thể áp dụng cho các bài toán giám sát môi trường khác như biến động đất đai, đô thị hóa, biến đổi sử dụng đất.

- **Chi phí thấp:** Sử dụng dữ liệu vệ tinh miễn phí (Sentinel-1/2) và mô hình nhẹ (có thể chạy trên máy tính thông thường), phù hợp với điều kiện Việt Nam.

## 6. Cấu trúc đồ án

Đồ án được tổ chức thành 5 chương:

- **Chương 1 - Tổng quan về vấn đề nghiên cứu:** Trình bày bối cảnh mất rừng, công nghệ viễn thám, tổng quan các nghiên cứu liên quan và khoảng trống nghiên cứu.

- **Chương 2 - Cơ sở lý thuyết:** Giới thiệu chi tiết về công nghệ viễn thám (Sentinel-1/2), lý thuyết mạng Neural Tích chập (CNN), các phương pháp phân loại ảnh và đánh giá mô hình.

- **Chương 3 - Phương pháp nghiên cứu:** Mô tả chi tiết khu vực nghiên cứu, dữ liệu, quy trình xử lý, kiến trúc mô hình CNN đề xuất, phương pháp huấn luyện và đánh giá.

- **Chương 4 - Kết quả và thảo luận:** Trình bày kết quả huấn luyện, đánh giá mô hình, phân loại toàn vùng, so sánh với Random Forest, phân tích lỗi và trực quan hóa.

- **Chương 5 - Kết luận và kiến nghị:** Tóm tắt các kết quả đạt được, đóng góp khoa học, hạn chế và hướng phát triển tiếp theo.

---

📚 **Xem danh sách đầy đủ tài liệu tham khảo:** [REFERENCES.md](REFERENCES.md)
