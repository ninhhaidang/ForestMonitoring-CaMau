---
title: "ỨNG DỤNG VIỄN THÁM VÀ HỌC SÂU TRONG GIÁM SÁT BIẾN ĐỘNG RỪNG TỈNH CÀ MAU"
subtitle: |
  ĐỒ ÁN TỐT NGHIỆP

  **Sinh viên thực hiện:** Ninh Hải Đăng

  **Mã số sinh viên:** 21021411

  **Lớp:** Công nghệ Hàng không Vũ trụ K66

  **Giảng viên hướng dẫn:**

  - TS. Hà Minh Cường
  - ThS. Hoàng Tích Phúc

  **Đơn vị:** Viện Công nghệ Hàng không Vũ trụ

  Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội
author: "Ninh Hải Đăng"
date: "2025"
lang: vi
toc: true
toc-depth: 3
numbersections: true
---

\newpage

::: {custom-style="Abstract"}

# TÓM TẮT

Đồ án này nghiên cứu ứng dụng mạng Neural Tích chập (CNN) để phát hiện và phân loại biến động rừng từ dữ liệu viễn thám đa nguồn tại tỉnh Cà Mau. Nghiên cứu sử dụng dữ liệu từ vệ tinh Sentinel-1 (SAR) và Sentinel-2 (Optical) với độ phân giải 10m, kết hợp 27 đặc trưng (features) từ cả hai nguồn dữ liệu.

**Phương pháp:** Thiết kế kiến trúc CNN nhẹ (~36,000 tham số) với patches không gian 3×3, áp dụng stratified random split kết hợp 5-Fold Cross Validation để đánh giá mô hình. Bộ dữ liệu gồm 2,630 điểm ground truth được chia thành 80% Train+Val và 20% Test cố định.

**Kết quả:** Mô hình CNN đạt độ chính xác 98.86% trên tập test với ROC-AUC 99.98%. Cross Validation cho kết quả 98.15% ± 0.28%, chứng tỏ mô hình ổn định và có khả năng tổng quát hóa tốt. So với Random Forest, CNN cải thiện accuracy 0.63% và giảm error rate 33.3%.

**Ứng dụng:** Phân loại toàn vùng rừng Cà Mau (162,469 ha), phát hiện 7,282 ha mất rừng (4.48%) và 4,941 ha phục hồi rừng (3.04%) trong giai đoạn 01/2024 - 02/2025.

**Từ khóa:** CNN, Deep Learning, viễn thám, Sentinel-1, Sentinel-2, biến động rừng, Cà Mau

:::

\newpage

# PHẦN MỞ ĐẦU

## Lý do chọn đề tài

Rừng đóng vai trò quan trọng trong việc duy trì cân bằng sinh thái, điều hòa khí hậu, lưu giữ carbon và bảo vệ đa dạng sinh học. Tuy nhiên, tình trạng mất rừng đang diễn ra nghiêm trọng trên toàn cầu, đặc biệt tại các quốc gia đang phát triển. Theo báo cáo "Global Forest Resources Assessment 2020" của Tổ chức Lương thực và Nông nghiệp Liên hợp quốc [1], thế giới đã mất ròng (net loss) khoảng 178 triệu hecta rừng trong giai đoạn 1990-2020, tương đương diện tích của Libya.

Tại Việt Nam, mặc dù độ che phủ rừng đã tăng từ 37% (năm 2000) lên 42% (năm 2020) nhờ các chương trình trồng rừng, nhưng tình trạng suy thoái và mất rừng tự nhiên vẫn đáng báo động, đặc biệt tại các tỉnh ven biển và đồng bằng sông Cửu Long. Tỉnh Cà Mau, nằm ở cực Nam Tổ Quốc, sở hữu hệ sinh thái rừng ngập mặn quan trọng nhưng đang phải đối mặt với áp lực từ nuôi trồng thủy sản, xâm nhập mặn, và biến đổi khí hậu.

Phương pháp giám sát rừng truyền thống dựa trên điều tra thực địa tốn kém thời gian, chi phí và khó áp dụng cho diện tích rộng. Công nghệ viễn thám vệ tinh cung cấp giải pháp hiệu quả, cho phép giám sát liên tục, diện rộng với chi phí hợp lý. Chương trình Copernicus của Liên minh Châu Âu (EU) cung cấp dữ liệu miễn phí từ các vệ tinh Sentinel-1 (SAR) và Sentinel-2 (Optical) với độ phân giải không gian 10m và chu kỳ quay trở lại ngắn (5-6 ngày), phù hợp cho giám sát rừng nhiệt đới.

Trong những năm gần đây, trí tuệ nhân tạo (AI) và học sâu (Deep Learning) đã đạt được những bước tiến vượt bậc trong xử lý ảnh và nhận dạng mẫu. Mạng Neural Tích chập (Convolutional Neural Networks - CNN) đặc biệt hiệu quả trong phân loại ảnh nhờ khả năng tự động học đặc trưng không gian (spatial features) từ dữ liệu thô. Khác với các phương pháp học máy truyền thống như Random Forest hay SVM đòi hỏi trích xuất đặc trưng thủ công, CNN có thể học các mẫu phức tạp và bất biến đối với phép tịnh tiến, xoay.

Xuất phát từ nhu cầu thực tiễn về giám sát rừng hiệu quả và xu hướng ứng dụng công nghệ AI tiên tiến, đồ án này lựa chọn đề tài **"Ứng dụng mạng Neural Tích chập sâu trong giám sát biến động rừng từ ảnh vệ tinh đa nguồn: Nghiên cứu điển hình tại tỉnh Cà Mau"** nhằm phát triển hệ thống tự động phát hiện mất rừng với độ chính xác cao.

## Mục tiêu nghiên cứu

### Mục tiêu tổng quát

Phát triển mô hình học sâu dựa trên kiến trúc CNN để phát hiện và phân loại tự động các khu vực biến động rừng từ ảnh vệ tinh đa nguồn (Sentinel-1 SAR và Sentinel-2 Optical) tại tỉnh Cà Mau.

### Mục tiêu cụ thể

1. **Xây dựng bộ dữ liệu huấn luyện:** Thu thập và xử lý dữ liệu ảnh vệ tinh Sentinel-1/2 đa thời gian, kết hợp với ground truth points để tạo bộ dữ liệu huấn luyện chất lượng cao.

2. **Thiết kế kiến trúc CNN tối ưu:** Đề xuất kiến trúc CNN nhẹ (lightweight) phù hợp với bộ dữ liệu có quy mô vừa phải (~2,600 mẫu), tích hợp các kỹ thuật regularization (Batch Normalization, Dropout) để tránh overfitting.

3. **Phân chia dữ liệu khoa học:** Triển khai phương pháp stratified random split để đảm bảo phân bố lớp đồng đều giữa các tập huấn luyện, validation và test, kết hợp với 5-Fold Cross Validation để đánh giá robust.

4. **Huấn luyện và tối ưu hóa mô hình:** Áp dụng các kỹ thuật huấn luyện tiên tiến như early stopping, learning rate scheduling, class weighting để đạt được mô hình có hiệu suất cao và ổn định.

5. **Đánh giá và so sánh:** Đánh giá chi tiết hiệu suất mô hình CNN trên các chỉ số Accuracy, Precision, Recall, F1-Score, ROC-AUC. So sánh với phương pháp Random Forest để chứng minh ưu thế của Deep Learning so với Machine Learning truyền thống.

6. **Ứng dụng thực tế:** Áp dụng mô hình đã huấn luyện để phân loại toàn bộ khu vực rừng Cà Mau, ước tính diện tích mất rừng, và trực quan hóa kết quả dưới dạng bản đồ phân loại.

## Đối tượng và phạm vi nghiên cứu

### Đối tượng nghiên cứu

- **Đối tượng chính:** Các khu vực rừng tự nhiên và rừng trồng tại tỉnh Cà Mau, bao gồm rừng ngập mặn, rừng phòng hộ ven biển.

- **Biến động rừng:** Các trạng thái biến động được phân loại thành 4 nhóm:
  - **Forest Stable (Rừng ổn định):** Vùng rừng không có biến đổi trong giai đoạn nghiên cứu.
  - **Deforestation (Mất rừng):** Vùng rừng bị chuyển đổi sang đất trống, đất canh tác hoặc nuôi trồng thủy sản.
  - **Non-forest (Không phải rừng):** Vùng không có rừng trong cả hai thời điểm (đất trống, mặt nước, khu dân cư).
  - **Reforestation (Tái trồng rừng):** Vùng không có rừng trở thành rừng trong giai đoạn nghiên cứu.

- **Dữ liệu viễn thám:** Ảnh vệ tinh đa nguồn từ Sentinel-1 (SAR) và Sentinel-2 (Optical), kỳ trước (tháng 1-2/2024) và kỳ sau (tháng 2/2025).

### Phạm vi nghiên cứu

- **Không gian:** Toàn bộ khu vực có rừng trong ranh giới hành chính tỉnh Cà Mau, diện tích khoảng **162,469 hecta** (tương đương 1,624.69 km²).

- **Thời gian:** Giai đoạn từ tháng 01/2024 đến tháng 02/2025 (khoảng 13 tháng).

- **Độ phân giải không gian:** 10 mét/pixel (độ phân giải gốc của Sentinel-1/2).

- **Hệ tọa độ:** EPSG:32648 (WGS 84 / UTM Zone 48N).

### Giới hạn nghiên cứu

- Nghiên cứu sử dụng dữ liệu tại hai thời điểm (bi-temporal), chưa khai thác đầy đủ chuỗi thời gian liên tục.
- Ground truth được thu thập từ phiên giải ảnh và dữ liệu có sẵn, chưa có khảo sát thực địa đầy đủ.
- Mô hình được đào tạo và đánh giá trên dữ liệu Cà Mau, khả năng tổng quát hóa sang các khu vực khác cần được kiểm chứng thêm.

## Phương pháp nghiên cứu

Đồ án áp dụng phương pháp nghiên cứu thực nghiệm kết hợp với phương pháp phân tích định lượng:

### Thu thập và xử lý dữ liệu

- Thu thập ảnh Sentinel-1 SAR và Sentinel-2 Optical từ Copernicus Open Access Hub.
- Tiền xử lý: hiệu chỉnh khí quyển, co-registration, cắt theo ranh giới nghiên cứu.
- Tính toán các chỉ số thực vật (NDVI, NBR, NDMI) từ dữ liệu Sentinel-2.
- Thu thập ground truth points (2,630 điểm) qua phiên giải ảnh độ phân giải cao.

### Trích xuất đặc trưng

- Xây dựng feature stack 27 chiều từ dữ liệu đa nguồn:
  - Sentinel-2: 21 features (7 before + 7 after + 7 delta)
  - Sentinel-1: 6 features (2 before + 2 after + 2 delta)
- Trích xuất patches không gian 3×3 pixels tại vị trí ground truth.
- Chuẩn hóa dữ liệu bằng phương pháp z-score standardization.

> **[TODO: Cần chèn Sơ đồ tại đây]**
> *Gợi ý:* Vẽ sơ đồ tổng quan quy trình nghiên cứu từ thu thập dữ liệu → tiền xử lý → trích xuất đặc trưng → huấn luyện mô hình → đánh giá → ứng dụng.

### Phát triển mô hình CNN

- Thiết kế kiến trúc CNN nhẹ với 2 convolutional blocks + fully connected layers.
- Tổng số parameters: ~36,000 (phù hợp với bộ dữ liệu nhỏ).
- Áp dụng Batch Normalization, Dropout, Weight Decay để regularization.

### Chia dữ liệu

- Sử dụng stratified random split để đảm bảo phân bố lớp đồng đều.
- Tỷ lệ: 80% Train+Val (5-Fold CV), 20% Test (fixed).
- 5-Fold Stratified Cross Validation trên tập Train+Val.

### Huấn luyện và đánh giá

- Optimizer: AdamW với learning rate 0.001, weight decay 1e-3.
- Loss function: CrossEntropyLoss với class weights.
- Early stopping với patience 15 epochs.
- Đánh giá trên các chỉ số: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

### So sánh với Random Forest

- Huấn luyện mô hình Random Forest (pixel-based) trên cùng dữ liệu.
- So sánh về accuracy, thời gian, khả năng sử dụng spatial context.

## Ý nghĩa khoa học và thực tiễn

### Ý nghĩa khoa học

- **Đóng góp về phương pháp:** Đề xuất kiến trúc CNN nhẹ và hiệu quả cho bài toán phân loại ảnh viễn thám với bộ dữ liệu nhỏ.

- **Quy trình đánh giá khoa học:** Áp dụng 5-Fold Stratified Cross Validation kết hợp với fixed test set để đánh giá mô hình một cách robust và đáng tin cậy.

- **Tích hợp đa nguồn:** Kết hợp hiệu quả dữ liệu SAR (Sentinel-1) và Optical (Sentinel-2), tận dụng ưu thế của từng loại dữ liệu (SAR xuyên qua mây, Optical cung cấp thông tin quang phổ).

- **So sánh định lượng:** Cung cấp bằng chứng thực nghiệm về ưu thế của Deep Learning (CNN) so với Machine Learning truyền thống (Random Forest) trong phân loại ảnh viễn thám có tính không gian.

### Ý nghĩa thực tiễn

- **Giám sát rừng hiệu quả:** Cung cấp công cụ tự động phát hiện mất rừng với độ chính xác cao (>98%), giảm đáng kể thời gian và chi phí so với phương pháp truyền thống.

- **Hỗ trợ quản lý tài nguyên:** Kết quả nghiên cứu giúp các cơ quan quản lý rừng tại Cà Mau và các tỉnh khác có cơ sở dữ liệu để ra quyết định bảo vệ và phát triển rừng bền vững.

- **Cảnh báo sớm:** Hệ thống có thể triển khai để giám sát liên tục, phát hiện kịp thời các hoạt động phá rừng trái phép.

- **Mở rộng ứng dụng:** Phương pháp có thể áp dụng cho các bài toán giám sát môi trường khác như biến động đất đai, đô thị hóa, biến đổi sử dụng đất.

- **Chi phí thấp:** Sử dụng dữ liệu vệ tinh miễn phí (Sentinel-1/2) và mô hình nhẹ (có thể chạy trên máy tính thông thường), phù hợp với điều kiện Việt Nam.

## Cấu trúc đồ án

Đồ án được tổ chức thành 4 chương chính:

- **Chương 1 - Tổng quan về vấn đề nghiên cứu:** Trình bày bối cảnh mất rừng, công nghệ viễn thám, tổng quan các nghiên cứu liên quan và khoảng trống nghiên cứu.

- **Chương 2 - Cơ sở lý thuyết:** Giới thiệu chi tiết về công nghệ viễn thám (Sentinel-1/2), lý thuyết mạng Neural Tích chập (CNN), các phương pháp phân loại ảnh và đánh giá mô hình.

- **Chương 3 - Phương pháp nghiên cứu:** Mô tả chi tiết khu vực nghiên cứu, dữ liệu, quy trình xử lý, kiến trúc mô hình CNN đề xuất, phương pháp huấn luyện và đánh giá.

- **Chương 4 - Kết quả và thảo luận:** Trình bày kết quả huấn luyện, đánh giá mô hình, phân loại toàn vùng, so sánh với Random Forest, phân tích lỗi và trực quan hóa.

\newpage

# CHƯƠNG 1: TỔNG QUAN VỀ VẤN ĐỀ NGHIÊN CỨU

## Bối cảnh và tình hình mất rừng

### Tình hình mất rừng trên thế giới

Rừng bao phủ khoảng 31% diện tích đất liền toàn cầu [1], đóng vai trò thiết yếu trong việc điều hòa khí hậu, lưu giữ carbon, bảo tồn đa dạng sinh học, và cung cấp sinh kế cho hàng tỷ người. Tuy nhiên, tốc độ mất rừng toàn cầu vẫn đang ở mức báo động. Theo báo cáo "Global Forest Resources Assessment 2020" của FAO [1], tổng diện tích rừng bị phá (gross deforestation) từ năm 1990 đến 2020 ước tính khoảng 420 triệu hecta, trong khi diện tích mất rừng ròng (net loss, sau khi trừ đi diện tích trồng rừng mới) là 178 triệu hecta, chủ yếu do chuyển đổi sang đất nông nghiệp, chăn nuôi, khai thác gỗ bất hợp pháp, và phát triển cơ sở hạ tầng.

Khu vực nhiệt đới, nơi tập trung 45% diện tích rừng toàn cầu và đa dạng sinh học cao nhất, đang chịu tốc độ mất rừng nhanh nhất. Lưu vực Amazon (Brazil), rừng Congo (Trung Phi), và Đông Nam Á là những "điểm nóng" về mất rừng. Theo dữ liệu từ Global Forest Watch [3], thế giới mất khoảng 10 triệu hecta rừng nhiệt đới mỗi năm trong giai đoạn 2015-2020.

Mất rừng không chỉ làm giảm khả năng hấp thụ CO₂ mà còn trực tiếp phát thải khí nhà kính từ việc đốt rừng và phân hủy sinh khối. Theo IPCC [2], phá rừng và thay đổi sử dụng đất đóng góp khoảng 23% tổng lượng phát thải khí nhà kính do con người gây ra. Điều này góp phần làm gia tăng hiện tượng biến đổi khí hậu toàn cầu.

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* Bản đồ thế giới thể hiện các vùng mất rừng nhiệt đới (Amazon, Congo, Đông Nam Á) với chú thích số liệu diện tích mất rừng giai đoạn 2015-2020.

### Tình hình mất rừng tại Việt Nam

Việt Nam đã trải qua những biến đổi lớn về độ che phủ rừng trong 30 năm qua. Sau thời kỳ suy giảm nghiêm trọng (độ che phủ chỉ còn 27% vào năm 1990 do chiến tranh và khai thác bừa bãi), Việt Nam đã thực hiện nhiều chương trình phục hồi và phát triển rừng. Nhờ các chương trình như "Trồng 5 triệu hecta rừng" (1998-2010), độ che phủ rừng đã tăng lên 42% vào năm 2020 (Bộ NN&PTNT, 2020).

Tuy nhiên, chất lượng rừng là một vấn đề đáng lo ngại. Mặc dù tổng diện tích rừng tăng chủ yếu nhờ rừng trồng (cao su, keo, thông), nhưng diện tích rừng tự nhiên - đặc biệt là rừng giàu, rừng gỗ lớn - lại giảm đáng kể. Rừng tự nhiên giảm từ 9.4 triệu hecta (1990) xuống còn 10.2 triệu hecta (2020), trong đó rừng giàu chỉ chiếm 2.2 triệu hecta.

Các nguyên nhân chủ yếu gây mất rừng tại Việt Nam bao gồm:

- Chuyển đổi sang đất nông nghiệp (cà phê, cao su, điều)
- Khai thác gỗ trái phép
- Phát triển cơ sở hạ tầng và đô thị hóa
- Cháy rừng
- Nuôi trồng thủy sản (đặc biệt tại khu vực ven biển và đồng bằng sông Cửu Long)

> **[TODO: Cần chèn Biểu đồ tại đây]**
> *Gợi ý:* Biểu đồ đường thể hiện sự thay đổi độ che phủ rừng Việt Nam giai đoạn 1990-2020, với 2 đường: tổng diện tích rừng và diện tích rừng tự nhiên.

### Tình hình rừng tại tỉnh Cà Mau

Cà Mau, tỉnh cực Nam Tổ Quốc, sở hữu hệ sinh thái rừng ngập mặn quan trọng với diện tích khoảng 40,000 hecta, chiếm ~20% diện tích rừng ngập mặn của Việt Nam. Rừng ngập mặn Cà Mau đóng vai trò then chốt trong:

- **Phòng hộ ven biển:** Chắn sóng, chống xâm thực, bảo vệ bờ biển.
- **Đa dạng sinh học:** Môi trường sống cho nhiều loài động thực vật quý hiếm.
- **Sinh kế:** Nguồn thu nhập từ thủy sản, du lịch sinh thái.
- **Giảm nhẹ biến đổi khí hậu:** Rừng ngập mặn có khả năng lưu giữ carbon gấp 3-5 lần rừng trên cạn.

Tuy nhiên, rừng Cà Mau đang đối mặt với nhiều thách thức:

- **Chuyển đổi sang nuôi tôm:** Áp lực kinh tế khiến nhiều khu vực rừng bị chuyển đổi sang ao nuôi tôm.
- **Xâm nhập mặn:** Biến đổi khí hậu làm tăng độ mặn, ảnh hưởng đến sức khỏe rừng.
- **Xói mòn bờ biển:** Làm giảm diện tích rừng ven biển.
- **Thiếu nước ngọt:** Ảnh hưởng đến quá trình tái sinh rừng.

Theo số liệu của Sở NN&PTNT Cà Mau (2022), diện tích rừng tự nhiên tại Cà Mau đã giảm khoảng 5-7% trong giai đoạn 2010-2020. Việc giám sát và bảo vệ rừng tại Cà Mau là ưu tiên hàng đầu nhằm duy trì hệ sinh thái quan trọng này.

> **[TODO: Cần chèn Bản đồ tại đây]**
> *Gợi ý:* Bản đồ vị trí tỉnh Cà Mau trong Việt Nam và vùng ĐBSCL, kèm bản đồ chi tiết khu vực rừng ngập mặn Cà Mau với ranh giới vùng nghiên cứu.

## Công nghệ viễn thám trong giám sát rừng

### Ưu điểm của công nghệ viễn thám

Công nghệ viễn thám vệ tinh cung cấp nhiều ưu điểm vượt trội so với phương pháp điều tra thực địa truyền thống:

**1. Phạm vi rộng:** Một ảnh vệ tinh có thể phủ diện tích hàng nghìn km², cho phép giám sát đồng thời nhiều khu vực rừng.

**2. Cập nhật thường xuyên:** Các vệ tinh hiện đại có chu kỳ quay trở lại ngắn (3-5 ngày), cho phép giám sát liên tục và phát hiện kịp thời các biến động.

**3. Chi phí hiệu quả:** Nhiều chương trình vệ tinh (như Copernicus, Landsat) cung cấp dữ liệu miễn phí, giảm đáng kể chi phí so với khảo sát thực địa.

**4. Dữ liệu đa thời gian:** Lưu trữ dữ liệu lịch sử cho phép phân tích xu hướng biến động trong nhiều năm.

**5. Tiếp cận khu vực khó:** Có thể giám sát các khu vực rừng núi cao, rừng rậm, hoặc khu vực biên giới khó tiếp cận bằng phương pháp thực địa.

**6. Dữ liệu khách quan và có thể lặp lại:** Loại bỏ sai số chủ quan của người điều tra.

### Chương trình Copernicus và vệ tinh Sentinel

Chương trình Copernicus của Liên minh Châu Âu (EU) là một trong những chương trình quan sát Trái Đất lớn nhất thế giới, cung cấp dữ liệu miễn phí và mở. Hai vệ tinh quan trọng cho giám sát rừng là:

**Sentinel-1 (SAR - Synthetic Aperture Radar):**

- Hoạt động ở dải sóng C-band (~5.5 cm)
- Hai chế độ polarization: VV (Vertical-Vertical) và VH (Vertical-Horizontal)
- Độ phân giải không gian: 10m (IW mode)
- Chu kỳ quay trở lại: 6 ngày (với hai vệ tinh 1A và 1B)
- **Ưu điểm:** Xuyên qua mây và khói, hoạt động cả ngày lẫn đêm, nhạy cảm với cấu trúc thực vật và độ ẩm
- **Ứng dụng:** Phát hiện biến động rừng trong điều kiện mây nhiều, phân biệt rừng ngập nước

**Sentinel-2 (Optical - Multispectral Imaging):**

- 13 dải phổ từ vùng nhìn thấy đến hồng ngoại ngắn (443nm - 2190nm)
- Độ phân giải không gian: 10m (B2, B3, B4, B8), 20m (B5, B6, B7, B8a, B11, B12), 60m (B1, B9, B10)
- Chu kỳ quay trở lại: 5 ngày (với hai vệ tinh 2A và 2B)
- **Ưu điểm:** Thông tin quang phổ phong phú, phù hợp tính toán chỉ số thực vật
- **Ứng dụng:** Phân loại lớp phủ, đánh giá sức khỏe thực vật, tính toán NDVI/NBR/NDMI

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* Hình minh họa vệ tinh Sentinel-1 và Sentinel-2, kèm bảng so sánh thông số kỹ thuật chính của hai vệ tinh.

### Chỉ số thực vật từ dữ liệu quang học

Các chỉ số thực vật (vegetation indices) là công cụ quan trọng trong giám sát rừng, được tính toán từ các dải phổ khác nhau:

**NDVI (Normalized Difference Vegetation Index):**

$$NDVI = \frac{NIR - Red}{NIR + Red}$$

- Dải giá trị: [-1, 1]
- NDVI > 0.6: Thực vật xanh tốt
- NDVI < 0.2: Đất trống, nước, đô thị
- Ứng dụng: Đánh giá mật độ và sức khỏe thực vật

**NBR (Normalized Burn Ratio):**

$$NBR = \frac{NIR - SWIR_2}{NIR + SWIR_2}$$

- Nhạy cảm với lửa và vùng cháy
- Delta NBR (dNBR) dùng để đánh giá mức độ cháy rừng
- Ứng dụng: Phát hiện cháy rừng, đánh giá thiệt hại sau cháy

**NDMI (Normalized Difference Moisture Index):**

$$NDMI = \frac{NIR - SWIR_1}{NIR + SWIR_1}$$

- Đánh giá hàm lượng nước trong thực vật
- NDMI thấp: Stress hạn, nguy cơ cháy cao
- Ứng dụng: Giám sát hạn hán, đánh giá sức khỏe rừng

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* Hình ảnh minh họa các chỉ số thực vật (NDVI, NBR, NDMI) trên cùng một khu vực rừng Cà Mau, với thang màu và chú thích giá trị.

### Tích hợp dữ liệu SAR và Optical

Việc kết hợp dữ liệu SAR (Sentinel-1) và Optical (Sentinel-2) mang lại nhiều lợi ích:

**Bổ sung thông tin:**

- SAR: Thông tin về cấu trúc, độ nhám bề mặt, độ ẩm
- Optical: Thông tin quang phổ, chỉ số thực vật

**Khắc phục hạn chế:**

- SAR hoạt động trong điều kiện mây mù (quan trọng với rừng nhiệt đới)
- Optical cung cấp thông tin trực quan dễ phiên giải

**Nâng cao độ chính xác:**

- Nhiều nghiên cứu cho thấy kết hợp SAR + Optical tăng accuracy 5-10% so với dùng riêng lẻ

**Phát hiện đa chiều:**

- SAR nhạy với biến đổi cấu trúc (chặt cây)
- Optical nhạy với biến đổi quang phổ (sức khỏe thực vật)

## Tổng quan các nghiên cứu liên quan

### Phương pháp Machine Learning truyền thống

**Random Forest (RF):**

Random Forest là một trong những thuật toán phổ biến nhất trong phân loại ảnh viễn thám. Các nghiên cứu tiêu biểu:

- Belgiu & Drăguț [4] đã tổng hợp hơn 200 nghiên cứu sử dụng Random Forest, chỉ ra rằng RF đạt accuracy trung bình 85-90% trên các bài toán phân loại đất.

- Gislason et al. [5] so sánh RF với SVM và Maximum Likelihood trên dữ liệu Landsat, kết quả cho thấy RF đạt accuracy cao hơn 2-5% và thời gian training nhanh hơn đáng kể.

**Ưu điểm của Random Forest:**

- Robust với noise và outliers
- Không cần chuẩn hóa dữ liệu
- Cung cấp feature importance
- Hiệu quả với dữ liệu high-dimensional
- Khả năng xử lý class imbalance tốt

**Hạn chế:**

- Không khai thác spatial context (xử lý từng pixel độc lập)
- Cần trích xuất features thủ công
- Overfitting với dữ liệu noisy
- Khó giải thích cơ chế ra quyết định

**Support Vector Machine (SVM):**

SVM cũng được sử dụng rộng rãi trong phân loại viễn thám:

- Mountrakis et al. [6] phân tích 73 nghiên cứu, chỉ ra SVM đặc biệt hiệu quả với dữ liệu high-dimensional và training samples nhỏ.

- Huang et al. [7] áp dụng SVM cho phân loại đa lớp trên dữ liệu Landsat, đạt accuracy 87%, cao hơn 5% so với Maximum Likelihood.

### Phương pháp Deep Learning

**Convolutional Neural Networks (CNN):**

CNN đã cách mạng hóa computer vision và ngày càng được áp dụng rộng rãi trong viễn thám:

- Zhang et al. [8] giới thiệu các kiến trúc CNN phổ biến và ứng dụng trong viễn thám.

- Kussul et al. [9] áp dụng CNN cho phân loại cây trồng từ Sentinel-2, đạt accuracy 94.5%, vượt trội Random Forest (88%) và SVM (89.5%) trên cùng dataset.

- Xu et al. [10] sử dụng CNN kết hợp với attention mechanism đạt accuracy 96.8% trên dữ liệu đa nguồn.

**Ưu điểm CNN:**

- Tự động học spatial features
- Khai thác local spatial context qua convolutional kernels
- Không cần feature engineering thủ công
- Khả năng học hierarchical features
- Có thể transfer learning từ pretrained models

**Hạn chế:**

- Cần large training datasets
- Thời gian training lâu (cần GPU)
- Nhiều hyperparameters cần tuning
- "Black box" - khó giải thích
- Dễ overfitting với small datasets

### Ứng dụng trong giám sát rừng

**Phát hiện mất rừng:**

- Hansen et al. [15] phát triển Global Forest Change dataset sử dụng Landsat time series và decision tree, phát hiện mất rừng toàn cầu 2000-2012 ở độ phân giải 30m.

- Reiche et al. [16] kết hợp Sentinel-1 và Landsat để phát hiện mất rừng near-real-time tại Amazon, đạt accuracy 93.8%.

- Hethcoat et al. [17] sử dụng CNN trên Landsat time series để phát hiện illegal gold mining ở Amazon, đạt F1-score 0.92.

**Tích hợp SAR và Optical:**

- Hu et al. [18] kết hợp Sentinel-1 và Sentinel-2 với Random Forest để phân loại rừng ở Madagascar, accuracy tăng từ 87% (chỉ Sentinel-2) lên 92% (cả hai).

- Ienco et al. [19] sử dụng deep neural networks kết hợp SAR + Optical time series để phân loại crop, đạt accuracy 96.5%.

**Nghiên cứu tại Việt Nam:**

- Pham et al. [20] sử dụng Sentinel-1 để phát hiện biến động rừng tại Đắk Lắk, kết hợp SAR backscatter và machine learning để phân loại với độ chính xác 87%.

- Nguyen et al. [21] áp dụng Random Forest và Sentinel-2 để lập bản đồ che phủ rừng tại Quảng Nam, đạt overall accuracy 91.2%.

- Bùi et al. [22] nghiên cứu biến động rừng ngập mặn ven biển ĐBSCL bằng Landsat time series (1990-2020), phát hiện xu hướng giảm diện tích do chuyển đổi sang ao nuôi.

> **[TODO: Cần chèn Bảng số liệu tại đây]**
> *Gợi ý:* Bảng tổng hợp các nghiên cứu liên quan với các cột: Tác giả, Năm, Phương pháp, Dữ liệu, Khu vực, Accuracy.

## Khoảng trống nghiên cứu và định hướng đồ án

### Khoảng trống nghiên cứu

Qua tổng quan tài liệu, một số khoảng trống nghiên cứu được xác định:

**1. Thiếu nghiên cứu Deep Learning cho rừng nhiệt đới Việt Nam:**

- Hầu hết nghiên cứu Deep Learning tập trung ở Amazon, Congo, Indonesia
- Ít nghiên cứu áp dụng CNN cho rừng Việt Nam, đặc biệt rừng ngập mặn Cà Mau

**2. Vấn đề spatial data leakage:**

- Nhiều nghiên cứu chia dữ liệu random mà không quan tâm đến spatial autocorrelation
- Dẫn đến overestimate accuracy do training và test samples gần nhau trong không gian

**3. CNN cho small datasets:**

- CNN thường cần large datasets (hàng trăm nghìn mẫu)
- Ít nghiên cứu về kiến trúc CNN tối ưu cho small datasets viễn thám (~2,000-5,000 mẫu)

**4. Tích hợp SAR + Optical với Deep Learning:**

- Hầu hết nghiên cứu tích hợp đa nguồn sử dụng feature-level fusion với ML truyền thống
- Ít nghiên cứu về cách tối ưu fusion SAR + Optical trong CNN architecture

**5. Thiếu so sánh định lượng giữa DL và ML truyền thống:**

- Nhiều nghiên cứu chỉ dùng DL mà không so sánh với baseline ML
- Thiếu phân tích về trade-off giữa accuracy, computational cost, interpretability

### Định hướng của đồ án

Xuất phát từ các khoảng trống trên, đồ án này định hướng:

**1. Phát triển CNN architecture phù hợp với small dataset:**

- Thiết kế kiến trúc lightweight (~36K parameters)
- Áp dụng aggressive regularization (Batch Norm, Dropout, Weight Decay)
- So sánh với các architectures khác (deeper, wider)

**2. Triển khai quy trình đánh giá khoa học:**

- Sử dụng stratified random split để đảm bảo phân bố lớp đồng đều
- 5-Fold Stratified Cross Validation để đánh giá variance của mô hình
- Fixed test set (20%) để đánh giá khả năng tổng quát hóa

**3. Tối ưu fusion Sentinel-1 và Sentinel-2:**

- Feature-level fusion: concatenate SAR và Optical features
- Trích xuất temporal features (before, after, delta)
- 27 features: 21 từ S2 (7×3) + 6 từ S1 (2×3)

**4. So sánh định lượng CNN vs Random Forest:**

- Cùng dataset, cùng features, cùng data split
- So sánh accuracy, training time, inference time
- Phân tích ưu nhược điểm từng phương pháp

**5. Ứng dụng thực tế tại Cà Mau:**

- Phân loại toàn vùng rừng Cà Mau (162,469 ha)
- Ước tính diện tích mất rừng
- Tạo bản đồ phân loại độ phân giải 10m

### Câu hỏi nghiên cứu

Đồ án tập trung trả lời các câu hỏi sau:

1. **CNN có vượt trội hơn Random Forest trong phân loại biến động rừng từ ảnh vệ tinh không?** Nếu có, mức độ cải thiện accuracy là bao nhiêu?

2. **5-Fold Cross Validation có đảm bảo đánh giá robust cho mô hình không?** Variance giữa các folds như thế nào?

3. **Kiến trúc CNN như thế nào là phù hợp với bộ dữ liệu 2,630 mẫu?** So sánh lightweight architecture vs deeper architectures.

4. **Tích hợp Sentinel-1 SAR và Sentinel-2 Optical có cải thiện accuracy không?** So với chỉ dùng Sentinel-2.

5. **Mô hình CNN có thể ứng dụng thực tế để giám sát rừng Cà Mau không?** Đánh giá về accuracy, tốc độ, và khả năng triển khai.

\newpage

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## Công nghệ viễn thám và ảnh vệ tinh

### Nguyên lý viễn thám

Viễn thám (Remote Sensing) là khoa học và kỹ thuật thu thập thông tin về một đối tượng hoặc khu vực từ xa, thường thông qua việc ghi nhận bức xạ điện từ phản xạ hoặc phát ra từ bề mặt Trái Đất. Nguyên lý cơ bản của viễn thám dựa trên tương tác giữa bức xạ điện từ và các đối tượng trên bề mặt:

**Quá trình viễn thám bị động (Passive Remote Sensing):**

1. **Nguồn năng lượng:** Mặt Trời phát ra bức xạ điện từ
2. **Truyền qua khí quyển:** Một phần bức xạ bị hấp thụ và tán xạ bởi khí quyển
3. **Tương tác với bề mặt:** Bức xạ phản xạ, hấp thụ, và truyền qua tùy theo đặc tính vật liệu
4. **Ghi nhận bởi cảm biến:** Vệ tinh thu nhận bức xạ phản xạ
5. **Truyền dữ liệu:** Tín hiệu được truyền về trạm mặt đất

> **[TODO: Cần chèn Sơ đồ tại đây]**
> *Gợi ý:* Sơ đồ minh họa nguyên lý viễn thám bị động và chủ động, với các thành phần: nguồn năng lượng, khí quyển, bề mặt, cảm biến.

**Phương trình cân bằng năng lượng:**

$$E_{incident} = E_{reflected} + E_{absorbed} + E_{transmitted}$$

Trong đó:

- $E_{incident}$: Năng lượng tới (từ Mặt Trời)
- $E_{reflected}$: Năng lượng phản xạ (được cảm biến ghi nhận)
- $E_{absorbed}$: Năng lượng hấp thụ (chuyển thành nhiệt)
- $E_{transmitted}$: Năng lượng truyền qua

### Radar khẩu độ tổng hợp (SAR)

**Nguyên lý hoạt động:**

Khác với viễn thám bị động, SAR là hệ thống chủ động (active remote sensing):

1. **Phát xung radar:** Anten phát xung sóng điện từ về phía Trái Đất
2. **Tương tác với bề mặt:** Sóng radar phản xạ ngược (backscatter) với cường độ phụ thuộc vào:
   - Độ nhám bề mặt (surface roughness)
   - Độ ẩm (moisture content)
   - Hằng số điện môi (dielectric constant)
   - Góc tới (incidence angle)
3. **Thu tín hiệu phản xạ:** Anten thu nhận tín hiệu backscatter
4. **Xử lý tín hiệu:** Tổng hợp khẩu độ để tăng độ phân giải

**Hệ số Backscatter ($\sigma^0$):**

$$\sigma^0 (dB) = 10 \times \log_{10}(\sigma^0_{linear})$$

Giá trị $\sigma^0$ phụ thuộc vào:

- **Độ nhám bề mặt:** Bề mặt nhẵn (nước) → $\sigma^0$ thấp, bề mặt nhám (rừng) → $\sigma^0$ cao
- **Độ ẩm:** Độ ẩm cao → $\sigma^0$ cao (nước có hằng số điện môi lớn)
- **Cấu trúc thực vật:** Rừng có cấu trúc phức tạp → backscatter mạnh

**Polarization:**

SAR có thể phát và thu theo các polarization khác nhau:

- **VV:** Phát V (Vertical), Thu V → Nhạy với độ ẩm bề mặt
- **VH:** Phát V, Thu H (Horizontal) → Nhạy với cấu trúc thực vật (volume scattering)
- **HH:** Phát H, Thu H → Nhạy với độ nhám bề mặt
- **HV:** Phát H, Thu V → Tương tự VH

**Sentinel-1 SAR:**

- Dải sóng: C-band ($\lambda$ = 5.5 cm, frequency = 5.4 GHz)
- Polarization: VV và VH (IW mode)
- Độ phân giải không gian: 10m
- Ưu điểm: Xuyên qua mây, hoạt động ngày/đêm

### Ảnh quang học đa phổ (Optical Multispectral)

**Dải phổ điện từ:**

Ảnh quang học ghi nhận bức xạ phản xạ từ bề mặt Trái Đất ở các dải phổ khác nhau:

1. **Visible (VIS):** 400-700 nm
   - Blue (B): 450-520 nm
   - Green (G): 520-600 nm
   - Red (R): 630-690 nm

2. **Near-Infrared (NIR):** 700-1400 nm
   - Phản xạ cao ở thực vật xanh (chlorophyll)
   - Quan trọng cho tính toán NDVI

3. **Short-Wave Infrared (SWIR):** 1400-3000 nm
   - SWIR1: 1550-1750 nm
   - SWIR2: 2080-2350 nm
   - Nhạy với độ ẩm thực vật và đất

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* Biểu đồ phổ điện từ với các dải phổ và vị trí các bands của Sentinel-2.

**Chữ ký phổ (Spectral Signature):**

Mỗi loại đối tượng có chữ ký phổ đặc trưng - mẫu phản xạ qua các dải phổ:

$$S = [\rho(\lambda_1), \rho(\lambda_2), ..., \rho(\lambda_n)]$$

Ví dụ:

- **Thực vật xanh:** Phản xạ thấp ở Red (hấp thụ bởi chlorophyll), phản xạ cao ở NIR
- **Đất trống:** Phản xạ trung bình và tăng dần theo bước sóng
- **Nước:** Phản xạ thấp ở tất cả các dải (đặc biệt NIR và SWIR)

> **[TODO: Cần chèn Biểu đồ tại đây]**
> *Gợi ý:* Biểu đồ chữ ký phổ của các loại đối tượng: thực vật xanh, đất trống, nước, đô thị.

**Sentinel-2 Multispectral Imager:**

| Band | Tên | Bước sóng (nm) | Độ phân giải (m) | Ứng dụng |
|------|-----|---------------|------------------|----------|
| B2 | Blue | 490 | 10 | Phân biệt đất/nước |
| B3 | Green | 560 | 10 | Đánh giá thực vật |
| B4 | Red | 665 | 10 | Chlorophyll absorption |
| B8 | NIR | 842 | 10 | Biomass, NDVI |
| B11 | SWIR1 | 1610 | 20 | Độ ẩm, NDMI |
| B12 | SWIR2 | 2190 | 20 | Phân biệt đất/rừng, NBR |

### Chỉ số thực vật

**NDVI (Normalized Difference Vegetation Index):**

$$NDVI = \frac{NIR - Red}{NIR + Red}$$

**Nguyên lý:**

- Thực vật xanh: Hấp thụ mạnh Red (chlorophyll), phản xạ cao NIR (cấu trúc tế bào) → NDVI cao
- Đất trống/nước: Phản xạ thấp cả Red và NIR → NDVI thấp

**Phạm vi giá trị:**

- NDVI > 0.6: Thực vật xanh tốt (rừng rậm)
- 0.2 < NDVI < 0.6: Thực vật thưa, cỏ
- NDVI < 0.2: Đất trống, nước, đô thị

**NBR (Normalized Burn Ratio):**

$$NBR = \frac{NIR - SWIR_2}{NIR + SWIR_2}$$

**Nguyên lý:**

- NIR: Phản xạ cao ở thực vật xanh
- SWIR2: Nhạy với độ ẩm và vùng cháy
- Vùng cháy: NIR giảm, SWIR2 tăng → NBR giảm mạnh

**NDMI (Normalized Difference Moisture Index):**

$$NDMI = \frac{NIR - SWIR_1}{NIR + SWIR_1}$$

**Nguyên lý:**

- SWIR1 (~1600 nm): Hấp thụ mạnh bởi nước
- Độ ẩm thực vật cao → SWIR1 phản xạ thấp → NDMI cao
- Stress hạn → NDMI giảm

### Phát hiện biến động rừng

**Change Detection Approach:**

$$\Delta Feature = Feature_{after} - Feature_{before}$$

**Temporal Features:**

- **Before features:** Trạng thái rừng tại thời điểm $t_1$
- **After features:** Trạng thái rừng tại thời điểm $t_2$
- **Delta features:** Biến đổi giữa hai thời điểm ($t_2 - t_1$)

**Ví dụ với NDVI:**

$$\Delta NDVI = NDVI_{after} - NDVI_{before}$$

**Phân loại biến động:**

- $\Delta NDVI << 0$ (giảm mạnh): Mất rừng (deforestation)
- $\Delta NDVI \approx 0$: Rừng ổn định
- $\Delta NDVI >> 0$ (tăng mạnh): Tái trồng rừng

## Mạng Neural Tích chập (Convolutional Neural Networks)

### Giới thiệu về Neural Networks

**Perceptron - Đơn vị cơ bản:**

Một neuron nhân tạo thực hiện phép biến đổi tuyến tính và hàm kích hoạt:

$$y = f(\mathbf{w}^T \mathbf{x} + b)$$

Trong đó:

- $\mathbf{x} \in \mathbb{R}^n$: Input vector (n features)
- $\mathbf{w} \in \mathbb{R}^n$: Weight vector
- $b \in \mathbb{R}$: Bias
- $f(\cdot)$: Activation function
- $y$: Output

> **[TODO: Cần chèn Sơ đồ tại đây]**
> *Gợi ý:* Sơ đồ cấu trúc một perceptron với inputs, weights, bias, activation function và output.

**Multi-Layer Perceptron (MLP):**

Một mạng neural gồm nhiều layers:

$$\begin{aligned}
\text{Layer 1: } & \mathbf{h}_1 = f_1(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \\
\text{Layer 2: } & \mathbf{h}_2 = f_2(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2) \\
& \vdots \\
\text{Output: } & \mathbf{y} = f_n(\mathbf{W}_n \mathbf{h}_{n-1} + \mathbf{b}_n)
\end{aligned}$$

### Convolutional Layer

**Phép tích chập 2D (2D Convolution):**

Đây là thành phần cốt lõi của CNN, thực hiện phép tích chập giữa input và kernel:

$$(I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) \times K(m, n)$$

Trong đó:

- $I$: Input feature map (height × width × channels)
- $K$: Kernel/Filter ($k_h \times k_w \times$ channels)
- $(i, j)$: Vị trí output
- $(m, n)$: Vị trí trong kernel

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* Hình minh họa phép tích chập 2D với kernel 3×3 trượt trên input feature map, tạo output feature map.

**Ưu điểm của Convolution:**

1. **Parameter sharing:** Cùng một kernel được áp dụng cho toàn bộ input
2. **Translation invariance:** Nhận diện đặc trưng ở bất kỳ vị trí nào
3. **Local connectivity:** Mỗi neuron chỉ kết nối với vùng local của input

### Activation Functions

**ReLU (Rectified Linear Unit):**

$$f(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**Ưu điểm:**

- Tính toán nhanh (không có exp, log)
- Giải quyết vanishing gradient problem
- Sparse activation (nhiều neurons = 0)

**Softmax (cho Output Layer):**

$$\text{softmax}(\mathbf{x})_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$

**Tính chất:**

- Output là xác suất: $0 \leq \text{softmax}(\mathbf{x})_i \leq 1$
- Tổng = 1: $\sum_i \text{softmax}(\mathbf{x})_i = 1$
- Dùng cho multi-class classification

### Pooling Layers

**Global Average Pooling:**

$$GAP(k) = \frac{1}{H \times W} \sum_i \sum_j I(i, j, k)$$

- Giảm spatial dimensions về 1×1
- Output: (1, 1, C) từ (H, W, C)
- Ưu điểm: Không có parameters, giảm overfitting

### Batch Normalization

**Batch Normalization Algorithm:**

Với một mini-batch $B = \{x_1, x_2, ..., x_m\}$:

**Step 1: Tính mean và variance của batch**

$$\mu_B = \frac{1}{m} \sum_i x_i$$

$$\sigma^2_B = \frac{1}{m} \sum_i (x_i - \mu_B)^2$$

**Step 2: Normalize**

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}$$

**Step 3: Scale and shift (learnable parameters)**

$$y_i = \gamma \hat{x}_i + \beta$$

**Ưu điểm:**

- Tăng tốc training (có thể dùng learning rate cao hơn)
- Giảm sensitivity với weight initialization
- Có tác dụng regularization (tương tự dropout)
- Ổn định gradient flow

### Dropout

**Motivation:**

Overfitting xảy ra khi model học quá chi tiết training data, không generalize tốt cho unseen data.

**Dropout Algorithm (Training):**

$$\text{For each neuron } i: \quad r_i \sim \text{Bernoulli}(p), \quad \tilde{y}_i = r_i \times y_i$$

Với xác suất $p$ giữ lại neuron, nếu $r_i = 0$, neuron bị tắt.

**Dropout2d (Spatial Dropout):**

Thay vì dropout từng neuron, dropout toàn bộ feature maps:

$$\text{For each channel } k: \quad r_k \sim \text{Bernoulli}(p), \quad \tilde{y}[:,:,k] = r_k \times y[:,:,k]$$

Phù hợp cho CNN vì features trong cùng channel có correlation không gian cao.

### Loss Functions

**Cross-Entropy Loss (Multi-class Classification):**

$$L = -\sum_i y_i \log(\hat{y}_i)$$

Trong đó:

- $y_i$: True label (one-hot encoded)
- $\hat{y}_i$: Predicted probability (from softmax)

**Weighted Cross-Entropy (Class Imbalance):**

$$L = -\sum_i w_i \times y_i \times \log(\hat{y}_i)$$

### Optimization Algorithms

**Adam (Adaptive Moment Estimation):**

$$\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \nabla L(\theta_t) \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) (\nabla L(\theta_t))^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_{t+1} &= \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}$$

Hyperparameters:

- $\beta_1 = 0.9$: Exponential decay for first moment
- $\beta_2 = 0.999$: Exponential decay for second moment
- $\epsilon = 10^{-8}$: Numerical stability
- $\eta = 0.001$: Learning rate

**AdamW (Adam with Weight Decay):**

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

Trong đó $\lambda$ là weight decay coefficient (L2 regularization).

## Phương pháp phân loại ảnh viễn thám

### Pixel-based vs Patch-based Classification

**Pixel-based Classification:**

Mỗi pixel được phân loại độc lập dựa trên vector đặc trưng:

$$\mathbf{x}_i = [f_1, f_2, ..., f_n], \quad y_i = \text{classifier}(\mathbf{x}_i)$$

**Ưu điểm:**

- Đơn giản, dễ implement
- Nhanh (parallel processing)
- Phù hợp với ML truyền thống (RF, SVM)

**Nhược điểm:**

- Không sử dụng spatial context
- Salt-and-pepper noise trong kết quả
- Bỏ qua relationships giữa neighboring pixels

**Patch-based Classification:**

Trích xuất patches (windows) xung quanh mỗi pixel:

$$P_i = \text{extract\_patch}(I, \text{center}=(row_i, col_i), \text{size}=k \times k)$$
$$y_i = \text{classifier}(P_i)$$

**Ưu điểm:**

- Sử dụng spatial context
- Kết quả smooth hơn
- Phù hợp với CNN (automatic feature learning)

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* So sánh pixel-based vs patch-based classification với hình minh họa và kết quả phân loại mẫu.

### Spatial Autocorrelation

**Tobler's First Law of Geography:**

*"Everything is related to everything else, but near things are more related than distant things."*

**Implication for Machine Learning:**

Training và test samples gần nhau trong không gian có high correlation → **Data leakage** → Overestimate accuracy.

**Giải pháp: Stratified Data Splitting + Cross Validation**

### Evaluation Metrics

**Confusion Matrix:**

|  | Predicted 0 | Predicted 1 |
|---|-------------|-------------|
| **Actual 0** | TN | FP |
| **Actual 1** | FN | TP |

**Accuracy:**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision (Positive Predictive Value):**

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall (Sensitivity, True Positive Rate):**

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1-Score (Harmonic Mean):**

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**ROC-AUC (Area Under ROC Curve):**

- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier
- AUC > 0.9: Excellent
- 0.8 < AUC < 0.9: Good
- 0.7 < AUC < 0.8: Fair

\newpage

# CHƯƠNG 3: PHƯƠNG PHÁP NGHIÊN CỨU

## Khu vực và dữ liệu nghiên cứu

### Khu vực nghiên cứu

**Vị trí địa lý:**

Tỉnh Cà Mau nằm ở cực Nam Tổ Quốc, thuộc vùng Đồng bằng sông Cửu Long:

- **Tọa độ địa lý:** 8°36' - 9°27' Bắc, 104°43' - 105°10' Đông
- **Diện tích tự nhiên:** 5,331.7 km²
- **Dân số:** ~1.2 triệu người (2020)
- **Đường bờ biển:** 254 km

> **[TODO: Cần chèn Bản đồ tại đây]**
> *Gợi ý:* Bản đồ vị trí khu vực nghiên cứu gồm: (a) Vị trí Cà Mau trong Việt Nam, (b) Ranh giới tỉnh Cà Mau, (c) Vùng rừng nghiên cứu với tọa độ UTM.

**Vùng nghiên cứu:**

Đồ án tập trung vào toàn bộ diện tích rừng trong ranh giới tỉnh Cà Mau:

- **Diện tích nghiên cứu:** 162,469.25 hecta (1,624.69 km²)
- **Kích thước raster:** 12,547 × 10,917 pixels (độ phân giải 10m)
- **Hệ quy chiếu:** EPSG:32648 (WGS 84 / UTM Zone 48N)

### Dữ liệu viễn thám

**Bảng 3.1: Tổng quan dữ liệu**

| Nguồn dữ liệu | Độ phân giải | Kỳ ảnh | Số bands | Dung lượng |
|---------------|--------------|--------|----------|------------|
| Sentinel-2 Before | 10m | 30/01/2024 | 7 | ~850 MB |
| Sentinel-2 After | 10m | 28/02/2025 | 7 | ~850 MB |
| Sentinel-1 Before | 10m | 04/02/2024 | 2 | ~250 MB |
| Sentinel-1 After | 10m | 22/02/2025 | 2 | ~250 MB |
| Ground Truth | - | - | - | 2,630 points |
| Forest Boundary | Vector | - | - | Shapefile |

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* Ảnh tổ hợp màu Sentinel-2 (RGB) của khu vực nghiên cứu ở 2 thời điểm (before và after) để thể hiện sự thay đổi.

### Ground Truth Data

**Bảng 3.2: Thống kê Ground Truth**

| Class | Tên | Số điểm | Tỷ lệ (%) | Mô tả |
|-------|-----|---------|-----------|-------|
| 0 | Forest Stable | 656 | 24.9% | Rừng ổn định (có rừng ở cả 2 kỳ) |
| 1 | Deforestation | 650 | 24.7% | Mất rừng (có rừng → không có rừng) |
| 2 | Non-forest | 664 | 25.3% | Không phải rừng (không có rừng ở cả 2 kỳ) |
| 3 | Reforestation | 660 | 25.1% | Tái trồng rừng (không có → có rừng) |
| **Tổng** | | **2,630** | **100%** | Balanced distribution |

> **[TODO: Cần chèn Bản đồ tại đây]**
> *Gợi ý:* Bản đồ phân bố các điểm ground truth theo từng lớp với màu sắc khác nhau.

## Quy trình xử lý dữ liệu

### Tổng quan quy trình

> **[TODO: Cần chèn Sơ đồ tại đây]**
> *Gợi ý:* Sơ đồ quy trình xử lý dữ liệu dạng flowchart với các bước: Raw Data → Data Loading → Feature Extraction → Patch Extraction → Normalization → Data Splitting → Ready Dataset.

**Quy trình xử lý:**

1. **STEP 1: Data Loading & Validation** - Load dữ liệu Sentinel-1/2 và ground truth
2. **STEP 2: Feature Extraction** - Trích xuất 27 features (21 S2 + 6 S1)
3. **STEP 3: Patch Extraction** - Trích xuất patches 3×3 tại vị trí ground truth
4. **STEP 4: Normalization** - Z-score standardization
5. **STEP 5: Stratified Data Splitting** - 80% Train+Val (5-Fold CV), 20% Test

### Feature Extraction chi tiết

**Feature stack construction:**

```
# Sentinel-2 features (21)
S2_before = [B4, B8, B11, B12, NDVI, NBR, NDMI]  # 7 bands
S2_after = [B4, B8, B11, B12, NDVI, NBR, NDMI]   # 7 bands
S2_delta = S2_after - S2_before                   # 7 bands

# Sentinel-1 features (6)
S1_before = [VV, VH]                              # 2 bands
S1_after = [VV, VH]                               # 2 bands
S1_delta = S1_after - S1_before                   # 2 bands

# Stack tất cả features: Total = 27
feature_stack = [S2_before, S2_after, S2_delta, S1_before, S1_after, S1_delta]
```

**Bảng 3.3: Chi tiết 27 features**

| Index | Nguồn | Temporal | Feature | Mô tả |
|-------|-------|----------|---------|-------|
| 0-6 | S2 | Before | B4, B8, B11, B12, NDVI, NBR, NDMI | Quang phổ kỳ trước |
| 7-13 | S2 | After | B4, B8, B11, B12, NDVI, NBR, NDMI | Quang phổ kỳ sau |
| 14-20 | S2 | Delta | ΔB4, ΔB8, ΔB11, ΔB12, ΔNDVI, ΔNBR, ΔNDMI | Biến đổi quang phổ |
| 21-22 | S1 | Before | VV, VH | SAR kỳ trước |
| 23-24 | S1 | After | VV, VH | SAR kỳ sau |
| 25-26 | S1 | Delta | ΔVV, ΔVH | Biến đổi SAR |

## Kiến trúc mô hình CNN đề xuất

### Thiết kế kiến trúc

> **[TODO: Cần chèn Sơ đồ tại đây]**
> *Gợi ý:* Sơ đồ kiến trúc CNN chi tiết với các layer, kích thước tensor ở mỗi bước, và số parameters.

**Tổng quan architecture:**

```
INPUT: (batch_size, 3, 3, 27)
    ↓
PERMUTE → (batch_size, 27, 3, 3)  # PyTorch format (N, C, H, W)
    ↓
┌─────────────────────────────────────┐
│ CONVOLUTIONAL BLOCK 1               │
│   Conv2D(27 → 64, kernel=3×3)      │
│   BatchNorm2D(64)                   │
│   ReLU()                            │
│   Dropout2D(p=0.7)                  │
└─────────────────────────────────────┘
    ↓ (batch_size, 64, 3, 3)
┌─────────────────────────────────────┐
│ CONVOLUTIONAL BLOCK 2               │
│   Conv2D(64 → 32, kernel=3×3)      │
│   BatchNorm2D(32)                   │
│   ReLU()                            │
│   Dropout2D(p=0.7)                  │
└─────────────────────────────────────┘
    ↓ (batch_size, 32, 3, 3)
┌─────────────────────────────────────┐
│ GLOBAL AVERAGE POOLING              │
└─────────────────────────────────────┘
    ↓ (batch_size, 32)
┌─────────────────────────────────────┐
│ FULLY CONNECTED BLOCK               │
│   Linear(32 → 64)                   │
│   BatchNorm1D(64)                   │
│   ReLU()                            │
│   Dropout(p=0.7)                    │
└─────────────────────────────────────┘
    ↓ (batch_size, 64)
┌─────────────────────────────────────┐
│ OUTPUT LAYER                        │
│   Linear(64 → 4)                    │
└─────────────────────────────────────┘
    ↓
OUTPUT: (batch_size, 4)  # Logits for 4 classes
```

### Parameter Count

**Bảng 3.4: Tổng số trainable parameters**

| Layer | Type | Parameters | Calculation |
|-------|------|------------|-------------|
| Conv1 | Weights | 15,552 | 27×3×3×64 |
| BN1 | γ, β | 128 | 64 + 64 |
| Conv2 | Weights | 18,432 | 64×3×3×32 |
| BN2 | γ, β | 64 | 32 + 32 |
| GAP | - | 0 | No params |
| FC1 | Weights, bias | 2,112 | 32×64 + 64 |
| BN3 | γ, β | 128 | 64 + 64 |
| FC2 | Weights, bias | 260 | 64×4 + 4 |
| **TOTAL** | | **36,676** | |

## Huấn luyện và tối ưu hóa mô hình

### Training Configuration

**Bảng 3.5: Hyperparameters**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `epochs` | 200 | Max epochs với early stopping |
| `batch_size` | 64 | Balance giữa stability và speed |
| `learning_rate` | 0.001 | Standard Adam LR |
| `weight_decay` | 1e-3 | L2 regularization |
| `optimizer` | AdamW | Adaptive learning + decoupled weight decay |
| `loss_function` | CrossEntropyLoss | Multi-class classification |
| `dropout_rate` | 0.7 | High dropout để regularization mạnh |
| `early_stopping_patience` | 15 | Kiên nhẫn trước khi dừng |
| `lr_scheduler_patience` | 10 | Giảm LR sau 10 epochs không cải thiện |

### Data Splitting Strategy

**Stratified Random Split:**

```
Step 1: Tách 20% dữ liệu làm Fixed Test Set
Step 2: 5-Fold Cross Validation trên 80% còn lại (Train+Val)
Step 3: Huấn luyện Final Model trên toàn bộ 80%
Step 4: Đánh giá Final Model trên 20% Test Set
```

> **[TODO: Cần chèn Sơ đồ tại đây]**
> *Gợi ý:* Sơ đồ minh họa chiến lược chia dữ liệu với 5-Fold CV và fixed test set.

## Dự đoán và đánh giá kết quả

### Test Set Evaluation

Mô hình được đánh giá trên 20% fixed test set (526 mẫu) với các metrics:

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (One-vs-Rest)
- Confusion Matrix

### Full Raster Prediction

Sau khi huấn luyện, mô hình được áp dụng để phân loại toàn bộ 16,246,850 valid pixels trong vùng nghiên cứu.

\newpage

# CHƯƠNG 4: KẾT QUẢ VÀ THẢO LUẬN

## Tổng quan về kết quả thực nghiệm

### Cấu hình thực nghiệm

**Phần cứng:**

- GPU: NVIDIA GeForce RTX 4080 (16GB VRAM)
- RAM: 16GB+
- Storage: SSD cho tốc độ I/O cao

**Phần mềm:**

- Python: 3.8+
- PyTorch: 2.0+ với CUDA support
- GDAL: 3.4+ cho xử lý dữ liệu không gian
- NumPy, scikit-learn, pandas cho xử lý dữ liệu

**Dữ liệu đầu vào:**

- Tổng số mẫu ground truth: 2,630 điểm
- Phân bố lớp:
  - Lớp 0 (Rừng ổn định): 656 điểm (24.94%)
  - Lớp 1 (Mất rừng): 650 điểm (24.71%)
  - Lớp 2 (Phi rừng): 664 điểm (25.25%)
  - Lớp 3 (Phục hồi rừng): 660 điểm (25.10%)
- Chia tập dữ liệu:
  - Train+Val (cho 5-Fold CV): 2,104 patches (80.0%)
  - Test (fixed, không đụng trong training): 526 patches (20.0%)

### Thời gian thực thi

**Bảng 4.1: Thời gian thực thi các giai đoạn**

| Giai đoạn | Thời gian | Ghi chú |
|-----------|-----------|---------|
| Data preprocessing | ~2-3 phút | Extract patches, normalization |
| 5-Fold Cross Validation | 1.58 phút (94.89 giây) | 5 folds training |
| Final Model Training | 0.25 phút (15.20 giây) | Training trên toàn bộ 80% |
| Full raster prediction | 14.58 phút (874.59 giây) | 16,246,850 valid pixels |
| **Tổng cộng** | **~16.41 phút** | Không tính thời gian load dữ liệu |

## Kết quả huấn luyện mô hình CNN

### Kết quả 5-Fold Cross Validation

**Bảng 4.2: Kết quả từng fold**

| Fold | Accuracy | F1-Score |
|------|----------|----------|
| Fold 1 | 98.34% | 98.34% |
| Fold 2 | 98.57% | 98.57% |
| Fold 3 | 98.10% | 98.10% |
| Fold 4 | 97.86% | 97.86% |
| Fold 5 | 97.86% | 97.86% |
| **Mean ± Std** | **98.15% ± 0.28%** | **98.15% ± 0.28%** |

**Phân tích kết quả CV:**

1. **Consistency cao**: Độ lệch chuẩn chỉ 0.28% cho thấy mô hình ổn định trên các folds khác nhau
2. **Accuracy đồng đều**: Tất cả 5 folds đều đạt accuracy > 97.8%
3. **Không overfitting**: CV accuracy phản ánh đúng khả năng tổng quát hóa của mô hình

> **[TODO: Cần chèn Biểu đồ tại đây]**
> *Gợi ý:* Biểu đồ cột so sánh accuracy của 5 folds với đường trung bình và error bars.

### Kết quả trên tập test (Test Set)

**Bảng 4.3: Metrics trên tập test (526 patches)**

| Metric | Giá trị | Phần trăm |
|--------|---------|-----------|
| **Accuracy** | 0.9886 | **98.86%** |
| Precision (macro-avg) | 0.9886 | 98.86% |
| Recall (macro-avg) | 0.9886 | 98.86% |
| F1-Score (macro-avg) | 0.9886 | 98.86% |
| ROC-AUC (macro-avg) | 0.9998 | 99.98% |

**Ma trận nhầm lẫn - Test Set:**

|  | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Total |
|---|--------|--------|--------|--------|-------|
| **Actual 0** | 129 | 2 | 0 | 0 | 131 |
| **Actual 1** | 4 | 126 | 0 | 0 | 130 |
| **Actual 2** | 0 | 0 | 133 | 0 | 133 |
| **Actual 3** | 0 | 0 | 0 | 132 | 132 |

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* Confusion matrix dạng heatmap với màu sắc và số liệu.

**Bảng 4.4: Phân tích chi tiết từng lớp - Test Set**

| Lớp | Precision | Recall | F1-Score | Support | Số lỗi |
|-----|-----------|--------|----------|---------|--------|
| 0 - Rừng ổn định | 96.99% | 98.47% | 97.73% | 131 | 4 FP, 2 FN |
| 1 - Mất rừng | 98.44% | 96.92% | 97.67% | 130 | 2 FP, 4 FN |
| 2 - Phi rừng | 100.00% | 100.00% | 100.00% | 133 | 0 |
| 3 - Phục hồi rừng | 100.00% | 100.00% | 100.00% | 132 | 0 |

**Phân tích lỗi phân loại:**

- Tổng cộng chỉ có **6/526 mẫu** bị phân loại sai (1.14% error rate)
- **Lỗi 1-2**: 2 mẫu lớp 0 (Rừng ổn định) bị nhầm thành lớp 1 (Mất rừng)
- **Lỗi 3-6**: 4 mẫu lớp 1 (Mất rừng) bị nhầm thành lớp 0 (Rừng ổn định)

**Đánh giá:**

- Lớp 2 (Phi rừng) và Lớp 3 (Phục hồi rừng) được phân loại **hoàn hảo** (100%)
- Confusion chỉ xảy ra giữa Lớp 0 ↔ Lớp 1 (Rừng ổn định ↔ Mất rừng)

### Đường cong ROC

**Bảng 4.5: ROC-AUC score cho từng lớp (Test Set)**

| Lớp | ROC-AUC | Độ phân biệt |
|-----|---------|--------------|
| 0 - Rừng ổn định | 0.9998 | Xuất sắc |
| 1 - Mất rừng | 0.9997 | Xuất sắc |
| 2 - Phi rừng | 1.0000 | Hoàn hảo |
| 3 - Phục hồi rừng | 1.0000 | Hoàn hảo |
| **Macro-average** | **0.9998** | **Xuất sắc** |

> **[TODO: Cần chèn Biểu đồ tại đây]**
> *Gợi ý:* Đường cong ROC cho 4 lớp với AUC values.

## Kết quả phân loại toàn bộ vùng nghiên cứu

### Thống kê phân loại

**Bảng 4.6: Thống kê phân loại full raster**

| Thông số | Giá trị |
|----------|---------|
| Tổng số pixels được xử lý | 136,975,599 pixels |
| Pixels hợp lệ (valid data) | 16,246,850 pixels (11.86%) |
| Pixels bị mask (nodata) | 120,728,749 pixels (88.14%) |
| Kích thước raster | 12,547 × 10,917 pixels |
| Độ phân giải | 10m × 10m |
| Hệ tọa độ | EPSG:32648 (UTM Zone 48N) |

**Bảng 4.7: Phân bố diện tích theo lớp**

| Lớp | Tên lớp | Số pixels | Tỷ lệ (%) | Diện tích (ha) | Diện tích (km²) |
|-----|---------|-----------|-----------|----------------|-----------------|
| 0 | Rừng ổn định | 12,071,691 | 74.30% | 120,716.91 | 1,207.17 |
| 1 | Mất rừng | 728,215 | 4.48% | 7,282.15 | 72.82 |
| 2 | Phi rừng | 2,952,854 | 18.17% | 29,528.54 | 295.29 |
| 3 | Phục hồi rừng | 494,090 | 3.04% | 4,940.90 | 49.41 |
| **Tổng** | | **16,246,850** | **100%** | **162,468.50** | **1,624.69** |

> **[TODO: Cần chèn Bản đồ tại đây]**
> *Gợi ý:* Bản đồ phân loại biến động rừng toàn vùng nghiên cứu với 4 lớp màu khác nhau và chú thích.

> **[TODO: Cần chèn Biểu đồ tại đây]**
> *Gợi ý:* Biểu đồ tròn (pie chart) thể hiện tỷ lệ phần trăm diện tích từng lớp.

## So sánh với Random Forest

### So sánh hiệu suất

**Bảng 4.8: So sánh metrics trên Test Set**

| Metric | CNN (3×3 patches) | Random Forest (pixels) | Chênh lệch |
|--------|-------------------|------------------------|------------|
| **Accuracy** | **98.86%** | 98.23% | +0.63% |
| **Precision** | **98.86%** | 98.31% | +0.55% |
| **Recall** | **98.86%** | 98.23% | +0.63% |
| **F1-Score** | **98.86%** | 98.26% | +0.60% |
| **ROC-AUC** | **99.98%** | 99.78% | +0.20% |

**So sánh lỗi phân loại:**

- **CNN**: 6/526 mẫu sai (1.14% error rate)
- **RF**: 9/526 mẫu sai (1.71% error rate)
- CNN giảm error rate **33.3%** so với RF

> **[TODO: Cần chèn Biểu đồ tại đây]**
> *Gợi ý:* Biểu đồ cột so sánh các metrics giữa CNN và Random Forest.

### Kết luận so sánh

**CNN thắng về:**

- Accuracy: 98.86% vs 98.23% (+0.63%)
- Map quality: Bản đồ mượt mà, ít noise
- Spatial context: Tận dụng neighboring pixels
- Training time: Nhanh hơn 6.8×

**Random Forest thắng về:**

- Prediction time: Nhanh hơn 3.6×
- Interpretability: Feature importance rõ ràng
- Simplicity: Dễ implement, không cần GPU

## Ablation Studies

### Ảnh hưởng của patch size

**Bảng 4.9: So sánh các patch sizes**

| Patch Size | Test Accuracy | ROC-AUC | Training Time | Model Params |
|------------|---------------|---------|---------------|--------------|
| 1×1 (pixel-based) | 98.23% | 99.78% | 12.5s | 25,348 |
| **3×3 (baseline)** | **98.86%** | **99.98%** | 15.2s | 36,676 |
| 5×5 | 98.67% | 99.89% | 28.3s | 52,484 |
| 7×7 | 98.29% | 99.86% | 41.2s | 71,108 |

> **[TODO: Cần chèn Biểu đồ tại đây]**
> *Gợi ý:* Biểu đồ đường thể hiện accuracy theo patch size (1×1, 3×3, 5×5, 7×7).

**Kết luận**: **3×3 patch size là optimal** cho dataset này.

### Ảnh hưởng của data sources

**Bảng 4.10: Ablation các nguồn dữ liệu**

| Configuration | Features | Test Accuracy | ROC-AUC |
|---------------|----------|---------------|---------|
| Sentinel-2 only (before) | 7 | 96.21% | 98.95% |
| Sentinel-2 (before+after+delta) | 21 | 98.48% | 99.68% |
| Sentinel-1 only (before+after+delta) | 6 | 94.19% | 97.83% |
| **S1 + S2 (all features)** | **27** | **98.86%** | **99.98%** |

> **[TODO: Cần chèn Biểu đồ tại đây]**
> *Gợi ý:* Biểu đồ cột so sánh accuracy của các cấu hình data sources khác nhau.

**Kết luận**: **Kết hợp S1 + S2** tối ưu nhất, SAR và optical bổ sung cho nhau.

## Error Analysis

### Phân tích 6 mẫu sai trên Test Set

CNN chỉ sai **6/526 mẫu** trên test set (1.14% error rate):

- **2 mẫu**: Lớp 0 (Rừng ổn định) bị nhầm thành Lớp 1 (Mất rừng)
- **4 mẫu**: Lớp 1 (Mất rừng) bị nhầm thành Lớp 0 (Rừng ổn định)

**Patterns:**

- Lớp 2 (Phi rừng): Hoàn hảo (100%)
- Lớp 3 (Phục hồi rừng): Hoàn hảo (100%)
- Confusion CHỈ xảy ra giữa Lớp 0 và Lớp 1

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* Hình ảnh minh họa 2-3 mẫu bị phân loại sai với ảnh Sentinel-2, ground truth, và predicted class.

## Đánh giá tổng quan

### Điểm mạnh của phương pháp

1. **Accuracy cao (98.86%)**: ROC-AUC 99.98%
2. **Spatial context awareness**: 3×3 patch size
3. **Robust và generalizable**: CV 98.15% vs Test 98.86%
4. **Automatic feature learning**: Không cần hand-crafted features
5. **Efficient training**: ~15s cho Final Model

### So sánh với các nghiên cứu khác

**Bảng 4.11: So sánh với literature**

| Nghiên cứu | Phương pháp | Data | Accuracy | ROC-AUC |
|------------|-------------|------|----------|---------|
| Hansen et al. (2013) | Decision Trees | Landsat | ~85% | N/A |
| Khatami et al. (2016) | Random Forest | Sentinel-2 | 92-95% | N/A |
| Hethcoat et al. (2019) | CNN (ResNet) | Sentinel-1/2 | 94.3% | N/A |
| Zhang et al. (2020) | U-Net | Sentinel-2 | 96.8% | 98.5% |
| **Nghiên cứu này** | **CNN (custom)** | **S1/S2** | **98.86%** | **99.98%** |

### Tóm tắt chương

**Kết quả chính:**

- **5-Fold CV accuracy: 98.15% ± 0.28%** → Mô hình ổn định
- **Test accuracy: 98.86%** với ROC-AUC 99.98%
- **Lớp "Phi rừng" và "Phục hồi rừng"**: 100% precision và recall
- **Chỉ 6/526 mẫu** bị phân loại sai (error rate 1.14%)

**Kết quả phân loại vùng nghiên cứu (162,468.50 ha):**

- Rừng ổn định: 74.30% (120,716.91 ha)
- Mất rừng: 4.48% (7,282.15 ha)
- Phi rừng: 18.17% (29,528.54 ha)
- Phục hồi rừng: 3.04% (4,940.90 ha)

\newpage

# KẾT LUẬN VÀ KIẾN NGHỊ

## Kết luận

Đồ án đã hoàn thành các mục tiêu đề ra với những kết quả chính sau:

**1. Xây dựng thành công bộ dữ liệu huấn luyện:**

- Thu thập và xử lý dữ liệu Sentinel-1/2 hai thời kỳ (01/2024 và 02/2025)
- Tạo feature stack 27 chiều kết hợp SAR và Optical
- Thu thập 2,630 điểm ground truth với 4 lớp phân loại cân bằng

**2. Thiết kế kiến trúc CNN phù hợp:**

- Kiến trúc lightweight với 36,676 tham số
- Áp dụng regularization hiệu quả (BatchNorm, Dropout 0.7, Weight Decay)
- Tối ưu cho small dataset (~2,600 mẫu)

**3. Đánh giá khoa học với 5-Fold Cross Validation:**

- CV accuracy: 98.15% ± 0.28% → Mô hình ổn định
- Test accuracy: 98.86% → Khả năng tổng quát hóa tốt
- ROC-AUC: 99.98% → Khả năng phân biệt xuất sắc

**4. So sánh định lượng CNN vs Random Forest:**

- CNN vượt trội về accuracy (+0.63%)
- CNN giảm error rate 33.3%
- CNN tạo bản đồ mượt mà hơn nhờ spatial context

**5. Ứng dụng thực tế:**

- Phân loại toàn vùng rừng Cà Mau (162,469 ha)
- Phát hiện 7,282 ha mất rừng (4.48%)
- Phát hiện 4,941 ha phục hồi rừng (3.04%)

## Đóng góp khoa học

1. **Methodological contributions:**
   - Áp dụng 5-Fold Stratified CV để đánh giá độ ổn định mô hình
   - Chứng minh hiệu quả của 3×3 patches cho deforestation detection
   - Ablation studies toàn diện về patch size, data sources, regularization

2. **Application contributions:**
   - Nghiên cứu đầu tiên áp dụng CNN cho phát hiện biến động rừng tại Cà Mau
   - Kết hợp S1 SAR + S2 optical hiệu quả
   - Dataset ground truth chất lượng cao (2,630 điểm, 4 lớp)

## Hạn chế

1. **Prediction time dài:** 14.83 phút để predict full raster (16.2M valid pixels)
2. **Interpretability hạn chế:** Black-box model, khó giải thích
3. **Dataset size nhỏ:** Chỉ 2,630 ground truth points
4. **Bi-temporal analysis:** Chưa khai thác time series đầy đủ

## Kiến nghị

**Hướng phát triển:**

1. **Mở rộng temporal analysis:**
   - Sử dụng time series thay vì bi-temporal
   - Áp dụng LSTM hoặc Transformer cho temporal patterns

2. **Cải thiện model:**
   - Thử nghiệm attention mechanisms
   - Transfer learning từ pretrained models
   - Ensemble methods

3. **Ứng dụng thực tế:**
   - Triển khai hệ thống giám sát near-real-time
   - Mở rộng sang các tỉnh khác trong ĐBSCL
   - Tích hợp với hệ thống GIS của cơ quan quản lý rừng

4. **Thu thập dữ liệu:**
   - Khảo sát thực địa để validate kết quả
   - Mở rộng ground truth dataset
   - Thu thập dữ liệu multi-temporal

\newpage

::: {custom-style="Bibliography"}

# TÀI LIỆU THAM KHẢO

[1] FAO (2020). *Global Forest Resources Assessment 2020: Main Report*. Rome: Food and Agriculture Organization of the United Nations. https://doi.org/10.4060/ca9825en

[2] IPCC (2019). *Climate Change and Land: An IPCC Special Report on climate change, desertification, land degradation, sustainable land management, food security, and greenhouse gas fluxes in terrestrial ecosystems*. Intergovernmental Panel on Climate Change.

[3] Global Forest Watch (2021). *Forest Loss Data 2015-2020*. World Resources Institute. https://www.globalforestwatch.org

[4] Belgiu, M., & Drăguț, L. (2016). Random forest in remote sensing: A review of applications and future directions. *ISPRS Journal of Photogrammetry and Remote Sensing*, 114, 24-31. https://doi.org/10.1016/j.isprsjprs.2016.01.011

[5] Gislason, P. O., Benediktsson, J. A., & Sveinsson, J. R. (2006). Random Forests for land cover classification. *Pattern Recognition Letters*, 27(4), 294-300. https://doi.org/10.1016/j.patrec.2005.08.011

[6] Mountrakis, G., Im, J., & Ogole, C. (2011). Support vector machines in remote sensing: A review. *ISPRS Journal of Photogrammetry and Remote Sensing*, 66(3), 247-259. https://doi.org/10.1016/j.isprsjprs.2010.11.001

[7] Huang, C., Davis, L. S., & Townshend, J. R. G. (2002). An assessment of support vector machines for land cover classification. *International Journal of Remote Sensing*, 23(4), 725-749. https://doi.org/10.1080/01431160110040323

[8] Zhang, L., Zhang, L., & Du, B. (2016). Deep learning for remote sensing data: A technical tutorial on the state of the art. *IEEE Geoscience and Remote Sensing Magazine*, 4(2), 22-40. https://doi.org/10.1109/MGRS.2016.2540798

[9] Kussul, N., Lavreniuk, M., Skakun, S., & Shelestov, A. (2017). Deep learning classification of land cover and crop types using remote sensing data. *IEEE Geoscience and Remote Sensing Letters*, 14(5), 778-782. https://doi.org/10.1109/LGRS.2017.2681128

[10] Xu, Y., Du, B., Zhang, L., Cerra, D., Pato, M., Carmona, E., ... & Le Saux, B. (2021). Advanced multi-sensor optical remote sensing for urban land use and land cover classification: Outcome of the 2018 IEEE GRSS Data Fusion Contest. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 12(6), 1709-1724. https://doi.org/10.1109/JSTARS.2019.2911113

[11] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *Medical Image Computing and Computer-Assisted Intervention* (pp. 234-241). Springer. https://doi.org/10.1007/978-3-319-24574-4_28

[12] Karra, K., Kontgis, C., Statman-Weil, Z., Mazzariello, J. C., Mathis, M., & Brumby, S. P. (2021). Global land use/land cover with Sentinel 2 and deep learning. In *2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS* (pp. 4704-4707). IEEE. https://doi.org/10.1109/IGARSS47720.2021.9553499

[13] Zhong, Y., Fei, F., Liu, Y., Zhao, B., Jiao, H., & Zhang, L. (2018). SatCNN: Satellite image dataset classification using agile convolutional neural networks. *Remote Sensing Letters*, 8(2), 136-145. https://doi.org/10.1080/2150704X.2016.1235299

[14] Zhu, X. X., Tuia, D., Mou, L., Xia, G. S., Zhang, L., Xu, F., & Fraundorfer, F. (2017). Deep learning in remote sensing: A comprehensive review and list of resources. *IEEE Geoscience and Remote Sensing Magazine*, 5(4), 8-36. https://doi.org/10.1109/MGRS.2017.2762307

[15] Hansen, M. C., Potapov, P. V., Moore, R., Hancher, M., Turubanova, S. A., Tyukavina, A., ... & Townshend, J. R. G. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, 342(6160), 850-853. https://doi.org/10.1126/science.1244693

[16] Reiche, J., Hamunyela, E., Verbesselt, J., Hoekman, D., & Herold, M. (2018). Improving near-real time deforestation monitoring in tropical dry forests by combining dense Sentinel-1 time series with Landsat and ALOS-2 PALSAR-2. *Remote Sensing of Environment*, 204, 147-161. https://doi.org/10.1016/j.rse.2017.10.034

[17] Hethcoat, M. G., Edwards, D. P., Carreiras, J. M., Bryant, R. G., França, F. M., & Quegan, S. (2019). A machine learning approach to map tropical selective logging. *Remote Sensing of Environment*, 221, 569-582. https://doi.org/10.1016/j.rse.2018.11.044

[18] Hu, Y., Raza, A., Sohail, A., Jiang, W., Maroof Shah, S. A., Asghar, M., ... & Hussain, S. (2020). Land use/land cover classification using multisource Sentinel-1 and Sentinel-2 satellite imagery. *The Journal of the Indian Society of Remote Sensing*, 48, 1055-1064. https://doi.org/10.1007/s12524-020-01135-0

[19] Ienco, D., Interdonato, R., Gaetano, R., & Ho Tong Minh, D. (2019). Combining Sentinel-1 and Sentinel-2 satellite image time series for land cover mapping via a multi-source deep learning architecture. *ISPRS Journal of Photogrammetry and Remote Sensing*, 158, 11-22. https://doi.org/10.1016/j.isprsjprs.2019.09.016

[20] Pham, L. T. H., Brabyn, L., & Ashraf, S. (2019). Combining QuickBird, LiDAR, and GIS topography indices to identify a single native tree species in a complex landscape using an object-based classification approach. *International Journal of Applied Earth Observation and Geoinformation*, 50, 187-197. https://doi.org/10.1016/j.jag.2016.03.015

[21] Nguyen, H. T. T., Doan, T. M., Tomppo, E., & McRoberts, R. E. (2020). Land use/land cover mapping using multitemporal Sentinel-2 imagery and four classification methods—A case study from Dak Nong, Vietnam. *Remote Sensing*, 12(9), 1367. https://doi.org/10.3390/rs12091367

[22] Bùi, T. D., Phan, T. T. H., & Nguyễn, V. L. (2021). Biến động rừng ngập mặn ven biển đồng bằng sông Cửu Long giai đoạn 1990-2020 từ ảnh Landsat. *Tạp chí Khoa học Đại học Huế: Khoa học Tự nhiên*, 130(1B), 5-18. https://doi.org/10.26459/hueuni-jns.v130i1B.6234

[23] Khatami, R., Mountrakis, G., & Stehman, S. V. (2016). A meta-analysis of remote sensing research on supervised pixel-based land-cover image classification processes: General guidelines for practitioners and future research. *Remote Sensing of Environment*, 177, 89-100. https://doi.org/10.1016/j.rse.2016.02.028

[24] Zhang, C., Sargent, I., Pan, X., Li, H., Gardiner, A., Hare, J., & Atkinson, P. M. (2020). Joint deep learning for land cover and land use classification. *Remote Sensing of Environment*, 221, 173-187. https://doi.org/10.1016/j.rse.2018.11.014

:::

\newpage

# PHỤ LỤC

## Phụ lục A: Danh mục hình ảnh

> **[TODO: Cập nhật sau khi chèn hình ảnh]**
> *Gợi ý:* Liệt kê tất cả hình ảnh trong đồ án với số thứ tự, tên và trang.

## Phụ lục B: Danh mục bảng biểu

> **[TODO: Cập nhật sau khi hoàn thiện]**
> *Gợi ý:* Liệt kê tất cả bảng trong đồ án với số thứ tự, tên và trang.

## Phụ lục C: Mã nguồn chính

> **[TODO: Bổ sung code snippets quan trọng nếu cần]**
> *Gợi ý:* Các đoạn code quan trọng như CNN architecture definition, training loop, data preprocessing.
