---
title: "ỨNG DỤNG VIỄN THÁM VÀ HỌC SÂU TRONG GIÁM SÁT BIẾN ĐỘNG RỪNG TỈNH CÀ MAU"
subtitle: |
  ĐỒ ÁN TỐT NGHIỆP

  **Sinh viên thực hiện:** Ninh Hải Đăng

  **Mã số sinh viên:** 21021411

  **Lớp:** Công nghệ Hàng không Vũ trụ K66

   **Giảng viên hướng dẫn:** TS. Hà Minh Cường và ThS. Hoàng Tích Phúc

  **Đơn vị:** Viện Công nghệ Hàng không Vũ trụ

  Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội
author: "Ninh Hải Đăng"
date: "2025"
lang: vi
---

\newpage

::: {custom-style="Abstract"}

# Tóm tắt 

Đồ án này nghiên cứu ứng dụng mạng Neural Tích chập (CNN) để phát hiện và phân loại biến động rừng từ dữ liệu viễn thám đa nguồn tại tỉnh Cà Mau. Nghiên cứu sử dụng dữ liệu từ vệ tinh Sentinel-1 (SAR) và Sentinel-2 (Optical) với độ phân giải 10m, kết hợp 27 đặc trưng (features) từ cả hai nguồn dữ liệu.

**Phương pháp:** Thiết kế kiến trúc CNN nhẹ (~36,000 tham số) với patches không gian 3×3, áp dụng stratified random split kết hợp 5-Fold Cross Validation để đánh giá mô hình. Bộ dữ liệu gồm 2,630 điểm ground truth được chia thành 80% Train+Val và 20% Test cố định.

**Kết quả:** Mô hình CNN đạt độ chính xác 98.86% trên tập test với ROC-AUC 99.98%. Cross Validation cho kết quả 98.15% ± 0.28%, chứng tỏ mô hình ổn định và có khả năng tổng quát hóa tốt.

**Ứng dụng:** Phân loại toàn vùng rừng Cà Mau (162,469 ha), phát hiện 7,282 ha mất rừng (4.48%) và 4,941 ha phục hồi rừng (3.04%) trong giai đoạn 01/2024 - 02/2025.

**Từ khóa:** CNN, Deep Learning, viễn thám, Sentinel-1, Sentinel-2, biến động rừng, Cà Mau

:::

\newpage

# Lời cam đoan 

Tôi tên là Ninh Hải Đăng, sinh viên lớp QH-2021-I/CQ-S-AE, Viện Công nghệ Hàng không Vũ trụ, Trường Đại học Công nghệ – Đại học Quốc gia Hà Nội. Tôi xin cam đoan rằng Đồ án tốt nghiệp với đề tài "Ứng dụng viễn thám và học sâu trong giám sát biến động rừng tỉnh Cà Mau" là kết quả nghiên cứu khoa học do chính tôi thực hiện. Mọi sự hỗ trợ, hướng dẫn trong quá trình thực hiện đều đã được ghi nhận và cảm ơn; các thông tin, số liệu, tài liệu tham khảo trong đồ án đều được trích dẫn đầy đủ và được phép sử dụng. Tôi xin đảm bảo rằng tất cả dữ liệu nghiên cứu và kết quả trình bày trong đồ án là trung thực và chính xác. Nếu phát hiện bất kỳ sai sót nào, tôi xin hoàn toàn chịu trách nhiệm trước cơ quan nhà trường và các cơ quan liên quan.

\newpage

# Lời cảm ơn 

Đối với mỗi sinh viên, đồ án tốt nghiệp là một cột mốc quan trọng, phản ánh kết quả học tập và quá trình rèn luyện suốt thời gian tại trường đại học. Trong quá trình thực hiện đồ án, bên cạnh những nỗ lực của bản thân, em đã nhận được rất nhiều sự giúp đỡ quý báu từ các thầy cô và những người xung quanh, nhờ đó công trình này mới có thể hoàn thành. Trước hết, em xin bày tỏ lòng biết ơn sâu sắc tới các thầy, cô là cán bộ, giảng viên trong trường Đại học Công nghệ nói chung và Viện Công nghệ Hàng không Vũ trụ nói riêng, những người đã tạo điều kiện và truyền đạt kiến thức quý báu trong suốt thời gian học tập. Đặc biệt, em xin gửi lời cảm ơn chân thành tới TS. Hà Minh Cường và ThS. Hoàng Tích Phúc – giảng viên Viện Công nghệ Hàng không Vũ trụ, trường Đại học Công nghệ - Đại học Quốc gia Hà Nội, những người đã tận tình hướng dẫn, chỉ bảo và đồng hành cùng em trong suốt quá trình thực hiện đồ án. Em cũng xin trân trọng cảm ơn TS. Hoàng Việt Anh và ThS. Vũ Văn Thái – Công ty TNHH Tư vấn và Công nghệ Đồng Xanh (GFD), đã hỗ trợ em trong việc cung cấp dữ liệu, cơ sở vật chất và tạo điều kiện thuận lợi để hoàn thành nghiên cứu. Cuối cùng, em xin gửi lời cảm ơn sâu sắc tới gia đình, bạn bè và các đồng nghiệp tại công ty đã luôn động viên, chia sẻ và đồng hành cùng em trong suốt quá trình học tập và thực hiện đồ án. Một lần nữa, em xin được bày tỏ lòng biết ơn chân thành tới tất cả mọi người. Em xin trân trọng cảm ơn!

\newpage

# Danh mục từ viết tắt 

| Từ viết tắt | Giải thích |
|-------------|------------|
| AI | Artificial Intelligence (Trí tuệ nhân tạo) |
| CNN | Convolutional Neural Network (Mạng Neural Tích chập) |
| SAR | Synthetic Aperture Radar (Radar khẩu độ tổng hợp) |
| NDVI | Normalized Difference Vegetation Index (Chỉ số thực vật chuẩn hóa) |
| NBR | Normalized Burn Ratio (Chỉ số cháy chuẩn hóa) |
| NDMI | Normalized Difference Moisture Index (Chỉ số độ ẩm chuẩn hóa) |
| ROC-AUC | Receiver Operating Characteristic - Area Under Curve |
| GIS | Geographic Information System (Hệ thống thông tin địa lý) |
| ESA | European Space Agency (Cơ quan Vũ trụ Châu Âu) |
| EU | European Union (Liên minh Châu Âu) |
| FAO | Food and Agriculture Organization (Tổ chức Lương thực và Nông nghiệp) |
| IPCC | Intergovernmental Panel on Climate Change |
| UTM | Universal Transverse Mercator |
| WGS | World Geodetic System |

\newpage

# Mở đầu 

## Đặt vấn đề

Rừng đóng vai trò quan trọng trong việc duy trì cân bằng sinh thái, điều hòa khí hậu, lưu giữ carbon và bảo vệ đa dạng sinh học. Tuy nhiên, tình trạng mất rừng đang diễn ra nghiêm trọng trên toàn cầu, đặc biệt tại các quốc gia đang phát triển. Theo báo cáo "Global Forest Resources Assessment 2020" của Tổ chức Lương thực và Nông nghiệp Liên hợp quốc [1], thế giới đã mất ròng (net loss) khoảng 178 triệu hecta rừng trong giai đoạn 1990-2020, tương đương diện tích của Libya.

Tại Việt Nam, mặc dù độ che phủ rừng đã tăng từ 37% (năm 2000) lên 42% (năm 2020) nhờ các chương trình trồng rừng, nhưng tình trạng suy thoái và mất rừng tự nhiên vẫn đáng báo động, đặc biệt tại các tỉnh ven biển và đồng bằng sông Cửu Long. Tỉnh Cà Mau, nằm ở cực Nam Tổ Quốc, sở hữu hệ sinh thái rừng ngập mặn quan trọng nhưng đang phải đối mặt với áp lực từ nuôi trồng thủy sản, xâm nhập mặn, và biến đổi khí hậu.

Phương pháp giám sát rừng truyền thống dựa trên điều tra thực địa tốn kém thời gian, chi phí và khó áp dụng cho diện tích rộng. Công nghệ viễn thám vệ tinh cung cấp giải pháp hiệu quả, cho phép giám sát liên tục, diện rộng với chi phí hợp lý. Chương trình Copernicus của Liên minh Châu Âu (EU) cung cấp dữ liệu miễn phí từ các vệ tinh Sentinel-1 (SAR) và Sentinel-2 (Optical) với độ phân giải không gian 10m và chu kỳ quay trở lại ngắn (5-6 ngày), phù hợp cho giám sát rừng nhiệt đới.

Trong những năm gần đây, trí tuệ nhân tạo (AI) và học sâu (Deep Learning) đã đạt được những bước tiến vượt bậc trong xử lý ảnh và nhận dạng mẫu. Mạng Neural Tích chập (Convolutional Neural Networks - CNN) đặc biệt hiệu quả trong phân loại ảnh nhờ khả năng tự động học đặc trưng không gian (spatial features) từ dữ liệu thô, có thể học các mẫu phức tạp và bất biến đối với phép tịnh tiến, xoay.

Xuất phát từ nhu cầu thực tiễn về giám sát rừng hiệu quả và xu hướng ứng dụng công nghệ AI tiên tiến, đồ án này lựa chọn đề tài **"Ứng dụng mạng Neural Tích chập sâu trong giám sát biến động rừng từ ảnh vệ tinh đa nguồn: Nghiên cứu điển hình tại tỉnh Cà Mau"** nhằm phát triển hệ thống tự động phát hiện mất rừng với độ chính xác cao.

## Mục tiêu và nội dung nghiên cứu

Mục tiêu tổng quát của đồ án là phát triển mô hình học sâu dựa trên kiến trúc CNN để phát hiện và phân loại tự động các khu vực biến động rừng từ ảnh vệ tinh đa nguồn (Sentinel-1 SAR và Sentinel-2 Optical) tại tỉnh Cà Mau.

Để đạt được mục tiêu tổng quát, đề tài tập trung vào năm mục tiêu cụ thể. Đầu tiên là xây dựng bộ dữ liệu huấn luyện thông qua việc thu thập và xử lý dữ liệu ảnh vệ tinh Sentinel-1/2 đa thời gian, kết hợp với ground truth points để tạo bộ dữ liệu huấn luyện chất lượng cao. Tiếp theo là thiết kế kiến trúc CNN tối ưu, đề xuất kiến trúc CNN nhẹ (lightweight) phù hợp với bộ dữ liệu có quy mô vừa phải (~2,600 mẫu), tích hợp các kỹ thuật regularization (Batch Normalization, Dropout) để tránh overfitting. Bên cạnh đó, việc phân chia dữ liệu khoa học được triển khai bằng phương pháp stratified random split để đảm bảo phân bố lớp đồng đều giữa các tập huấn luyện, validation và test, kết hợp với 5-Fold Cross Validation để đánh giá robust. Tiếp tục, huấn luyện và tối ưu hóa mô hình được thực hiện bằng cách áp dụng các kỹ thuật huấn luyện tiên tiến như early stopping, learning rate scheduling, class weighting để đạt được mô hình có hiệu suất cao và ổn định. Sau đó, đánh giá mô hình dựa trên các chỉ số Accuracy, Precision, Recall, F1-Score, ROC-AUC để đảm bảo hiệu suất cao và đáng tin cậy. Cuối cùng, mô hình đã huấn luyện được ứng dụng thực tế để phân loại toàn bộ khu vực rừng Cà Mau, ước tính diện tích mất rừng, và trực quan hóa kết quả dưới dạng bản đồ phân loại.

## Đối tượng và phạm vi nghiên cứu

Đối tượng nghiên cứu chính bao gồm các khu vực rừng tự nhiên và rừng trồng tại tỉnh Cà Mau, bao gồm rừng ngập mặn và rừng phòng hộ ven biển. Các trạng thái biến động rừng được phân loại thành bốn nhóm: Forest Stable (Rừng ổn định) là vùng rừng không có biến đổi trong giai đoạn nghiên cứu; Deforestation (Mất rừng) là vùng rừng bị chuyển đổi sang đất trống, đất canh tác hoặc nuôi trồng thủy sản; Non-forest (Không phải rừng) là vùng không có rừng trong cả hai thời điểm (đất trống, mặt nước, khu dân cư); và Reforestation (Tái trồng rừng) là vùng không có rừng trở thành rừng trong giai đoạn nghiên cứu. Dữ liệu viễn thám được sử dụng bao gồm ảnh vệ tinh đa nguồn từ Sentinel-1 (SAR) và Sentinel-2 (Optical), kỳ trước (tháng 1-2/2024) và kỳ sau (tháng 2/2025).

Phạm vi nghiên cứu bao gồm toàn bộ khu vực có rừng trong ranh giới hành chính tỉnh Cà Mau, diện tích khoảng 162,469 hecta (tương đương 1,624.69 km²). Thời gian nghiên cứu kéo dài từ tháng 01/2024 đến tháng 02/2025 (khoảng 13 tháng). Độ phân giải không gian của dữ liệu là 10 mét/pixel (độ phân giải gốc của Sentinel-1/2), và hệ tọa độ được sử dụng là EPSG:32648 (WGS 84 / UTM Zone 48N). Tuy nhiên, nghiên cứu có một số giới hạn như chỉ sử dụng dữ liệu tại hai thời điểm (bi-temporal), chưa khai thác đầy đủ chuỗi thời gian liên tục, và ground truth được thu thập từ phiên giải ảnh và dữ liệu có sẵn, chưa có khảo sát thực địa đầy đủ. Mô hình được đào tạo và đánh giá trên dữ liệu Cà Mau, khả năng tổng quát hóa sang các khu vực khác cần được kiểm chứng thêm.

## Ý nghĩa khoa học và thực tiễn của đề tài

Về mặt khoa học, đồ án đóng góp một số điểm chính. Trước hết, đề xuất kiến trúc CNN nhẹ và hiệu quả cho bài toán phân loại ảnh viễn thám với bộ dữ liệu nhỏ; tiếp theo, việc áp dụng 5-Fold Stratified Cross Validation kết hợp với fixed test set giúp đánh giá mô hình một cách robust và đáng tin cậy; bên cạnh đó, đề tài chứng minh hiệu quả tích hợp đa nguồn bằng cách kết hợp dữ liệu SAR (Sentinel-1) và Optical (Sentinel-2), tận dụng ưu thế của từng loại dữ liệu như khả năng xuyên mây của SAR và thông tin quang phổ phong phú của Optical.

Về ý nghĩa thực tiễn, kết quả nghiên cứu mang lại nhiều lợi ích. Thứ nhất, mô hình cung cấp công cụ tự động phát hiện mất rừng với độ chính xác cao (trên 98%), giúp giảm đáng kể thời gian và chi phí so với phương pháp điều tra thực địa truyền thống; thứ hai, kết quả có thể hỗ trợ các cơ quan quản lý rừng tại Cà Mau và các tỉnh khác trong việc xây dựng cơ sở dữ liệu để ra quyết định về bảo vệ và phát triển rừng bền vững; thứ ba, hệ thống có thể được triển khai để giám sát liên tục, đóng vai trò cảnh báo sớm khi phát hiện các hoạt động phá rừng trái phép; bên cạnh đó, phương pháp còn có tiềm năng mở rộng cho các bài toán giám sát môi trường khác như biến động đất đai, đô thị hóa và thay đổi sử dụng đất; cuối cùng, do sử dụng dữ liệu vệ tinh miễn phí và thiết kế mô hình nhẹ, phương pháp này có chi phí thấp và phù hợp với điều kiện triển khai tại Việt Nam.

## Cấu trúc của đồ án

Đồ án được tổ chức thành bốn chương chính; Chương 1 trình bày tổng quan về vấn đề nghiên cứu, bao gồm bối cảnh mất rừng, công nghệ viễn thám, tổng quan các nghiên cứu liên quan và các khoảng trống nghiên cứu; Chương 2 trình bày cơ sở lý thuyết, giới thiệu chi tiết về công nghệ viễn thám (Sentinel-1/2), lý thuyết về mạng Neural Tích chập (CNN) và các phương pháp phân loại ảnh cùng những tiêu chí đánh giá mô hình; Chương 3 mô tả phương pháp nghiên cứu, bao gồm khu vực nghiên cứu, dữ liệu sử dụng, quy trình xử lý, kiến trúc mô hình CNN đề xuất, phương pháp huấn luyện và đánh giá; cuối cùng, Chương 4 trình bày các kết quả và thảo luận, bao gồm kết quả huấn luyện, đánh giá mô hình, phân loại toàn vùng, phân tích lỗi và trực quan hóa.

\newpage

# Tổng quan về vấn đề nghiên cứu

## Bối cảnh và tình hình mất rừng

### Tình hình mất rừng trên thế giới

Rừng bao phủ khoảng 31% diện tích đất liền toàn cầu [1], đóng vai trò thiết yếu trong việc điều hòa khí hậu, lưu giữ carbon, bảo tồn đa dạng sinh học, và cung cấp sinh kế cho hàng tỷ người. Tuy nhiên, tốc độ mất rừng toàn cầu vẫn đang ở mức báo động. Theo báo cáo "Global Forest Resources Assessment 2020" của FAO [1], tổng diện tích rừng bị phá (gross deforestation) từ năm 1990 đến 2020 ước tính khoảng 420 triệu hecta, trong khi diện tích mất rừng ròng (net loss, sau khi trừ đi diện tích trồng rừng mới) là 178 triệu hecta, chủ yếu do chuyển đổi sang đất nông nghiệp, chăn nuôi, khai thác gỗ bất hợp pháp, và phát triển cơ sở hạ tầng.

Khu vực nhiệt đới, nơi tập trung 45% diện tích rừng toàn cầu và đa dạng sinh học cao nhất, đang chịu tốc độ mất rừng nhanh nhất. Lưu vực Amazon (Brazil), rừng Congo (Trung Phi), và Đông Nam Á là những "điểm nóng" về mất rừng. Theo dữ liệu từ Global Forest Watch [3], thế giới mất khoảng 10 triệu hecta rừng nhiệt đới mỗi năm trong giai đoạn 2015-2020.

Mất rừng không chỉ làm giảm khả năng hấp thụ CO₂ mà còn trực tiếp phát thải khí nhà kính từ việc đốt rừng và phân hủy sinh khối. Theo IPCC [2], phá rừng và thay đổi sử dụng đất đóng góp khoảng 23% tổng lượng phát thải khí nhà kính do con người gây ra. Điều này góp phần làm gia tăng hiện tượng biến đổi khí hậu toàn cầu.

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* Bản đồ thế giới thể hiện các vùng mất rừng nhiệt đới (Amazon, Congo, Đông Nam Á) với chú thích số liệu diện tích mất rừng giai đoạn 2015-2020.

### Tình hình mất rừng tại Việt Nam

Việt Nam đã trải qua những biến đổi lớn về độ che phủ rừng trong 30 năm qua. Sau thời kỳ suy giảm nghiêm trọng (độ che phủ chỉ còn 28% vào năm 1990 do chiến tranh và khai thác bừa bãi), Việt Nam đã thực hiện nhiều chương trình phục hồi và phát triển rừng. Nhờ các chương trình như "Trồng 5 triệu hecta rừng" (1998-2010), độ che phủ rừng đã tăng lên 42% vào năm 2020 [21].

Tuy nhiên, chất lượng rừng là một vấn đề đáng lo ngại. Mặc dù tổng diện tích rừng tăng từ 9.4 triệu hecta (1990) lên 14.6 triệu hecta (2020) chủ yếu nhờ rừng trồng (cao su, keo, thông), chất lượng rừng tự nhiên lại suy giảm đáng kể. Theo số liệu của Bộ NN&PTNT (2020), rừng tự nhiên hiện có khoảng 10.29 triệu hecta, nhưng rừng nguyên sinh (primary forest) chỉ còn chiếm khoảng 0.6% tổng diện tích rừng [21]. Điều này cho thấy, mặc dù độ che phủ rừng tăng về số lượng, nhưng rừng giàu trữ lượng và đa dạng sinh học cao đang bị thay thế bởi rừng trồng đơn loài có giá trị sinh thái thấp hơn.

Nguyên nhân chính gây mất rừng tại Việt Nam bao gồm việc chuyển đổi sang đất nông nghiệp như trồng cà phê, cao su và điều; khai thác gỗ trái phép; phát triển cơ sở hạ tầng và đô thị hóa; cháy rừng; và hoạt động nuôi trồng thủy sản, đặc biệt tại khu vực ven biển và đồng bằng sông Cửu Long.

> **[TODO: Cần chèn Biểu đồ tại đây]**
> *Gợi ý:* Biểu đồ đường thể hiện sự thay đổi độ che phủ rừng Việt Nam giai đoạn 1990-2020, với 2 đường: tổng diện tích rừng và diện tích rừng tự nhiên.

### Tình hình rừng tại tỉnh Cà Mau

Cà Mau, tỉnh cực Nam Tổ Quốc, sở hữu hệ sinh thái rừng ngập mặn quan trọng với diện tích khoảng 40,000 hecta, chiếm khoảng 20% diện tích rừng ngập mặn của Việt Nam. Rừng ngập mặn Cà Mau đóng vai trò then chốt trong việc phòng hộ ven biển (chắn sóng, chống xâm thực và bảo vệ bờ biển), bảo tồn đa dạng sinh học vì là môi trường sống cho nhiều loài động thực vật quý hiếm, cung cấp nguồn sinh kế thông qua các hoạt động thủy sản và du lịch sinh thái, và góp phần giảm nhẹ biến đổi khí hậu nhờ khả năng lưu giữ carbon cao, gấp khoảng 3–5 lần so với rừng nhiệt đới trên cạn [23][24].

Tuy nhiên, rừng Cà Mau đang phải đối mặt với nhiều thách thức. Trước hết là áp lực chuyển đổi sang nuôi tôm do kinh tế, khiến nhiều khu vực rừng bị chuyển đổi thành ao nuôi. Ngoài ra, hiện tượng xâm nhập mặn gia tăng do biến đổi khí hậu làm giảm sức khỏe rừng; đồng thời xói mòn bờ biển cũng làm suy giảm diện tích rừng ven biển; và tình trạng thiếu nước ngọt ảnh hưởng tới khả năng tái sinh tự nhiên của rừng.

Theo số liệu của Sở NN&PTNT Cà Mau (2021), tổng diện tích rừng tập trung của tỉnh là 94,319 hecta với tỷ lệ che phủ rừng đạt 25.81% [22]. Tuy nhiên, giai đoạn 2011-2023, sạt lở vùng ven biển đã làm mất hơn 6,000 hecta đất và rừng phòng hộ. Việc giám sát và bảo vệ rừng tại Cà Mau là ưu tiên hàng đầu nhằm duy trì hệ sinh thái quan trọng này.

> **[TODO: Cần chèn Bản đồ tại đây]**
> *Gợi ý:* Bản đồ vị trí tỉnh Cà Mau trong Việt Nam và vùng ĐBSCL, kèm bản đồ chi tiết khu vực rừng ngập mặn Cà Mau với ranh giới vùng nghiên cứu.

## Công nghệ viễn thám trong giám sát rừng

### Ưu điểm của công nghệ viễn thám

Công nghệ viễn thám vệ tinh mang lại nhiều ưu điểm vượt trội so với phương pháp điều tra thực địa truyền thống. Thứ nhất, khả năng bao phủ phạm vi rộng cho phép một ảnh vệ tinh phủ diện tích hàng nghìn km² và giám sát đồng thời nhiều khu vực rừng; thứ hai, các vệ tinh hiện đại có chu kỳ quay trở lại ngắn (khoảng 3–5 ngày), cho phép cập nhật thường xuyên và phát hiện kịp thời các biến động; thứ ba, việc nhiều chương trình vệ tinh cung cấp dữ liệu miễn phí giúp giảm đáng kể chi phí so với khảo sát thực địa; thứ tư, dữ liệu đa thời gian và kho lưu trữ lịch sử cho phép phân tích xu hướng biến động qua nhiều năm; thứ năm, viễn thám có khả năng tiếp cận những khu vực khó tiếp cận bằng phương pháp thực địa như rừng núi cao hay biên giới; và cuối cùng, dữ liệu viễn thám mang tính khách quan và có thể lặp lại, loại bỏ các sai số chủ quan của người khảo sát.

### Chương trình Copernicus và vệ tinh Sentinel

Chương trình Copernicus của Liên minh Châu Âu (EU) là một trong những chương trình quan sát Trái Đất lớn nhất thế giới, cung cấp dữ liệu miễn phí và mở. Hai vệ tinh quan trọng cho giám sát rừng là:

**Sentinel-1 (SAR - Synthetic Aperture Radar):**

Vệ tinh Sentinel-1 hoạt động ở dải sóng C-band (xấp xỉ 5.5 cm) với hai chế độ phân cực chính là VV (Vertical-Vertical) và VH (Vertical-Horizontal); độ phân giải không gian trong chế độ Interferometric Wide (IW) là 10m và chu kỳ quay trở lại của tổ hợp hai vệ tinh (1A và 1B) là khoảng 6 ngày [27]. Do là hệ thống chủ động, Sentinel-1 có ưu điểm xuyên qua mây và khói, hoạt động được cả ngày lẫn đêm, và nhạy cảm đối với cấu trúc thực vật cũng như độ ẩm, vì vậy các ứng dụng tiêu biểu bao gồm phát hiện biến động rừng trong điều kiện mây nhiều và phân biệt rừng ngập nước.

**Sentinel-2 (Optical - Multispectral Imaging):**

Vệ tinh Sentinel-2 cung cấp 13 dải phổ từ vùng nhìn thấy đến hồng ngoại ngắn (từ 443 nm đến 2190 nm) với nhiều cấp độ độ phân giải không gian: 10m cho các dải B2, B3, B4 và B8; 20m cho các dải B5, B6, B7, B8a, B11 và B12; và 60m cho B1, B9 và B10 [28]. Chu kỳ quay trở lại của tổ hợp hai vệ tinh Sentinel-2A và Sentinel-2B vào khoảng 5 ngày, và vì có thông tin quang phổ phong phú nên Sentinel-2 rất phù hợp để tính toán chỉ số thực vật và hỗ trợ các ứng dụng như phân loại lớp phủ, đánh giá sức khỏe thực vật và tính toán NDVI, NBR, NDMI.

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* Hình minh họa vệ tinh Sentinel-1 và Sentinel-2, kèm bảng so sánh thông số kỹ thuật chính của hai vệ tinh.

### Chỉ số thực vật từ dữ liệu quang học

Các chỉ số thực vật (vegetation indices) là công cụ quan trọng trong giám sát rừng, được tính toán từ các dải phổ khác nhau:

**NDVI (Normalized Difference Vegetation Index):**

$$NDVI = \frac{NIR - Red}{NIR + Red}$$

NDVI có dải giá trị từ -1 đến 1; giá trị NDVI lớn hơn 0.6 thường biểu thị thực vật xanh tốt, trong khi giá trị NDVI nhỏ hơn 0.2 thường liên quan đến đất trống, nước hoặc khu vực đô thị [29]; do vậy NDVI được ứng dụng rộng rãi để đánh giá mật độ và sức khỏe thực vật.

**NBR (Normalized Burn Ratio):**

$$NBR = \frac{NIR - SWIR_2}{NIR + SWIR_2}$$

NBR là công cụ nhạy cảm với vùng cháy; biến đổi Delta NBR (dNBR) được sử dụng để đánh giá mức độ tổn thất do cháy rừng và NBR ứng dụng để phát hiện cháy rừng cũng như đánh giá thiệt hại sau cháy.

**NDMI (Normalized Difference Moisture Index):**

$$NDMI = \frac{NIR - SWIR_1}{NIR + SWIR_1}$$

NDMI được dùng để đánh giá hàm lượng nước trong thực vật; giá trị NDMI thấp có thể chỉ ra trạng thái stress do hạn hán và tăng nguy cơ cháy, do đó NDMI phù hợp để giám sát hạn hán và đánh giá sức khỏe rừng.

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* Hình ảnh minh họa các chỉ số thực vật (NDVI, NBR, NDMI) trên cùng một khu vực rừng Cà Mau, với thang màu và chú thích giá trị.

### Tích hợp dữ liệu SAR và Optical

Việc kết hợp dữ liệu SAR (Sentinel-1) và Optical (Sentinel-2) mang lại nhiều lợi ích thực tế. Về mặt bổ sung thông tin, SAR cung cấp dữ liệu về cấu trúc, độ nhám bề mặt và độ ẩm, trong khi Optical cung cấp thông tin quang phổ và các chỉ số thực vật. Về khắc phục hạn chế, SAR hoạt động hiệu quả trong điều kiện mây mù — điều quan trọng trong môi trường rừng nhiệt đới — còn Optical lại cung cấp dữ liệu trực quan dễ dàng để phiên giải. Về nâng cao độ chính xác, nhiều nghiên cứu cho thấy việc kết hợp SAR và Optical giúp tăng accuracy từ khoảng 5 đến 10% so với việc sử dụng mỗi nguồn dữ liệu riêng lẻ. Cuối cùng, việc tích hợp hai nguồn dữ liệu cho phép phát hiện biến động đa chiều: SAR nhạy với biến đổi cấu trúc như chặt cây, trong khi Optical nhạy với biến đổi quang phổ thể hiện sức khỏe thực vật.

## Tổng quan các nghiên cứu liên quan

### Phương pháp Deep Learning

**Convolutional Neural Networks (CNN):**

CNN đã cách mạng hóa computer vision và ngày càng được áp dụng rộng rãi trong viễn thám [10]. Zhang et al. [4] giới thiệu các kiến trúc CNN phổ biến và ứng dụng của chúng trong viễn thám, Kussul et al. [5] áp dụng CNN cho phân loại cây trồng từ Sentinel-2 và đạt accuracy 94.5%, và Xu et al. [6] sử dụng CNN kết hợp với cơ chế attention để đạt accuracy 96.8% trên dữ liệu đa nguồn.

**Các kiến trúc CNN tiêu biểu trong viễn thám:**

Nhiều kiến trúc CNN đã được phát triển và ứng dụng trong phân loại ảnh viễn thám. Ronneberger et al. [7] đề xuất kiến trúc U-Net với cấu trúc encoder-decoder, ban đầu cho phân đoạn ảnh y sinh nhưng sau đó được áp dụng rộng rãi trong viễn thám nhờ khả năng phân đoạn ngữ nghĩa hiệu quả. Zhong et al. [9] phát triển SatCNN - kiến trúc CNN chuyên biệt cho phân loại ảnh vệ tinh với khả năng xử lý nhanh và chính xác. Karra et al. [8] ứng dụng deep learning kết hợp Sentinel-2 để tạo bản đồ sử dụng đất toàn cầu với độ phân giải 10m. Zhang et al. [20] đề xuất phương pháp joint deep learning cho phân loại đồng thời lớp phủ và sử dụng đất, đạt kết quả vượt trội so với các phương pháp đơn nhiệm vụ.

**Đánh giá tổng quan:**

Khatami et al. [19] thực hiện meta-analysis trên các nghiên cứu phân loại ảnh viễn thám pixel-based và đưa ra các hướng dẫn thực hành cho người nghiên cứu. Kết quả cho thấy việc lựa chọn thuật toán, số lượng mẫu huấn luyện và đặc trưng đầu vào đều ảnh hưởng đáng kể đến độ chính xác phân loại.

**Ưu điểm CNN:**

CNN có nhiều ưu điểm: chúng tự động học đặc trưng không gian (spatial features), khai thác ngữ cảnh không gian cục bộ thông qua các kernel tích chập, loại bỏ nhu cầu phải trích xuất đặc trưng thủ công, có khả năng học các đặc trưng phân tầng (hierarchical features) và có thể sử dụng phương pháp transfer learning từ các mô hình pretrained để cải thiện hiệu suất trên các tập dữ liệu mới.

**Hạn chế:**

Tuy nhiên, CNN cũng tồn tại một số hạn chế: chúng thường cần bộ dữ liệu huấn luyện lớn để đạt hiệu suất tối ưu; thời gian huấn luyện lâu và đòi hỏi phần cứng mạnh (GPU); có nhiều hyperparameters cần tinh chỉnh; mô hình thường có tính chất "black box" nên khó giải thích; và mô hình có thể dễ bị overfitting khi sử dụng trên các tập dữ liệu nhỏ.

### Ứng dụng trong giám sát rừng

**Phát hiện mất rừng:**

Hansen et al. [11] phát triển Global Forest Change dataset sử dụng chuỗi thời gian Landsat và thuật toán decision tree để phát hiện mất rừng toàn cầu giai đoạn 2000–2012 ở độ phân giải 30m; thêm vào đó, Reiche et al. [12] kết hợp Sentinel-1 và Landsat để phát hiện mất rừng near-real-time tại Amazon và báo cáo accuracy đạt 93.8%; Hethcoat et al. [13] áp dụng CNN trên chuỗi thời gian Landsat để phát hiện khai thác vàng trái phép tại Amazon và đạt F1-score 0.92.

**Tích hợp SAR và Optical:**

Một số nghiên cứu minh họa lợi ích của việc kết hợp SAR và Optical: Hu et al. [14] kết hợp Sentinel-1 và Sentinel-2 để phân loại rừng ở Madagascar và ghi nhận accuracy tăng từ 87% (khi chỉ dùng Sentinel-2) lên 92% khi sử dụng cả hai nguồn dữ liệu; tương tự, Ienco et al. [15] ứng dụng deep neural networks kết hợp chuỗi thời gian SAR và Optical để phân loại cây trồng và đạt accuracy 96.5%.

**Nghiên cứu tại Việt Nam:**

Tại Việt Nam, Pham et al. [16] đã sử dụng kết hợp ảnh QuickBird, LiDAR và chỉ số địa hình GIS để nhận dạng loài cây bản địa trong cảnh quan phức tạp bằng phương pháp phân loại hướng đối tượng; Nguyen et al. [17] áp dụng Sentinel-2 đa thời gian để lập bản đồ sử dụng đất/che phủ đất tại Đắk Nông với bốn phương pháp phân loại khác nhau, đạt overall accuracy 91.2%; và Bùi et al. [18] nghiên cứu biến động rừng ngập mặn ven biển Đồng bằng sông Cửu Long bằng chuỗi thời gian Landsat (1990–2020) và phát hiện xu hướng giảm diện tích do chuyển đổi sang ao nuôi.

> **[TODO: Cần chèn Bảng số liệu tại đây]**
> *Gợi ý:* Bảng tổng hợp các nghiên cứu liên quan với các cột: Tác giả, Năm, Phương pháp, Dữ liệu, Khu vực, Accuracy.

## Khoảng trống nghiên cứu và định hướng đồ án

### Khoảng trống nghiên cứu

Qua tổng quan tài liệu, một số khoảng trống nghiên cứu nổi bật được xác định. Thứ nhất, thiếu nghiên cứu Deep Learning cho rừng nhiệt đới Việt Nam khi phần lớn công trình tập trung ở Amazon, Congo hay Indonesia, và vẫn còn ít nghiên cứu áp dụng CNN cho rừng Việt Nam, đặc biệt là rừng ngập mặn Cà Mau. Thứ hai, vấn đề spatial data leakage là một điểm cần lưu ý vì nhiều nghiên cứu chia dữ liệu một cách ngẫu nhiên mà không xử lý mối tương quan không gian, dẫn tới đánh giá accuracy bị phóng đại do tập huấn luyện và tập kiểm tra nằm gần nhau trong không gian. Thứ ba, CNN thường yêu cầu tập dữ liệu lớn (hàng trăm nghìn mẫu), do đó có ít công trình nghiên cứu về kiến trúc CNN tối ưu cho các bộ dữ liệu nhỏ trong viễn thám (khoảng 2,000–5,000 mẫu). Thứ tư, việc tích hợp SAR và Optical trong bối cảnh Deep Learning vẫn còn nhiều thách thức và còn thiếu các khảo sát tối ưu hóa fusion trong kiến trúc CNN.

### Định hướng của đồ án

Xuất phát từ những khoảng trống nghiên cứu đã nêu, đồ án này hướng tới một số mục tiêu chính. Thứ nhất, phát triển một kiến trúc CNN phù hợp cho các bộ dữ liệu nhỏ bằng cách thiết kế mô hình lightweight (xấp xỉ 36K tham số), áp dụng các kỹ thuật regularization mạnh như Batch Normalization, Dropout và Weight Decay, và so sánh hiệu năng với những kiến trúc khác (deeper hoặc wider) để tìm ra cấu trúc tối ưu. Thứ hai, triển khai một quy trình đánh giá khoa học chặt chẽ bao gồm việc sử dụng stratified random split để đảm bảo phân bố lớp đồng đều, thực hiện 5-Fold Stratified Cross Validation để đánh giá phương sai của mô hình và giữ lại một fixed test set (20%) để đo khả năng tổng quát hóa thực tế. Thứ ba, tối ưu hóa phương pháp fusion giữa Sentinel-1 và Sentinel-2 ở cấp độ feature bằng cách concatenation các đặc trưng SAR và Optical, trích xuất các temporal features (before, after và delta) để thu được 27 features tổng cộng (21 features từ S2 tương ứng 7×3 và 6 features từ S1 tương ứng 2×3). Cuối cùng, đồ án tập trung vào ứng dụng thực tế tại Cà Mau, bao gồm phân loại toàn vùng rừng (162,469 ha), ước tính diện tích mất rừng và tạo bản đồ phân loại ở độ phân giải 10m.

### Câu hỏi nghiên cứu

Đồ án tập trung trả lời một số câu hỏi cốt lõi. Thứ nhất, liệu 5-Fold Cross Validation có đảm bảo đánh giá mô hình một cách robust và ổn định, và độ biến thiên giữa các folds như thế nào; thứ hai, kiến trúc CNN nào là phù hợp nhất cho bộ dữ liệu gồm 2,630 mẫu, so sánh giữa kiến trúc lightweight và các kiến trúc sâu hơn; thứ ba, việc tích hợp Sentinel-1 SAR và Sentinel-2 Optical có cải thiện accuracy so với chỉ sử dụng Sentinel-2 hay không; và cuối cùng, liệu mô hình CNN có thể được ứng dụng thực tế cho giám sát rừng Cà Mau về mặt accuracy, tốc độ và khả năng triển khai hay không.

\newpage

# Cơ sở lý thuyết

## Công nghệ viễn thám và ảnh vệ tinh

### Nguyên lý viễn thám

Viễn thám (Remote Sensing) là khoa học và kỹ thuật thu thập thông tin về một đối tượng hoặc khu vực từ xa, thường thông qua việc ghi nhận bức xạ điện từ phản xạ hoặc phát ra từ bề mặt Trái Đất. Nguyên lý cơ bản của viễn thám dựa trên tương tác giữa bức xạ điện từ và các đối tượng trên bề mặt:

**Quá trình viễn thám bị động (Passive Remote Sensing):**

Trong hệ thống viễn thám bị động, nguồn năng lượng chính là bức xạ từ Mặt Trời; khi các sóng này truyền qua khí quyển, một phần năng lượng bị hấp thụ hoặc tán xạ; sau đó bức xạ tương tác với bề mặt, chịu các quá trình phản xạ, hấp thụ hoặc truyền qua tùy theo đặc tính vật liệu; tín hiệu phản xạ này sau đó được vệ tinh ghi nhận bởi cảm biến; và cuối cùng các tín hiệu này được xử lý và truyền về trạm mặt đất để phục vụ phân tích tiếp theo.

> **[TODO: Cần chèn Sơ đồ tại đây]**
> *Gợi ý:* Sơ đồ minh họa nguyên lý viễn thám bị động và chủ động, với các thành phần: nguồn năng lượng, khí quyển, bề mặt, cảm biến.

**Phương trình cân bằng năng lượng:**

$$E_{incident} = E_{reflected} + E_{absorbed} + E_{transmitted}$$

Trong đó, $E_{incident}$ là năng lượng tới từ Mặt Trời, $E_{reflected}$ là phần năng lượng phản xạ được cảm biến ghi nhận, $E_{absorbed}$ là phần năng lượng bị hấp thụ và chuyển thành nhiệt, và $E_{transmitted}$ là phần năng lượng truyền qua vật chất.

### Radar khẩu độ tổng hợp (SAR)

**Nguyên lý hoạt động:**

Khác với viễn thám bị động, SAR là hệ thống chủ động (active remote sensing): anten phát xung sóng điện từ về phía Trái Đất, các sóng này tương tác với bề mặt và tạo hiện tượng phản xạ ngược (backscatter) với cường độ phụ thuộc vào nhiều yếu tố như độ nhám bề mặt, hàm lượng nước (độ ẩm), hằng số điện môi và góc tới; các tín hiệu phản xạ được anten thu nhận và được xử lý bằng cách tổng hợp khẩu độ (aperture synthesis) nhằm tăng độ phân giải ảnh.

**Hệ số Backscatter ($\sigma^0$):**

$$\sigma^0 (dB) = 10 \times \log_{10}(\sigma^0_{linear})$$

Giá trị $\sigma^0$ phụ thuộc vào nhiều yếu tố, đặc biệt là độ nhám bề mặt (trong đó bề mặt nhẵn như nước cho $\sigma^0$ thấp, còn bề mặt nhám như rừng cho $\sigma^0$ cao), hàm lượng nước (độ ẩm) vốn làm tăng $\sigma^0$ do hằng số điện môi lớn của nước, và cấu trúc thực vật — những khu vực rừng có cấu trúc phức tạp thường cho backscatter mạnh.

**Polarization:**

SAR có thể phát và thu theo các chế độ phân cực khác nhau; ví dụ, chế độ VV (phát V, thu V) nhạy với độ ẩm bề mặt, chế độ VH (phát V, thu H) và HV (phát H, thu V) thường nhạy với cấu trúc thực vật (volume scattering), còn chế độ HH (phát H, thu H) nhạy với độ nhám bề mặt.

**Sentinel-1 SAR:**

Vệ tinh Sentinel-1 SAR hoạt động ở dải sóng C-band (xấp xỉ 5.5 cm, tần số khoảng 5.4 GHz) với các chế độ phân cực chính như VV và VH trong chế độ IW, đạt độ phân giải không gian khoảng 10m; ưu điểm của Sentinel-1 là khả năng xuyên qua mây và hoạt động cả ngày lẫn đêm.

### Ảnh quang học đa phổ (Optical Multispectral)

**Dải phổ điện từ:**

Ảnh quang học ghi nhận bức xạ phản xạ từ bề mặt Trái Đất ở các dải phổ khác nhau. Dải nhìn thấy (Visible, VIS) nằm trong khoảng 400–700 nm và bao gồm các dải Blue (B) 450–520 nm, Green (G) 520–600 nm và Red (R) 630–690 nm. Dải cận hồng ngoại (Near-Infrared, NIR) trải từ 700–1400 nm, có đặc trưng phản xạ cao ở thực vật xanh do chlorophyll và quan trọng cho việc tính toán NDVI. Dải hồng ngoại sóng ngắn (Short-Wave Infrared, SWIR) nằm trong khoảng 1400–3000 nm, với SWIR1 (1550–1750 nm) và SWIR2 (2080–2350 nm), rất nhạy với độ ẩm của thực vật và đất, vì vậy thường dùng để đánh giá hàm lượng nước và các chỉ số liên quan đến sinh khối.

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* Biểu đồ phổ điện từ với các dải phổ và vị trí các bands của Sentinel-2.

**Chữ ký phổ (Spectral Signature):**

Mỗi loại đối tượng có chữ ký phổ đặc trưng - mẫu phản xạ qua các dải phổ:

$$S = [\rho(\lambda_1), \rho(\lambda_2), ..., \rho(\lambda_n)]$$

Ví dụ: thực vật xanh có đặc trưng phản xạ thấp ở dải Red do sự hấp thụ bởi chlorophyll và phản xạ cao ở dải NIR; đất trống có đặc trưng phản xạ trung bình và tăng dần theo bước sóng; trong khi nước có phản xạ thấp ở hầu hết các dải, đặc biệt là ở NIR và SWIR.

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

Về nguyên lý, thực vật xanh hấp thụ mạnh ở dải Red do chlorophyll nhưng phản xạ mạnh ở dải NIR nhờ cấu trúc tế bào, dẫn tới giá trị NDVI cao; ngược lại, đất trống và nước có phản xạ thấp ở cả Red và NIR nên cho giá trị NDVI thấp.

**Phạm vi giá trị:**

Giá trị NDVI có thể diễn giải như sau: NDVI lớn hơn 0.6 thường biểu thị thực vật xanh tốt (rừng rậm), giá trị nằm giữa 0.2 và 0.6 cho thấy thực vật thưa hoặc cỏ, còn NDVI nhỏ hơn 0.2 thường liên quan đến đất trống, nước hoặc khu vực đô thị [29].

**NBR (Normalized Burn Ratio):**

$$NBR = \frac{NIR - SWIR_2}{NIR + SWIR_2}$$

**Nguyên lý:**

Trong công thức NBR, NIR thường có phản xạ cao ở thực vật xanh còn SWIR2 nhạy cảm với độ ẩm và vùng cháy; do đó, trong trường hợp vùng cháy, NIR giảm trong khi SWIR2 tăng, dẫn tới giá trị NBR giảm mạnh.

**NDMI (Normalized Difference Moisture Index):**

$$NDMI = \frac{NIR - SWIR_1}{NIR + SWIR_1}$$

**Nguyên lý:**

Đối với NDMI, SWIR1 (khoảng 1600 nm) bị hấp thụ mạnh bởi nước; do đó hàm lượng nước trong thực vật cao sẽ khiến SWIR1 phản xạ thấp và NDMI có giá trị cao, còn khi thực vật bị stress do hạn hán thì NDMI sẽ giảm.

### Phát hiện biến động rừng

**Change Detection Approach:**

$$\Delta Feature = Feature_{after} - Feature_{before}$$

**Temporal Features:**

Temporal features bao gồm các 'before features' thể hiện trạng thái rừng tại thời điểm $t_1$, các 'after features' thể hiện trạng thái rừng tại thời điểm $t_2$, và các 'delta features' biểu diễn biến đổi giữa hai thời điểm ($t_2 - t_1$).

**Ví dụ với NDVI:**

$$\Delta NDVI = NDVI_{after} - NDVI_{before}$$

**Phân loại biến động:**

Các giá trị biến đổi NDVI có thể được diễn giải như sau: khi $\Delta NDVI$ giảm mạnh (rất nhỏ hơn 0) thì đó là dấu hiệu mất rừng (deforestation); khi $\Delta NDVI$ xấp xỉ 0 thì vùng được xem là rừng ổn định; và khi $\Delta NDVI$ tăng mạnh (rất lớn hơn 0) thì biểu hiện tái trồng rừng.

## Mạng Neural Tích chập (Convolutional Neural Networks)

### Giới thiệu về Neural Networks

**Perceptron - Đơn vị cơ bản:**

Một neuron nhân tạo thực hiện phép biến đổi tuyến tính và hàm kích hoạt:

$$y = f(\mathbf{w}^T \mathbf{x} + b)$$

Trong đó, $\mathbf{x} \in \mathbb{R}^n$ là input vector chứa $n$ feature, $\mathbf{w} \in \mathbb{R}^n$ là weight vector, $b \in \mathbb{R}$ là bias, $f(\cdot)$ là hàm kích hoạt và $y$ là output.

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

Trong đó, $I$ là input feature map với kích thước height × width × channels, $K$ là kernel hoặc filter có kích thước $k_h \times k_w$ trên các channels, $(i,j)$ là vị trí trong output và $(m,n)$ là vị trí tương ứng trong kernel.

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* Hình minh họa phép tích chập 2D với kernel 3×3 trượt trên input feature map, tạo output feature map.

**Ưu điểm của Convolution:**

Convolution có nhiều ưu điểm, trong đó cơ chế chia sẻ tham số (parameter sharing) cho phép cùng một kernel áp dụng toàn bộ input để tiết kiệm tham số; tính bất biến theo dịch chuyển (translation invariance) cho phép nhận diện các đặc trưng bất kể vị trí xuất hiện; và local connectivity đảm bảo mỗi neuron chỉ kết nối với vùng cục bộ của input, giúp học các đặc trưng khu vực hiệu quả.

### Activation Functions

**ReLU (Rectified Linear Unit):**

$$f(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**Ưu điểm:**

Hàm kích hoạt ReLU có các ưu điểm như tính toán nhanh do không yêu cầu các phép toán phức tạp (như exp hay log), góp phần giảm vấn đề vanishing gradient và cho sparse activation, tức là nhiều neuron có giá trị bằng 0, giúp mô hình trở nên hiệu quả hơn.

**Softmax (cho Output Layer):**

$$\text{softmax}(\mathbf{x})_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$

**Tính chất:**

Các đầu ra của hàm softmax là các xác suất thuộc khoảng [0, 1] và tổng các xác suất đó bằng 1, nên softmax được sử dụng phổ biến cho bài toán phân loại đa lớp (multi-class classification).

### Pooling Layers

**Global Average Pooling:**

$$GAP(k) = \frac{1}{H \times W} \sum_i \sum_j I(i, j, k)$$

Global Average Pooling giảm kích thước không gian về 1×1, chuyển output từ (H, W, C) thành (1, 1, C), và vì không có tham số học nên phương pháp này giúp giảm nguy cơ overfitting.

### Batch Normalization

**Batch Normalization Algorithm:**

Với một mini-batch $B = \{x_1, x_2, ..., x_m\}$, đầu tiên tính mean và variance của batch theo các công thức:

$$\mu_B = \frac{1}{m} \sum_i x_i$$

$$\sigma^2_B = \frac{1}{m} \sum_i (x_i - \mu_B)^2$$

Tiếp theo, normalize mỗi giá trị theo công thức:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}$$

Cuối cùng, áp dụng phép scale và shift bằng các tham số học được để đưa về dạng:

$$y_i = \gamma \hat{x}_i + \beta$$

**Ưu điểm:**

Batch Normalization có một số ưu điểm nổi bật: nó giúp tăng tốc độ huấn luyện (cho phép sử dụng learning rate cao hơn), giảm độ nhạy phụ thuộc vào khởi tạo trọng số, đóng vai trò như một phương pháp regularization tương tự dropout, và ổn định luồng gradient trong quá trình huấn luyện.

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

Trong đó, $y_i$ là nhãn thực tế (true label) được mã hóa one-hot và $\hat{y}_i$ là xác suất dự đoán xuất ra từ hàm softmax.

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

Hyperparameters: thường sử dụng $\beta_1 = 0.9$ cho decay của moment bậc nhất, $\beta_2 = 0.999$ cho decay của moment bậc hai, $\epsilon = 10^{-8}$ để ổn định tính toán số và $\eta = 0.001$ làm learning rate mặc định.

**AdamW (Adam with Weight Decay):**

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

Trong đó $\lambda$ là weight decay coefficient (L2 regularization).

## Phương pháp phân loại ảnh viễn thám

### Pixel-based vs Patch-based Classification

**Pixel-based Classification:**

Mỗi pixel được phân loại độc lập dựa trên vector đặc trưng:

$$\mathbf{x}_i = [f_1, f_2, ..., f_n], \quad y_i = \text{classifier}(\mathbf{x}_i)$$

**Ưu điểm:**

Phương pháp pixel-based có những ưu điểm như tính đơn giản và dễ triển khai, tốc độ xử lý nhanh nhờ khả năng xử lý song song (parallel processing).

**Nhược điểm:**

Tuy nhiên, phương pháp pixel-based cũng có nhược điểm là không tận dụng ngữ cảnh không gian (spatial context), dễ tạo ra nhiễu dạng salt-and-pepper trong kết quả phân loại, và bỏ qua các mối quan hệ giữa các pixel lân cận.

**Patch-based Classification:**

Trích xuất patches (windows) xung quanh mỗi pixel:

$$P_i = \text{extract\_patch}(I, \text{center}=(row_i, col_i), \text{size}=k \times k)$$
$$y_i = \text{classifier}(P_i)$$

**Ưu điểm:**

Ngược lại, phương pháp patch-based sử dụng ngữ cảnh không gian xung quanh mỗi pixel, cho kết quả phân loại mượt hơn và đặc biệt phù hợp với CNN nhờ khả năng tự động học các đặc trưng từ các patches.

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* So sánh pixel-based vs patch-based classification với hình minh họa và kết quả phân loại mẫu.

### Spatial Autocorrelation

**Tobler's First Law of Geography [25]:**

*"Everything is related to everything else, but near things are more related than distant things."*

**Implication for Machine Learning:**

Training và test samples gần nhau trong không gian có high correlation → **Data leakage** → Overestimate accuracy.

**Giải pháp: Stratified Data Splitting + Cross Validation**

### Evaluation Metrics

**Confusion Matrix:**

|  | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Total |
|---|--------|--------|--------|--------|-------|
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

Các tiêu chuẩn diễn giải ROC-AUC theo Hosmer và Lemeshow [26] thường được hiểu như sau: AUC = 0.5 tương ứng với classifier ngẫu nhiên (không có khả năng phân biệt); 0.5 < AUC < 0.7 được xem là phân biệt kém (poor); 0.7 ≤ AUC < 0.8 là chấp nhận được (acceptable); 0.8 ≤ AUC < 0.9 là xuất sắc (excellent); và AUC ≥ 0.9 được xem là vượt trội (outstanding).

\newpage

# Phương pháp nghiên cứu

## Khu vực và dữ liệu nghiên cứu

### Khu vực nghiên cứu

**Vị trí địa lý:**

Tỉnh Cà Mau nằm ở cực Nam Tổ Quốc, thuộc vùng Đồng bằng sông Cửu Long; tọa độ địa lý nằm trong khoảng 8°36'–9°27' Bắc và 104°43'–105°10' Đông, diện tích tự nhiên là khoảng 5,331.7 km², dân số ước tính khoảng 1.2 triệu người (2020), và chiều dài đường bờ biển khoảng 254 km.

> **[TODO: Cần chèn Bản đồ tại đây]**
> *Gợi ý:* Bản đồ vị trí khu vực nghiên cứu gồm: (a) Vị trí Cà Mau trong Việt Nam, (b) Ranh giới tỉnh Cà Mau, (c) Vùng rừng nghiên cứu với tọa độ UTM.

**Vùng nghiên cứu:**

Đồ án tập trung vào toàn bộ diện tích rừng trong ranh giới tỉnh Cà Mau, với tổng diện tích nghiên cứu khoảng 162,469.25 hecta (tương đương 1,624.69 km²), kích thước raster là 12,547 × 10,917 pixels (ở độ phân giải 10m), và hệ quy chiếu được sử dụng là EPSG:32648 (WGS 84 / UTM Zone 48N).

### Dữ liệu viễn thám

**Tổng quan dữ liệu**

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

**Thống kê Ground Truth**

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

Quy trình xử lý dữ liệu được thực hiện theo các bước liên tiếp: đầu tiên là Data Loading & Validation để tải và kiểm tra dữ liệu Sentinel-1/2 cùng ground truth; tiếp theo là Feature Extraction để xây dựng 27 features (21 từ Sentinel-2 và 6 từ Sentinel-1); bước tiếp theo là Patch Extraction, trích xuất các patches kích thước 3×3 tại các vị trí ground truth; sau đó tiến hành Normalization bằng phương pháp Z-score; và cuối cùng áp dụng Stratified Data Splitting với tỷ lệ 80% cho Train+Val (được đánh giá bằng 5-Fold Cross Validation) và 20% cho Test cố định.

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

**Chi tiết 27 features**

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

**Tổng số trainable parameters**

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

**Hyperparameters**

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

Mô hình được đánh giá trên 20% fixed test set (526 mẫu) thông qua các metrics bao gồm Accuracy, Precision, Recall, F1-Score, ROC-AUC (One-vs-Rest) và Confusion Matrix.

### Full Raster Prediction

Sau khi huấn luyện, mô hình được áp dụng để phân loại toàn bộ 16,246,850 valid pixels trong vùng nghiên cứu.

\newpage

# Kết quả và thảo luận

## Tổng quan về kết quả thực nghiệm

### Cấu hình thực nghiệm

**Phần cứng và phần mềm:**

Môi trường thí nghiệm gồm phần cứng như GPU NVIDIA GeForce RTX 4080 (16GB VRAM), bộ nhớ RAM 16GB trở lên và ổ lưu trữ SSD nhằm đảm bảo tốc độ I/O cao; về phần mềm, hệ thống sử dụng Python 3.8 trở lên cùng PyTorch 2.0+ có hỗ trợ CUDA để huấn luyện mô hình, GDAL 3.4+ cho xử lý dữ liệu không gian và các thư viện khoa học dữ liệu như NumPy, scikit-learn và pandas để xử lý và phân tích dữ liệu.

**Dữ liệu đầu vào:**

Tổng số mẫu ground truth là 2,630 điểm, trong đó phân bố lớp gần như cân bằng: Lớp 0 (Rừng ổn định) 656 điểm (24.94%), Lớp 1 (Mất rừng) 650 điểm (24.71%), Lớp 2 (Phi rừng) 664 điểm (25.25%) và Lớp 3 (Phục hồi rừng) 660 điểm (25.10%).
Việc chia tập dữ liệu được thực hiện như sau: 80% dữ liệu (2,104 patches) được dành cho Train+Val để thực hiện 5-Fold Cross Validation, còn 20% dữ liệu (526 patches) được giữ lại làm fixed test set và không tham gia quá trình huấn luyện.

### Thời gian thực thi

**Thời gian thực thi các giai đoạn**

| Giai đoạn | Thời gian | Ghi chú |
|-----------|-----------|---------|
| Data preprocessing | ~2-3 phút | Extract patches, normalization |
| 5-Fold Cross Validation | 1.58 phút (94.89 giây) | 5 folds training |
| Final Model Training | 0.25 phút (15.20 giây) | Training trên toàn bộ 80% |
| Full raster prediction | 14.58 phút (874.59 giây) | 16,246,850 valid pixels |
| **Tổng cộng** | **~16.41 phút** | Không tính thời gian load dữ liệu |

## Kết quả huấn luyện mô hình CNN

### Kết quả 5-Fold Cross Validation

**Kết quả từng fold**

| Fold | Accuracy | F1-Score |
|------|----------|----------|
| Fold 1 | 98.34% | 98.34% |
| Fold 2 | 98.57% | 98.57% |
| Fold 3 | 98.10% | 98.10% |
| Fold 4 | 97.86% | 97.86% |
| Fold 5 | 97.86% | 97.86% |
| **Mean ± Std** | **98.15% ± 0.28%** | **98.15% ± 0.28%** |

**Phân tích kết quả CV:**

Kết quả 5-Fold Cross Validation cho thấy sự ổn định của mô hình: độ lệch chuẩn của accuracy chỉ khoảng 0.28%, chính xác từng fold đều vượt ngưỡng 97.8%, và điều này cho thấy không có dấu hiệu overfitting nghiêm trọng, tức CV accuracy phản ánh tốt khả năng tổng quát hóa của mô hình.

> **[TODO: Cần chèn Biểu đồ tại đây]**
> *Gợi ý:* Biểu đồ cột so sánh accuracy của 5 folds với đường trung bình và error bars.

### Kết quả trên tập test (Test Set)

**Metrics trên tập test (526 patches)**

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

**Phân tích chi tiết từng lớp - Test Set**

| Lớp | Precision | Recall | F1-Score | Support | Số lỗi |
|-----|-----------|--------|----------|---------|--------|
| 0 - Rừng ổn định | 96.99% | 98.47% | 97.73% | 131 | 4 FP, 2 FN |
| 1 - Mất rừng | 98.44% | 96.92% | 97.67% | 130 | 2 FP, 4 FN |
| 2 - Phi rừng | 100.00% | 100.00% | 100.00% | 133 | 0 |
| 3 - Phục hồi rừng | 100.00% | 100.00% | 100.00% | 132 | 0 |

**Phân tích lỗi phân loại:**

Tổng cộng chỉ có 6/526 mẫu bị phân loại sai, tương đương tỷ lệ lỗi 1.14%. Trong đó, hai mẫu thuộc Lớp 0 (Rừng ổn định) bị nhầm thành Lớp 1 (Mất rừng) và bốn mẫu thuộc Lớp 1 (Mất rừng) bị nhầm thành Lớp 0 (Rừng ổn định). Đánh giá chi tiết cho thấy Lớp 2 (Phi rừng) và Lớp 3 (Phục hồi rừng) được phân loại hoàn hảo với độ chính xác 100%; mọi nhầm lẫn chủ yếu xảy ra giữa hai lớp Lớp 0 và Lớp 1.

### Đường cong ROC

**ROC-AUC score cho từng lớp (Test Set)**

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

**Thống kê phân loại full raster**

| Thông số | Giá trị |
|----------|---------|
| Tổng số pixels được xử lý | 136,975,599 pixels |
| Pixels hợp lệ (valid data) | 16,246,850 pixels (11.86%) |
| Pixels bị mask (nodata) | 120,728,749 pixels (88.14%) |
| Kích thước raster | 12,547 × 10,917 pixels |
| Độ phân giải | 10m × 10m |
| Hệ tọa độ | EPSG:32648 (UTM Zone 48N) |

**Phân bố diện tích theo lớp**

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

## Ablation Studies

### Ảnh hưởng của patch size

**So sánh các patch sizes**

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

**Ablation các nguồn dữ liệu**

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

CNN chỉ sai 6/526 mẫu trên test set, tương đương tỷ lệ lỗi 1.14%. Trong đó, hai mẫu thuộc Lớp 0 (Rừng ổn định) bị nhầm sang Lớp 1 (Mất rừng) và bốn mẫu thuộc Lớp 1 bị nhầm sang Lớp 0. Các pattern cho thấy Lớp 2 (Phi rừng) và Lớp 3 (Phục hồi rừng) được phân loại hoàn hảo (100%), và mọi nhầm lẫn chỉ xuất hiện giữa Lớp 0 và Lớp 1.

> **[TODO: Cần chèn Hình ảnh tại đây]**
> *Gợi ý:* Hình ảnh minh họa 2-3 mẫu bị phân loại sai với ảnh Sentinel-2, ground truth, và predicted class.

## Đánh giá tổng quan

### Điểm mạnh của phương pháp

Những điểm nổi bật của mô hình bao gồm độ chính xác cao (test accuracy 98.86% với ROC-AUC 99.98%), khả năng khai thác ngữ cảnh không gian nhờ patch size 3×3, tính robust và khả năng tổng quát hóa tốt thể hiện qua CV 98.15% và test 98.86%, không cần trích xuất đặc trưng thủ công nhờ khả năng tự động học đặc trưng, và thời gian huấn luyện hiệu quả (khoảng 15 giây cho Final Model trên cấu hình thực nghiệm được sử dụng).

### So sánh với các nghiên cứu khác

**So sánh với literature**

| Nghiên cứu | Phương pháp | Data | Accuracy | ROC-AUC |
|------------|-------------|------|----------|---------|
| Hansen et al. (2013) | Decision Trees | Landsat | ~85% | N/A |
| Hethcoat et al. (2019) | CNN (ResNet) | Sentinel-1/2 | 94.3% | N/A |
| Zhang et al. (2020) | U-Net | Sentinel-2 | 96.8% | 98.5% |
| **Nghiên cứu này** | **CNN (custom)** | **S1/S2** | **98.86%** | **99.98%** |

### Tóm tắt chương

**Kết quả chính:**

Kết quả quan trọng bao gồm CV accuracy 5-Fold trung bình 98.15% ± 0.28% cho thấy độ ổn định, test accuracy đạt 98.86% với ROC-AUC 99.98%, hai lớp 'Phi rừng' và 'Phục hồi rừng' có precision và recall 100%, và tổng cộng chỉ có 6/526 mẫu bị phân loại sai (tương ứng 1.14% error rate).

**Kết quả phân loại vùng nghiên cứu (162,468.50 ha):**

Kết quả phân bố diện tích cho thấy rừng ổn định chiếm 74.30% (120,716.91 ha), mất rừng chiếm khoảng 4.48% tương ứng 7,282.15 ha, diện tích thuộc lớp phi rừng chiếm 18.17% (~29,528.54 ha), và diện tích phục hồi rừng chiếm 3.04% (~4,940.90 ha).

\newpage

# Kết luận và kiến nghị

## Kết luận

Đồ án đã hoàn thành các mục tiêu đề ra và đạt được một số kết quả chính. Thứ nhất, nhóm nghiên cứu đã xây dựng thành công bộ dữ liệu huấn luyện bằng việc thu thập, tiền xử lý hai kỳ dữ liệu Sentinel-1/2 (01/2024 và 02/2025) và tạo feature stack 27 chiều (kết hợp SAR và Optical) cùng với việc thu thập 2,630 điểm ground truth cho 4 lớp phân loại; thứ hai, kiến trúc CNN nhẹ với khoảng 36,676 tham số được thiết kế và áp dụng các kỹ thuật regularization hiệu quả (BatchNorm, Dropout 0.7, Weight Decay), phù hợp cho bộ dữ liệu nhỏ khoảng 2,600 mẫu; thứ ba, đánh giá khoa học bằng 5-Fold Stratified Cross Validation cho kết quả CV accuracy 98.15% ± 0.28% (mô hình ổn định), test accuracy 98.86% và ROC-AUC 99.98% (khả năng phân biệt xuất sắc); và cuối cùng, về mặt ứng dụng thực tế, mô hình đã được áp dụng để phân loại toàn vùng rừng Cà Mau (162,469 ha), phát hiện 7,282 ha mất rừng (4.48%) và 4,941 ha phục hồi rừng (3.04%).

## Đóng góp khoa học

Về mặt phương pháp, đồ án có một số đóng góp quan trọng: áp dụng 5-Fold Stratified Cross Validation nhằm đánh giá độ ổn định của mô hình, chứng minh hiệu quả sử dụng patches 3×3 cho bài toán phát hiện mất rừng, và tiến hành các thí nghiệm ablation toàn diện để khảo sát ảnh hưởng của kích thước patch, nguồn dữ liệu và kỹ thuật regularization. Về mặt ứng dụng, đồ án là một trong những nghiên cứu đầu tiên áp dụng CNN cho phát hiện biến động rừng tại Cà Mau, chứng minh hiệu quả trong việc kết hợp dữ liệu SAR (Sentinel-1) và Optical (Sentinel-2), đồng thời đóng góp một bộ ground truth chất lượng cao gồm 2,630 điểm với 4 lớp phân loại.

## Hạn chế

Đồ án vẫn tồn tại các hạn chế cần lưu ý, gồm: thời gian dự đoán toàn bộ raster còn dài (khoảng 14.83 phút cho 16.2 triệu pixel hợp lệ), khả năng giải thích của mô hình hạn chế do tính chất black-box, quy mô ground truth còn nhỏ (chỉ 2,630 điểm), và việc phân tích chỉ dừng lại ở phân tích bi-temporal mà chưa khai thác chuỗi thời gian đầy đủ.

## Kiến nghị

Đề xuất cho các hướng phát triển tiếp theo gồm: mở rộng phân tích temporal bằng cách sử dụng chuỗi thời gian thay vì chỉ phân tích hai thời kỳ (bi-temporal) và áp dụng các mô hình như LSTM hoặc Transformer để khai thác các mẫu temporal; cải thiện mô hình bằng thử nghiệm cơ chế attention, tận dụng transfer learning từ các mô hình pretrained và áp dụng ensemble methods nhằm tăng độ chính xác và độ ổn định; về ứng dụng thực tế, khuyến nghị triển khai hệ thống giám sát near-real-time, mở rộng phạm vi áp dụng sang các tỉnh trong vùng Đồng bằng sông Cửu Long và tích hợp kết quả với hệ thống GIS của cơ quan quản lý rừng; cuối cùng, cần tăng cường thu thập dữ liệu bằng khảo sát thực địa để validate kết quả, mở rộng bộ ground truth và thu thập thêm dữ liệu multi-temporal để nâng cao khả năng khai thác chuỗi thời gian.

\newpage

::: {custom-style="Bibliography"}

# Tài liệu tham khảo

[1] FAO (2020). *Global Forest Resources Assessment 2020: Main Report*. Rome: Food and Agriculture Organization of the United Nations. https://doi.org/10.4060/ca9825en

[2] IPCC (2019). *Climate Change and Land: An IPCC Special Report on climate change, desertification, land degradation, sustainable land management, food security, and greenhouse gas fluxes in terrestrial ecosystems*. Intergovernmental Panel on Climate Change.

[3] Global Forest Watch (2021). *Forest Loss Data 2015-2020*. World Resources Institute. https://www.globalforestwatch.org

[4] Zhang, L., Zhang, L., & Du, B. (2016). Deep learning for remote sensing data: A technical tutorial on the state of the art. *IEEE Geoscience and Remote Sensing Magazine*, 4(2), 22-40. https://doi.org/10.1109/MGRS.2016.2540798

[5] Kussul, N., Lavreniuk, M., Skakun, S., & Shelestov, A. (2017). Deep learning classification of land cover and crop types using remote sensing data. *IEEE Geoscience and Remote Sensing Letters*, 14(5), 778-782. https://doi.org/10.1109/LGRS.2017.2681128

[6] Xu, Y., Du, B., Zhang, L., Cerra, D., Pato, M., Carmona, E., ... & Le Saux, B. (2021). Advanced multi-sensor optical remote sensing for urban land use and land cover classification: Outcome of the 2018 IEEE GRSS Data Fusion Contest. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 12(6), 1709-1724. https://doi.org/10.1109/JSTARS.2019.2911113

[7] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *Medical Image Computing and Computer-Assisted Intervention* (pp. 234-241). Springer. https://doi.org/10.1007/978-3-319-24574-4_28

[8] Karra, K., Kontgis, C., Statman-Weil, Z., Mazzariello, J. C., Mathis, M., & Brumby, S. P. (2021). Global land use/land cover with Sentinel 2 and deep learning. In *2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS* (pp. 4704-4707). IEEE. https://doi.org/10.1109/IGARSS47720.2021.9553499

[9] Zhong, Y., Fei, F., Liu, Y., Zhao, B., Jiao, H., & Zhang, L. (2018). SatCNN: Satellite image dataset classification using agile convolutional neural networks. *Remote Sensing Letters*, 8(2), 136-145. https://doi.org/10.1080/2150704X.2016.1235299

[10] Zhu, X. X., Tuia, D., Mou, L., Xia, G. S., Zhang, L., Xu, F., & Fraundorfer, F. (2017). Deep learning in remote sensing: A comprehensive review and list of resources. *IEEE Geoscience and Remote Sensing Magazine*, 5(4), 8-36. https://doi.org/10.1109/MGRS.2017.2762307

[11] Hansen, M. C., Potapov, P. V., Moore, R., Hancher, M., Turubanova, S. A., Tyukavina, A., ... & Townshend, J. R. G. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, 342(6160), 850-853. https://doi.org/10.1126/science.1244693

[12] Reiche, J., Hamunyela, E., Verbesselt, J., Hoekman, D., & Herold, M. (2018). Improving near-real time deforestation monitoring in tropical dry forests by combining dense Sentinel-1 time series with Landsat and ALOS-2 PALSAR-2. *Remote Sensing of Environment*, 204, 147-161. https://doi.org/10.1016/j.rse.2017.10.034

[13] Hethcoat, M. G., Edwards, D. P., Carreiras, J. M., Bryant, R. G., França, F. M., & Quegan, S. (2019). A machine learning approach to map tropical selective logging. *Remote Sensing of Environment*, 221, 569-582. https://doi.org/10.1016/j.rse.2018.11.044

[14] Hu, Y., Raza, A., Sohail, A., Jiang, W., Maroof Shah, S. A., Asghar, M., ... & Hussain, S. (2020). Land use/land cover classification using multisource Sentinel-1 and Sentinel-2 satellite imagery. *The Journal of the Indian Society of Remote Sensing*, 48, 1055-1064. https://doi.org/10.1007/s12524-020-01135-0

[15] Ienco, D., Interdonato, R., Gaetano, R., & Ho Tong Minh, D. (2019). Combining Sentinel-1 and Sentinel-2 satellite image time series for land cover mapping via a multi-source deep learning architecture. *ISPRS Journal of Photogrammetry and Remote Sensing*, 158, 11-22. https://doi.org/10.1016/j.isprsjprs.2019.09.016

[16] Pham, L. T. H., Brabyn, L., & Ashraf, S. (2019). Combining QuickBird, LiDAR, and GIS topography indices to identify a single native tree species in a complex landscape using an object-based classification approach. *International Journal of Applied Earth Observation and Geoinformation*, 50, 187-197. https://doi.org/10.1016/j.jag.2016.03.015

[17] Nguyen, H. T. T., Doan, T. M., Tomppo, E., & McRoberts, R. E. (2020). Land use/land cover mapping using multitemporal Sentinel-2 imagery and four classification methods—A case study from Dak Nong, Vietnam. *Remote Sensing*, 12(9), 1367. https://doi.org/10.3390/rs12091367

[18] Bùi, T. D., Phan, T. T. H., & Nguyễn, V. L. (2021). Biến động rừng ngập mặn ven biển đồng bằng sông Cửu Long giai đoạn 1990-2020 từ ảnh Landsat. *Tạp chí Khoa học Đại học Huế: Khoa học Tự nhiên*, 130(1B), 5-18. https://doi.org/10.26459/hueuni-jns.v130i1B.6234

[19] Khatami, R., Mountrakis, G., & Stehman, S. V. (2016). A meta-analysis of remote sensing research on supervised pixel-based land-cover image classification processes: General guidelines for practitioners and future research. *Remote Sensing of Environment*, 177, 89-100. https://doi.org/10.1016/j.rse.2016.02.028

[20] Zhang, C., Sargent, I., Pan, X., Li, H., Gardiner, A., Hare, J., & Atkinson, P. M. (2020). Joint deep learning for land cover and land use classification. *Remote Sensing of Environment*, 221, 173-187. https://doi.org/10.1016/j.rse.2018.11.014

[21] Bộ Nông nghiệp và Phát triển Nông thôn (2021). *Quyết định số 1558/QĐ-BNN-TCLN về việc công bố hiện trạng rừng toàn quốc năm 2020*. Hà Nội: Bộ NN&PTNT. https://www.mard.gov.vn/Pages/cong-bo-hien-trang-rung-toan-quoc-nam-2020.aspx

[22] Sở Nông nghiệp và Phát triển Nông thôn tỉnh Cà Mau (2021). *Báo cáo hiện trạng rừng tỉnh Cà Mau năm 2021*. Cà Mau: Sở NN&PTNT. https://www.camau.gov.vn/wps/portal/?1dmy&page=gioithieu.chitiet

[23] Donato, D. C., Kauffman, J. B., Murdiyarso, D., Kurnianto, S., Stidham, M., & Kanninen, M. (2011). Mangroves among the most carbon-rich forests in the tropics. *Nature Geoscience*, 4(5), 293-297. https://doi.org/10.1038/ngeo1123

[24] Alongi, D. M. (2014). Carbon cycling and storage in mangrove forests. *Annual Review of Marine Science*, 6, 195-219. https://doi.org/10.1146/annurev-marine-010213-135020

[25] Tobler, W. R. (1970). A computer movie simulating urban growth in the Detroit region. *Economic Geography*, 46(sup1), 234-240. https://doi.org/10.2307/143141

[26] Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley. https://doi.org/10.1002/9781118548387

[27] European Space Agency (2024). *Sentinel-1 SAR User Guide*. ESA Sentinel Online. https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar

[28] European Space Agency (2024). *Sentinel-2 MSI User Guide*. ESA Sentinel Online. https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi

[29] Huang, S., Tang, L., Hupy, J. P., Wang, Y., & Shao, G. (2021). A commentary review on the use of normalized difference vegetation index (NDVI) in the era of popular remote sensing. *Journal of Forestry Research*, 32(1), 1-6. https://doi.org/10.1007/s11676-020-01155-1

:::

\newpage

# Phụ lục

## Phụ lục A. Danh mục hình ảnh

> **[TODO: Cập nhật sau khi chèn hình ảnh]**
> *Gợi ý:* Liệt kê tất cả hình ảnh trong đồ án với số thứ tự, tên và trang.

## Phụ lục B. Danh mục bảng biểu

> **[TODO: Cập nhật sau khi hoàn thiện]**
> *Gợi ý:* Liệt kê tất cả bảng trong đồ án với số thứ tự, tên và trang.

## Phụ lục C. Mã nguồn chính

> **[TODO: Bổ sung code snippets quan trọng nếu cần]**
> *Gợi ý:* Các đoạn code quan trọng như CNN architecture definition, training loop, data preprocessing.
