Tên đề tài: Ứng dụng viễn thám và học sâu trong giám sát biến động rừng tỉnh Cà Mau

Dữ liệu đã có:
    - 1285 điểm (Training_Points_CSV.csv: id, label, x, y): 
        + 650 điểm tại kỳ ảnh trước là rừng, kỳ ảnh sau vẫn còn rừng (không mất rừng)
        + 635 điểm tại kỳ ảnh trước là rừng, kỳ ảnh sau không còn rừng (mất rừng)
    - Sentinel-2 (7 bands: B4, B8, B11, B12, NDVI, NBR, NDMI) đã cắt theo ranh giới rừng của tỉnh Cà Mau:
        + Kỳ ảnh trước: 31/01/2024 (S2_2024_01_30.tif)
        + Kỳ ảnh sau: 28/02/2025 (S2_2025_02_28.tif)
    - Sentinel-1: (2 bands: VH và R (VV-VH nhưng có lẽ sẽ không dùng R nữa, chỉ dùng VH thôi)) đã cắt theo ranh giới rừng của tỉnh Cà Mau:
        +  Kỳ ảnh trước: 04/02/2024 thay thế cho 31/01/2024 (S1_2024_02_04_matched_S2_2024_01_30.tif)
        + Kỳ ảnh sau: 22/02/2025 thay thế cho 28/02/2025 (S1_2025_02_22_matched_S2_2025_02_28.tif)

Output đầu ra:
    - Bản đồ phân loại vùng mất rừng và không mất rừng toàn bộ khu vực (.tif)
    - Mô hình deep learning
    - Các metrics chi tiết để viết báo cáo
    - Và những thứ cần thiết khác phục vụ cho đồ án tốt nghiệp

Cấu hình máy tính Windows 10:
    - CPU Xeon E5 2678 v3
    - RAM 32GB DDR3
    - GPU Nvidia RTX A4000 16GB

Môi trường conda 'dang' đã thiết lập: environment.yml (requirements.txt)