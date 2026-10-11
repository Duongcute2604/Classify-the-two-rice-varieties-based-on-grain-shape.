# EDA trên tập train (ghi chú 11/10/2026)

- Chỉ khám phá trên `train` (70%, stratified, seed=42); không xem `test`.
- Phân bố lớp giữ nguyên ~57,2% Osmancik / 42,8% Cammeo ở cả 3 split (lệch < 3%).
- Không có giá trị khuyết; bản ghi trùng lặp tuyệt đối đã loại (xem `src/clean.py`).
- Ngoại lệ IQR chỉ báo cáo, không loại dòng bằng nhãn (tránh leakage).
- Quyết định: không scaler (cây không cần), giữ nguyên 7 đặc trưng hình học cho baseline depth=1.
