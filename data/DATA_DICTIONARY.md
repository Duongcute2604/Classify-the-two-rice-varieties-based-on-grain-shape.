# Data dictionary

| Cột | Kiểu | Đơn vị | Vai trò | Có sẵn lúc dự đoán |
|---|---|---|---|---|
| Area | float | pixel^2 | feature | có |
| Perimeter | float | pixel | feature | có |
| MajorAxis | float | pixel | feature | có |
| MinorAxis | float | pixel | feature | có |
| Eccentricity | float | [0,1] | feature | có |
| ConvexArea | float | pixel^2 | feature | có |
| Extent | float | [0,1] | feature | có |
| Class | Cammeo/Osmancik | — | target | không (cần dự đoán) |

Thời điểm dự đoán: sau khi trích xuất đủ 7 đặc trưng hình học từ ảnh hạt gạo.
