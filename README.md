# Phân loại 2 giống gạo Cammeo & Osmancik (Rice classification)

Đồ án kết thúc học phần — Học máy cơ bản, Bài 6: Cây quyết định (Decision Tree), cắt tỉa (Pruning), Rừng ngẫu nhiên (Random Forest).

## Kết quả chính (Key results, số thật từ lần chạy seed = 42)

- Dữ liệu (Data): UCI Rice, 3.810 mẫu — Osmancik 2.180 (57,2%), Cammeo 1.630 (42,8%). Giấy phép CC BY 4.0.
- Trích dẫn (Citation): Cinar & Koklu, UCI ML Repository, DOI 10.24432/C5MW4Z.
- Validation (571 mẫu): baseline F1 = 0,7283; cây cắt tỉa (ccp_alpha = 0,00879, 3 lá) F1 = 0,9289; rừng 200 cây F1 = 0,9355.
- Kiểm thử độc lập (Test, 572 mẫu, đánh giá 1 lần): Accuracy = 0,9178; F1 = 0,9295; ROC-AUC = 0,9769.
- Đặc trưng quan trọng nhất (Top feature): MajorAxis (Trục dài) 30,55%.

## Chạy lại từ máy mới (Reproduce, 3 lệnh)

```bash
/home/siinquyet/.local/bin/uv venv --python 3.12 .venv
/home/siinquyet/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python src/get_data.py && .venv/bin/python src/clean.py
.venv/bin/python -c "import pandas as pd; from src.split import stratified_split; df=pd.read_csv('data/processed/rice_clean.csv'); tr,va,te=stratified_split(df); [d.to_csv(f'data/processed/{n}.csv',index=False) for d,n in [(tr,'train'),(va,'val'),(te,'test')]]"
.venv/bin/python src/train.py && .venv/bin/python src/evaluate.py
.venv/bin/python -m pytest tests/ -v
.venv/bin/python app.py  # mở http://127.0.0.1:5000
```

File dữ liệu CSV không commit vào Git (tải bằng `src/get_data.py`, checksum in ra màn hình).

## Cấu trúc (Layout)

- `src/`: get_data, clean, split, train, evaluate (mỗi file 1 nhiệm vụ).
- `app.py` + `templates/`: 3 màn hình (Giới thiệu, Phân loại, Dashboard) + `POST /api/rice-classify`.
- `models/pipeline.pkl`: pipeline đã lưu (chỉ load khi serving, không train lại).
- `reports/`: metrics.json, test_report.json, figures/.
- `slides/answer.pptx`: 11 slide thuyết trình (số thật 100%).
- `demo_sample.csv`: 20 mẫu demo an toàn.

## Demo 5–7 phút chỉ bằng chuột (Mouse-only walkthrough)

1. Mở `/` — đọc phạm vi và giới hạn (1 phút).
2. Mở `/classify` — kéo 7 thanh trượt, bấm Dự đoán (Predict), đọc giống + xác suất + quy tắc (2 phút).
3. Bấm tải `demo_sample.csv` lên form batch, xem bảng 20 kết quả (1 phút).
4. Mở `/dashboard` — chỉ các số Accuracy/F1/ma trận (2 phút).
5. Cố tình nhập Area = -5 → xem thông báo lỗi tiếng Việt, app không sập.

## Phân công 2 thành viên (Team of 2)

- Thành viên 1: pipeline dữ liệu + train/evaluate (Tasks 1–4) + slide 1–6.
- Thành viên 2: app Flask + API (Task 5) + slide 7–11 + README.
- Cả hai cùng hiểu toàn bộ pipeline, cùng trả lời được về leakage, metric, mô hình và API.
