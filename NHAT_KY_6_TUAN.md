# Nhật ký 6 tuần

## Tuần 1 (01–07/10): chốt đề + repo + dữ liệu thô
- Nam: khởi tạo repo, .gitignore, get_data.py (4h). Kết quả: tải 3.810 mẫu. Vấn đề: mạng UCI chậm → thử lại.
- Chiến: requirements.txt, test_data (3h). Kết quả: schema PASS.

## Tuần 2 (08–14/10): làm sạch + split + baseline
- Nam: clean.py, split.py stratified 70/15/15 seed=42 (5h). Kết quả: không leakage.
- Chiến: test_split, EDA trên train, baseline depth=1 (5h). Kết quả: baseline F1=0,7283.

## Tuần 3 (15–21/10): mô hình chính + test cuối
- Nam: max_depth curve, ccp_alpha CV, forest 200 cây, pipeline.pkl (8h). Val: tree 0,9289 / forest 0,9355.
- Chiến: evaluate 1 lần trên test + error bins (5h). Test: acc 0,9178, F1 0,9295, AUC 0,9769.

## Tuần 4 (22–28/10): web + API
- Nam: POST /api/rice-classify + validation (6h).
- Chiến: 3 templates + CSS + demo_sample.csv + test_api (7h). Input xấu không sập app.

## Tuần 5 (29/10–05/11): slide + README
- Cả hai: 11 slide số thật, README tái lập 3 lệnh, data dictionary, model card (6h mỗi người).

## Tuần 6 (06–12/11): rehearsal + release
- Nam: QA pytest toàn bộ, kiểm tra máy sạch (4h).
- Chiến: demo 5–7 phút bằng chuột, tag v1.0 (3h).
