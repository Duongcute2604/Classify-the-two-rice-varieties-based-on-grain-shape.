# Model card — Rice classifier (seed=42)

- Bài toán: phân loại nhị phân 1 hạt gạo → Cammeo / Osmancik + xác suất + quy tắc cây.
- Mô hình thắng validation: RandomForest 200 cây (val F1=0,9355) vs cây cắt tỉa ccp_alpha=0,00879 (3 lá, val F1=0,9289) vs baseline F1=0,7283.
- Test độc lập 1 lần (572 mẫu): Accuracy=0,9178; F1=0,9295; ROC-AUC=0,9769.
- Top feature: MajorAxis 30,55%.
- Giới hạn: chỉ dùng khi đầu vào cùng quy trình trích xuất ảnh; p trong [0.45,0.55] → cảnh báo uncertain; ngoài miền train → cảnh báo out_of_domain. Không thay thế kiểm nghiệm giống chính thức.
- Serving: app chỉ load `models/pipeline.pkl`, không train lại mỗi request.
