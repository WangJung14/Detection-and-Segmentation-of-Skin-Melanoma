# Nhận Diện Và Phân Loại Ung Thư Da (Melanoma)

Dự án xây dựng một hệ thống phân tích hình ảnh y tế tự động nhằm hỗ trợ chẩn đoán bệnh ung thư da hắc tố (Melanoma). Dự án triển khai song song 2 phương pháp (Traditional Computer Vision và Deep Learning) để đối sánh (Baseline Comparison) và đánh giá hiệu quả, chú trọng vào tính giải thích (Explainable AI - XAI).

**Môn học:** Xử lý ảnh và thị giác máy tính

---

## 1. Tổng Quan Dự Án (Project Overview)

Mục tiêu chính của dự án là tự động hóa quá trình phân tích hình ảnh y khoa để phát hiện khối u hắc tố. Hệ thống sử dụng 2 cách tiếp cận khác nhau:
1. **Phương pháp Xử lý ảnh truyền thống (Traditional Computer Vision):** Chậm hơn nhưng mang tính giải thích cao, bám sát các tiêu chuẩn y khoa và cho phép theo dõi, giải thích từng bước biến đổi của hình ảnh (XAI).
2. **Phương pháp Học sâu (Deep Learning - U-Net):** Nhanh (Real-time), tự động hóa quá trình phân đoạn ở cấp độ pixel, hướng tới độ chính xác cao nhờ học các đặc trưng phức tạp, nhưng đóng vai trò như một Hộp đen (Black-box).

## 2. Ngôn Ngữ & Công Cụ (Tech Stack)

*   **Ngôn ngữ:** Python 3.x
*   **Thư viện CV Truyền thống:** OpenCV (`cv2`), NumPy, Matplotlib, Scikit-image (Morphological Snakes).
*   **Thư viện Deep Learning:** PyTorch (`torch`, `torchvision`), PIL.
*   **Giao diện Trình diễn:** Jupyter Notebook.
*   **Dữ liệu:** ISIC 2017

## 3. Kiến Trúc Hệ Thống Truyền Thống (Traditional CV Pipeline)

Hệ thống được thiết kế theo quy trình 3 giai đoạn chặt chẽ nhằm tối ưu hóa việc trích xuất đặc trưng khối u:

### Giai đoạn 1: Tiền Xử Lý (Preprocessing)
*   **Xóa nhiễu ống kính:** Kỹ thuật Circular Masking (giảm bán kính $0.85$) để loại bỏ 4 góc đen của kính hiển vi.
*   **Lọc lông trên da:** Thuật toán DullRazor kết hợp nội suy ảnh Inpainting (Telea) để xóa lông mà không làm hỏng vân da.
*   **Cân bằng sáng:** Sử dụng CLAHE cục bộ (clip_limit $1.1$) trên kênh L (Lightness) của không gian màu LAB để làm rõ viền khối u.

### Giai đoạn 2: Phân Đoạn (Segmentation)
*   **Spatial K-Means ($K=4$):** Bản nâng cấp của K-Means truyền thống. Thay vì phân cụm dựa trên độ sáng, thuật toán phân loại dựa vào **Khoảng cách tới tâm bức ảnh** (Centrality Prior), giúp bắt trọn vẹn Lõi u (đậm) và Viền u (nhạt dần) mà không bị nhầm lẫn với nhiễu phản quang.
*   **Đường viền động:** Thuật toán Morphological Snakes (lặp $35$ vòng) lấy output của K-Means làm mặt nạ khởi tạo, giúp bo sát các rãnh nham nhở của khối u.

### Giai đoạn 3: Trích Xuất Đặc Trưng Y Khoa (Clinical Feature Extraction)
Số hóa hệ thống chuẩn đoán y khoa **ABCD Rule** thành các con số toán học:
*   **A - Asymmetry (Bất đối xứng):** Dùng thuật toán PCA (Phân tích thành phần chính) và Ma trận xoay để gập khối u lại, đo lường diện tích dư thừa bằng toán tử XOR.
*   **B - Border (Đường viền):** Tính độ nham nhở thông qua công thức Compactness ($4\pi A / P^2$).
*   **C - Color (Màu sắc):** Đo lường khoảng cách Euclidean trong không gian 3D LAB để dò tìm 6 màu sắc nguy hiểm.
*   **D - Diameter (Đường kính):** Tính toán kích thước điểm ảnh sang Milimet thực tế.

## 4. Kiến Trúc Học Sâu Đối Chứng (Deep Learning Baseline)

Xây dựng mạng nơ-ron tích chập từ đầu để giải quyết bài toán Semantic Segmentation:
*   **Kiến trúc:** U-Net (Encoder - Bottleneck - Decoder) tích hợp Skip Connections để giữ nguyên vẹn chi tiết không gian ở cấp độ Pixel.
*   **Dataset & DataLoader:** Class custom để nạp ảnh gốc (RGB) và Mask (Grayscale nhị phân), resize về chuẩn $256 \times 256$.
*   **Training Pipeline:** Sử dụng loss kết hợp `BCE + Dice Loss`, Optimizer `AdamW`, và scheduler `ReduceLROnPlateau`.

## 5. Báo Cáo Thực Nghiệm & Live Demo (Notebooks)

Dự án sở hữu một hệ thống Notebook cực kỳ trực quan phục vụ cho việc thuyết trình và khảo sát tham số (Parameter Sweeping):
*   **`01_pipeline_demo.ipynb`:** Trình diễn End-to-End từ khâu làm sạch $\rightarrow$ Phân đoạn Spatial K-Means $\rightarrow$ Chấm điểm ABCD.
*   **`02_Segmentation_and_Tuning.ipynb`:** Đấu trường Ablation Study so sánh hiệu năng giữa Otsu vs K-Means vs Snakes.
*   **`03_ABCD_Rule_Explainability.ipynb`:** Phô diễn tính giải thích (XAI), trực quan hóa cách máy tính chấm điểm nham nhở và tìm trục bất đối xứng.
*   **`04_DeepLearning_UNet.ipynb`:** Soi trọng số `best_model.pth` và chạy Inference cực nhanh trên U-Net.
*   **`05_Final_Showdown.ipynb`:** Trận chiến tốc độ (FPS) và độ chính xác (IoU) giữa Xử lý ảnh truyền thống và Deep Learning.
*   **`06_Parameter_Sweep_Demo.ipynb`:** Khảo sát toàn diện mọi thông số (Cỡ chổi DullRazor, Ngưỡng CLAHE, K-Means, số vòng lặp Snakes và hệ số nhạy của đặc trưng ABCD).

## 6. Cấu Trúc Thư Mục (Folder Structure)

```text
├── data/                       # Chứa tập dữ liệu (ISIC 2017)
├── notebooks/                  # Bộ 6 file Live Demo Jupyter Notebook
├── src/                        # Chứa toàn bộ mã nguồn xử lý chính
│   ├── preprocessing.py        # Các hàm tiền xử lý (DullRazor, CLAHE, ...)
│   ├── segmentation.py         # Spatial K-Means, Morphological Snakes
│   ├── features.py             # Trích xuất đặc trưng ABCD Rule & TDS
│   ├── evaluation.py           # Đánh giá kết quả (IoU, Dice)
│   ├── parameter_sweep.py      # Bộ engine vẽ lưới đồ thị khảo sát tham số
│   └── u_net/                  # Mô-đun Deep Learning (U-Net)
├── ablation_study.py           # File chạy so sánh hiệu năng nhanh
├── main.py                     # Entry point chính của chương trình
└── requirements.txt            # Danh sách thư viện phụ thuộc
```

## 7. Hướng Dẫn Cài Đặt (Setup Instructions)

1. Mở terminal và clone dự án về.
2. Tạo môi trường ảo và cài đặt thư viện: `pip install -r requirements.txt`.
3. Khuyến nghị mở thư mục `notebooks` bằng VS Code hoặc Jupyter Lab và chạy tuỳ chọn **Run All** ở các file `.ipynb` để thưởng thức toàn bộ quá trình trực quan.
