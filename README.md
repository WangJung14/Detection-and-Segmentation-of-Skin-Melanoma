# Nhận Diện Và Phân Loại Ung Thư Da (Melanoma)

Dự án xây dựng một hệ thống phân tích hình ảnh y tế tự động nhằm hỗ trợ chẩn đoán bệnh ung thư da hắc tố (Melanoma). Dự án triển khai song song nhiều phương pháp từ Xử lý ảnh truyền thống (Traditional CV) đến Học máy (Machine Learning) và Học sâu (Deep Learning) để đối sánh (Baseline Comparison) và đánh giá hiệu quả, đặc biệt chú trọng vào tính giải thích (Explainable AI - XAI).

**Môn học:** Xử lý ảnh và thị giác máy tính

---

## 1. Tổng Quan Dự Án (Project Overview)

Mục tiêu chính của dự án là tự động hóa quá trình phân tích hình ảnh y khoa để phát hiện và phân loại khối u hắc tố. Hệ thống sử dụng các cách tiếp cận:
1. **Phương pháp Xử lý ảnh truyền thống:** Chậm hơn nhưng mang tính giải thích cao, bám sát các tiêu chuẩn y khoa và cho phép theo dõi, giải thích từng bước biến đổi của hình ảnh (XAI).
2. **Học máy (Machine Learning - SVM):** Tự động học trọng số từ các đặc trưng y khoa (ABCD Rule) thay vì dùng công thức cố định của bác sĩ.
3. **Học sâu (Deep Learning - U-Net):** Nhanh (Real-time), tự động hóa quá trình phân đoạn ở cấp độ pixel, hướng tới độ chính xác cao.

## 2. Ngôn Ngữ & Công Cụ (Tech Stack)

*   **Ngôn ngữ:** Python 3.x
*   **Thư viện CV Truyền thống:** OpenCV (`cv2`), NumPy, Matplotlib, Scikit-image.
*   **Thư viện Học máy:** Scikit-learn, Pandas, Seaborn.
*   **Thư viện Deep Learning:** PyTorch (`torch`), Albumentations.
*   **Giao diện Trình diễn:** Jupyter Notebook.
*   **Dữ liệu:** ISIC Archive (2017, 2018).

## 3. Kiến Trúc Hệ Thống (Pipeline)

Hệ thống được thiết kế theo quy trình chặt chẽ nhằm tối ưu hóa việc trích xuất và phân loại:

### Giai đoạn 1: Tiền Xử Lý (Preprocessing)
*   **Xóa nhiễu ống kính & lọc lông trên da:** Kỹ thuật Circular Masking, thuật toán DullRazor kết hợp Inpainting (Telea).
*   **Cân bằng sáng:** Sử dụng CLAHE cục bộ trên không gian màu LAB.

### Giai đoạn 2: Phân Đoạn (Segmentation)
*   **Traditional CV:** Kết hợp Spatial K-Means ($K=4$) để tạo mặt nạ thô và Morphological Snakes tiến hóa đường viền để bo sát rãnh u.
*   **Majority Voting (Ensemble):** Bầu chọn đa số để kết hợp kết quả từ nhiều mặt nạ, giảm False Positive.
*   **Deep Learning:** Sử dụng mạng U-Net kết hợp Data Augmentation mạnh mẽ (Albumentations) để tự động phân đoạn.

### Giai đoạn 3: Trích Xuất Đặc Trưng Y Khoa (Clinical Feature Extraction)
Số hóa hệ thống chuẩn đoán y khoa **ABCD Rule**:
*   **A - Asymmetry (Bất đối xứng):** Trục đối xứng PCA.
*   **B - Border (Đường viền):** Độ nham nhở Compactness.
*   **C - Color (Màu sắc):** Khoảng cách Euclidean không gian LAB.
*   **D - Diameter (Đường kính):** Quy đổi kích thước mm.

### Giai đoạn 4: Phân Loại Lành/Ác (Classification)
*   **Quy tắc Y khoa (TDS Baseline):** Chẩn đoán dựa trên điểm số TDS truyền thống của bác sĩ ($TDS = 1.3A + 0.1B + 0.5C + 0.5D$).
*   **Mô hình AI (Linear SVM):** Mô hình SVM học từ tập dữ liệu trích xuất, có khả năng giải thích (XAI) thông qua việc tự động điều chỉnh trọng số của từng đặc trưng ABCD. Quá trình train tự động lấy Ground Truth từ API của ISIC và xử lý mất cân bằng dữ liệu (`class_weight='balanced'`).

## 4. Cấu Trúc Thư Mục (Folder Structure)

```text
├── data/                       # Chứa tập dữ liệu ISIC và các file kết quả (.csv, .png)
├── notebooks/                  # Bộ Live Demo Jupyter Notebook
├── src/                        # Chứa toàn bộ mã nguồn xử lý chính
│   ├── preprocessing.py        # Tiền xử lý (DullRazor, CLAHE, ...)
│   ├── segmentation.py         # K-Means, Snakes
│   ├── ensemble.py             # Majority Voting
│   ├── features.py             # Tính toán đặc trưng ABCD Rule & TDS
│   ├── extract_features.py     # Script trích xuất hàng loạt đặc trưng
│   ├── fetch_isic_metadata_api.py  # Cào tự động nhãn (Label) từ ISIC API
│   ├── prepare_dataset.py      # Gộp (Merge) đặc trưng và nhãn thành Dataset
│   ├── train_and_evaluate.py   # Huấn luyện SVM và đánh giá so với TDS
│   ├── plot_metrics_comparison.py  # Vẽ biểu đồ so sánh trực quan
│   ├── evaluation.py           # Đánh giá kết quả (IoU, Dice)
│   ├── parameter_sweep.py      # Bộ engine vẽ lưới đồ thị khảo sát tham số
│   └── u_net/                  # Mô-đun Deep Learning (U-Net) & Dataset
├── ablation_study.py           # File chạy so sánh hiệu năng nhanh
├── main.py                     # Entry point chính của chương trình
└── requirements.txt            # Danh sách thư viện phụ thuộc
```

## 5. Hướng Dẫn Cài Đặt & Chạy Source Code (Setup & Run)

### Cài đặt môi trường
1. Clone dự án về máy tính.
2. Cài đặt các thư viện phụ thuộc: 
   ```bash
   pip install -r requirements.txt
   pip install isic-cli requests pandas scikit-learn seaborn
   ```

### Chạy hệ thống Phân loại (SVM vs TDS)
Để minh họa quá trình hoạt động của Giai đoạn 4, hãy chạy tuần tự các lệnh sau trong thư mục gốc của project:

1. **(Tùy chọn) Lấy dữ liệu nhãn từ máy chủ ISIC:**
   ```bash
   python src/fetch_isic_metadata_api.py --features_csv "data/abcd_features_train.csv" --output_csv "data/ISIC_2018_Task1_Metadata.csv"
   ```
2. **Gộp đặc trưng vào Dataset:**
   ```bash
   python src/prepare_dataset.py --features_csv "data/abcd_features_train.csv" --metadata_csv "data/ISIC_2018_Task1_Metadata.csv" --output_csv "data/final_dataset_svm_train.csv"
   ```
3. **Huấn luyện mô hình và Đánh giá (Output ra độ chính xác & biểu đồ trọng số):**
   ```bash
   python src/train_and_evaluate.py --train_csv "data/final_dataset_svm_train.csv" --test_csv "data/final_dataset_svm_val.csv" --output_plot "data/SVM_vs_TDS_Comparison.png"
   ```
4. **Vẽ biểu đồ Confusion Matrix & So sánh hiệu suất tổng thể:**
   ```bash
   python src/plot_metrics_comparison.py
   ```
   *Kết quả biểu đồ sẽ được tự động lưu vào thư mục `data/`.*

### Khảo sát trực quan bằng Notebooks
Mở thư mục `notebooks` bằng VS Code hoặc Jupyter Lab và chạy tuỳ chọn **Run All** ở các file `.ipynb` (ví dụ `01_pipeline_demo.ipynb` hay `04_DeepLearning_UNet.ipynb`) để xem trực quan từng bước biến đổi ảnh y khoa.
