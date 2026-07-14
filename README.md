# Nhận Diện Và Phân Loại Ung Thư Da (Melanoma)

Dự án xây dựng một hệ thống phân tích hình ảnh y tế tự động nhằm hỗ trợ chẩn đoán bệnh ung thư da hắc tố (Melanoma). Dự án triển khai song song 2 phương pháp (Traditional Computer Vision và Deep Learning) để đối sánh (Baseline Comparison) và đánh giá hiệu quả.

**Môn học:** Xử lý ảnh và thị giác máy tính

---

## 1. Tổng Quan Dự Án (Project Overview)

Mục tiêu chính của dự án là tự động hóa quá trình phân tích hình ảnh y khoa để phát hiện khối u hắc tố. Hệ thống sử dụng 2 cách tiếp cận khác nhau:
1. **Phương pháp Xử lý ảnh truyền thống (Traditional Computer Vision):** Mang tính giải thích cao (Explainable AI), bám sát các tiêu chuẩn y khoa và cho phép theo dõi từng bước biến đổi của hình ảnh.
2. **Phương pháp Học sâu (Deep Learning - U-Net):** Tự động hóa quá trình phân đoạn ở cấp độ pixel, hướng tới độ chính xác cao nhờ học các đặc trưng phức tạp.

## 2. Ngôn Ngữ & Công Cụ (Tech Stack)

*   **Ngôn ngữ:** Python 3.x
*   **Thư viện CV Truyền thống:** OpenCV (`cv2`), NumPy, Matplotlib, Scikit-image (Morphological Snakes).
*   **Thư viện Deep Learning:** PyTorch (`torch`, `torchvision`), PIL.
*   **Môi trường:** Virtual Environment (`venv`), GPU Training (CUDA).
*   **Dữ liệu:** ISIC 2017

## 3. Kiến Trúc Hệ Thống Truyền Thống (Traditional CV Pipeline)

Hệ thống được thiết kế theo quy trình 3 giai đoạn chặt chẽ nhằm tối ưu hóa việc trích xuất đặc trưng khối u:

### Giai đoạn 1: Tiền Xử Lý (Preprocessing)
*   **Xóa nhiễu ống kính:** Kỹ thuật Circular Masking (giảm bán kính $0.85$) để loại bỏ 4 góc đen của kính hiển vi.
*   **Lọc lông trên da:** Thuật toán DullRazor kết hợp nội suy ảnh (Inpainting) để xóa lông mà không làm hỏng vân da.
*   **Cân bằng sáng:** Sử dụng CLAHE cục bộ (clip_limit $1.1$) trên kênh L (Lightness) của không gian màu LAB để làm rõ viền khối u.

### Giai đoạn 2: Phân Đoạn (Segmentation)
*   **Phân cụm màu sắc:** Thuật toán K-Means Clustering ($K=4$) để phân tách 4 thành phần (lõi u, viền u, da nền, góc tối).
*   **Đường viền động:** Thuật toán Morphological Snakes (lặp $35$ vòng) lấy output của K-Means làm mặt nạ khởi tạo, giúp bo sát các rãnh nham nhở của khối u.
*   **Đánh giá:** Tính toán chỉ số IoU (Intersection over Union) so với Ground Truth (nếu có).

### Giai đoạn 3: Trích Xuất Đặc Trưng Y Khoa (Clinical Feature Extraction)
Số hóa hệ thống chuẩn đoán y khoa **ABCD Rule**:
*   **A - Asymmetry (Bất đối xứng):** Tính toán độ lệch tâm.
*   **B - Border (Đường viền):** Đo lường độ nham nhở của cạnh khối u do Snakes vẽ ra.
*   **C - Color (Màu sắc):** Đánh giá độ biến thiên màu sắc bên trong khối u.
*   **D - Diameter (Đường kính):** Tính toán kích thước điểm ảnh của khối u.

## 4. Kiến Trúc Học Sâu Đối Chứng (Deep Learning Baseline)

Xây dựng mạng nơ-ron tích chập từ đầu để giải quyết bài toán Semantic Segmentation:
*   **Kiến trúc:** U-Net (Encoder - Bottleneck - Decoder) tích hợp Skip Connections để giữ nguyên vẹn chi tiết không gian ở cấp độ Pixel.
*   **Dataset & DataLoader:** Class custom để nạp ảnh gốc (RGB) và Mask (Grayscale nhị phân), resize về chuẩn $256 \times 256$.
*   **Training Pipeline:** 
    *   **Hàm Loss:** Binary Cross Entropy with Logits (BCE).
    *   **Optimizer:** Adam ($learning\_rate = 1e-3$).
    *   **Chiến thuật:** Triển khai Proof of Concept trên Toy Dataset (18 ảnh cực khó) để xử lý bài toán Imbalanced Data, chứng minh pipeline hoạt động hoàn hảo trước khi scale lên bộ dữ liệu ISIC đầy đủ (6GB).

## 5. Báo Cáo Thực Nghiệm (Lab Report & Parameter Sweep)

Dự án có module tự động hóa việc khảo sát tham số nhằm minh chứng tính khoa học:
*   **Khảo sát K-Means:** So sánh kết quả phân đoạn khi $K \in \{2, 3, 4\}$.
*   **Khảo sát Snakes:** Đánh giá sự hội tụ của đường viền khi số vòng lặp $num\_iter \in \{5, 35, 100\}$.
*   **Trực quan hóa:** Hệ thống tự động xuất ảnh dạng lưới (grid stitching) để đưa vào file báo cáo, giúp dễ dàng so sánh bằng mắt thường.

## 6. Cấu Trúc Thư Mục (Folder Structure)

```text
├── data/                       # Chứa tập dữ liệu (ISIC Dataset / Toy Dataset)
│   ├── images/                 # Ảnh gốc
│   └── ground_truth/           # Masks (Ground Truth)
├── src/                        # Chứa toàn bộ mã nguồn xử lý chính
│   ├── preprocessing.py        # Các hàm tiền xử lý (DullRazor, CLAHE, ...)
│   ├── segmentation.py         # K-Means, Morphological Snakes
│   ├── dataset.py              # PyTorch Dataset & DataLoader
│   └── unet_model.py           # Định nghĩa kiến trúc U-Net
├── parameter_sweep.py          # Script khảo sát tham số tự động và xuất ảnh ghép
├── train_unet.py               # Script huấn luyện mô hình U-Net
├── test_unet.py                # Script chạy Inference mô hình học sâu và trực quan hóa
├── main.py                     # Entry point chính của chương trình
└── requirements.txt            # Danh sách thư viện phụ thuộc
```

## 7. Hướng Dẫn Cài Đặt (Setup Instructions)

1. Clone dự án về máy:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Tạo môi trường ảo và kích hoạt:
   ```bash
   python -m venv venv
   source venv/bin/activate  # (hoặc `venv\Scripts\activate` trên Windows)
   ```
3. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```
4. Chạy các script để thực nghiệm hoặc xem kết quả.
