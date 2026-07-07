import cv2
import numpy as np

# Đọc ảnh gốc (nhớ đổi đường dẫn nhé)
img = cv2.imread("../data/toy_data/melanoma/ISIC_0000074.jpg")
# Resize nhỏ lại để cửa sổ không bị tràn màn hình
img = cv2.resize(img, (600, 450))


def nothing(x):
    pass  # Hàm rỗng bắt buộc phải có cho trackbar của OpenCV


# 1. Tạo một cửa sổ giao diện
cv2.namedWindow('Tuning Station')

# 2. Tạo các thanh trượt (Tên thanh trượt, Tên cửa sổ, Giá trị mặc định, Giá trị Max, Hàm gọi lại)
cv2.createTrackbar('Kernel Size', 'Tuning Station', 15, 50, nothing)
cv2.createTrackbar('Inpaint Rad', 'Tuning Station', 3, 20, nothing)
cv2.createTrackbar('CLAHE Limit (x10)', 'Tuning Station', 15, 50, nothing)  # Chia 10 để lấy số thập phân

while True:
    # 3. Lấy giá trị hiện tại từ thanh trượt
    k_size = cv2.getTrackbarPos('Kernel Size', 'Tuning Station')
    inp_rad = cv2.getTrackbarPos('Inpaint Rad', 'Tuning Station')
    clahe_val = cv2.getTrackbarPos('CLAHE Limit (x10)', 'Tuning Station') / 10.0

    # Ép kernel_size luôn là số lẻ và lớn hơn 1
    if k_size % 2 == 0: k_size += 1
    if k_size < 3: k_size = 3
    if inp_rad < 1: inp_rad = 1

    # --- ĐOẠN NÀY LÀ LOGIC RÚT GỌN ĐỂ CHẠY REAL-TIME CHO NHANH ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, thresh_pre = cv2.threshold(blackhat, 10, 255, cv2.THRESH_TOZERO)
    _, mask = cv2.threshold(thresh_pre, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    clean = cv2.inpaint(img, mask, inpaintRadius=inp_rad, flags=cv2.INPAINT_TELEA)

    lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_val, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    final_res = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    # -------------------------------------------------------------

    # 4. Hiển thị ảnh ghép (Ảnh gốc và Ảnh đang chỉnh)
    combined = np.hstack((img, final_res))
    cv2.imshow('Tuning Station', combined)

    # Nhấn 'q' hoặc 'ESC' để thoát vòng lặp
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print(f"THÔNG SỐ CHỐT:\nKernel: {k_size}\nInpaint: {inp_rad}\nCLAHE: {clahe_val}")
        break

cv2.destroyAllWindows()