import cv2
import numpy as np
from skimage.segmentation import morphological_chan_vese

def get_otsu_mask(gray_image):
    """
    Tạo mặt nạ thô bằng thuật toán Otsu và làm sạch bằng hình thái học.
    Đây sẽ là điểm xuất phát (Initial Level Set) cho thuật toán Snakes.
    """
    # 1. Chạy Otsu
    # (Ảnh L-Channel thường có khối u màu Đen trên nền Trắng, nên ta dùng THRESH_BINARY_INV để đảo ngược: U thành Trắng, Nền thành Đen)
    _, mask_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Dọn rác (Morphological Opening & Closing)
    # Xóa các chấm trắng li ti bên ngoài khối u
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel_open)

    # Lấp các lỗ đen nhỏ bên trong khối u
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_close)

    return mask_cleaned


def get_snakes_mask(gray_image, init_mask, num_iter=35):
    image_float = gray_image.astype(np.float32) / 255.0
    mask_float = (init_mask > 0).astype(np.float32)

    # Chạy thuật toán
    snake_evolution = morphological_chan_vese(
        image_float,
        num_iter=num_iter,
        init_level_set=mask_float,
        smoothing=1,
        lambda1=1,
        lambda2=1
    )

    # --- ĐOẠN FIX LỖI Ở ĐÂY ---
    # 1. Chuyển kết quả sang mảng float64 trước
    res = snake_evolution.astype(np.float64)
    # 2. Nhân với 255
    res = res * 255
    # 3. Ép kiểu về uint8 một cách tường minh
    final_mask = res.astype(np.uint8)

    return final_mask


def get_kmeans_mask(color_image, k=3):
    """
    Sử dụng Machine Learning (K-Means Clustering) để gom nhóm màu sắc.
    Bao gồm màng lọc "Đảo lớn nhất" để loại bỏ các nhiễu rác xung quanh.
    """
    # 1. Chuyển ảnh thành mảng 2D
    pixel_values = color_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # 2. Thiết lập tiêu chí dừng cho K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # 3. Chạy thuật toán K-Means
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 4. centers chứa màu trung bình của K cụm. Ta tính độ sáng của từng cụm.
    brightness = np.sum(centers, axis=1)

    # Sắp xếp index các cụm từ Tối nhất -> Sáng nhất
    sorted_indices = np.argsort(brightness)

    # 5. Tạo mặt nạ (Mask) ban đầu
    labels = labels.flatten()
    mask = np.zeros_like(labels, dtype=np.uint8)

    # Lấy cụm tối nhất (Lõi khối u)
    mask[labels == sorted_indices[0]] = 255

    # Lấy thêm cụm tối thứ 2 (Viền mờ nhạt màu)
    if k >= 3:
        mask[labels == sorted_indices[1]] = 255

    # Đưa mask về lại kích thước ảnh ban đầu
    mask = mask.reshape(color_image.shape[:2])

    # ---------------------------------------------------------
    # NÂNG CẤP LÕI: THỦ THUẬT LỌC "ĐẢO LỚN NHẤT"
    # ---------------------------------------------------------
    # Quét tìm tất cả các cụm màu trắng (contours)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tạo một mask mới hoàn toàn đen
    clean_mask = np.zeros_like(mask)

    if contours:
        # Tìm contour có diện tích to nhất (chắc chắn là khối u chính)
        largest_contour = max(contours, key=cv2.contourArea)

        # Chỉ vẽ duy nhất contour này lên cái mask đen vừa tạo
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    # ---------------------------------------------------------

    # Dọn rác hình thái học nhẹ nhàng để lấp các lỗ thủng nhỏ bên trong khối u
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    return final_mask