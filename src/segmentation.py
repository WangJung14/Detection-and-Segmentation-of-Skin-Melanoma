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
    Sử dụng Spatial Prior (Ưu tiên Không gian) thay vì Độ sáng:
    Khối u luôn nằm ở trung tâm bức ảnh. Ta sẽ lấy các cụm gần tâm nhất.
    Bao gồm màng lọc "Đảo lớn nhất" để loại bỏ các nhiễu rác xung quanh.
    """
    h, w = color_image.shape[:2]
    # 1. Chuyển ảnh thành mảng 2D
    pixel_values = color_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # 2. Thiết lập tiêu chí dừng cho K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # 3. Chạy thuật toán K-Means
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()

    # 4. Tối ưu Không gian (Spatial Prior): Tính khoảng cách trung bình từ từng cụm đến tâm ảnh
    center_x, center_y = w // 2, h // 2
    Y, X = np.indices((h, w))
    X = X.flatten()
    Y = Y.flatten()
    
    cluster_dist = []
    for i in range(k):
        idx = (labels == i)
        # Loại bỏ cụm màu trắng toát (do circular mask 4 góc tạo ra)
        if np.mean(centers[i]) > 240:
            cluster_dist.append(float('inf'))
            continue
            
        if np.sum(idx) == 0:
            cluster_dist.append(float('inf'))
            continue
            
        # Tính khoảng cách Euclidean trung bình từ các pixel cụm i tới tâm
        dist = np.sqrt((X[idx] - center_x)**2 + (Y[idx] - center_y)**2)
        cluster_dist.append(np.mean(dist))

    # 5. Tạo mặt nạ (Mask) ban đầu
    mask = np.zeros_like(labels, dtype=np.uint8)

    # Lấy cụm lõi u (Gần tâm nhất)
    core_cluster = np.argmin(cluster_dist)
    mask[labels == core_cluster] = 255

    # Lấy thêm cụm thứ 2 (Viền mờ nhạt dần ra xung quanh) nếu K >= 3
    if k >= 3:
        cluster_dist[core_cluster] = float('inf') # Loại lõi ra khỏi danh sách
        margin_cluster = np.argmin(cluster_dist)
        mask[labels == margin_cluster] = 255

    # Đưa mask về lại kích thước ảnh ban đầu
    mask = mask.reshape((h, w))

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