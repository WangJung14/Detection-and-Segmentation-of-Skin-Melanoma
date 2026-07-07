import cv2
import numpy as np
import math


def calculate_asymmetry(mask):
    """
    Tính chữ A (Asymmetry) trong luật ABCD.
    Tịnh tiến khối u về tâm canvas, lật ảnh và tính toán sai số (1 - IoU).
    Trả về: Điểm bất đối xứng từ 0.0 (đối xứng) đến 1.0 (hoàn toàn méo mó).
    """
    # 1. Tìm trọng tâm (Centroid) của khối u bằng Moments
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return 0.0
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # 2. Tạo canvas đối xứng qua tâm để lật không bị mất góc
    h, w = mask.shape
    max_dist_x = max(cX, w - cX)
    max_dist_y = max(cY, h - cY)
    canvas_w = 2 * max_dist_x
    canvas_h = 2 * max_dist_y

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    offset_x = max_dist_x - cX
    offset_y = max_dist_y - cY
    canvas[offset_y:offset_y + h, offset_x:offset_x + w] = mask

    # 3. Lật ma trận canvas
    flipped_h = cv2.flip(canvas, 1)  # Lật qua trục đứng (Trái - Phải)
    flipped_v = cv2.flip(canvas, 0)  # Lật qua trục ngang (Trên - Dưới)

    # Hàm phụ tính IoU nội bộ
    def get_iou(m1, m2):
        inter = np.logical_and(m1 > 127, m2 > 127).sum()
        union = np.logical_or(m1 > 127, m2 > 127).sum()
        return inter / union if union > 0 else 1.0

    # Tính độ bất đối xứng (1 - IoU)
    asym_h = 1.0 - get_iou(canvas, flipped_h)
    asym_v = 1.0 - get_iou(canvas, flipped_v)

    # Trả về giá trị trung bình của 2 trục
    return (asym_h + asym_v) / 2.0


def calculate_border_irregularity(mask):
    """
    Tính chữ B (Border) bằng chỉ số Circularity (Độ tròn).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest_contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    if perimeter == 0:
        return 0.0

    circularity = (4 * math.pi * area) / (perimeter ** 2)
    return circularity


def calculate_color_variegation(color_image, mask):
    """
    Tính chữ C (Color) bằng Độ lệch chuẩn (Standard Deviation) của các kênh màu.
    Trả về: Độ biến thiên màu trung bình. Chỉ số càng cao, u càng loang lổ nhiều màu.
    """
    # Trích xuất các pixel nằm trong vùng mask trắng
    pixels = color_image[mask > 127]
    if len(pixels) == 0:
        return 0.0

    # Tính độ lệch chuẩn cho từng kênh màu B, G, R
    std_b = np.std(pixels[:, 0])
    std_g = np.std(pixels[:, 1])
    std_r = np.std(pixels[:, 2])

    # Trả về trung bình cộng độ lệch chuẩn của 3 kênh
    return (std_b + std_g + std_r) / 3.0


def calculate_diameter(mask, mm_per_pixel=0.02):
    """
    Tính chữ D (Diameter) bằng Vòng tròn ngoại tiếp nhỏ nhất.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest_contour = max(contours, key=cv2.contourArea)

    (_, _), radius = cv2.minEnclosingCircle(largest_contour)
    diameter_mm = (radius * 2) * mm_per_pixel
    return diameter_mm