import cv2
import numpy as np
import math


def calculate_A(mask):
    """Tính chữ A (Asymmetry) có tích hợp xoay trục chính (PCA/Ellipse)"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    cnt = max(contours, key=cv2.contourArea)

    
    if len(cnt) >= 5:  
        (cX, cY), (MA, ma), angle = cv2.fitEllipse(cnt)
    else:
        return 0

    
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

    
    x, y, w, h = cv2.boundingRect(rotated_mask)
    roi = rotated_mask[y:y + h, x:x + w]

    
    flipped_h = cv2.flip(roi, 1)
    flipped_v = cv2.flip(roi, 0)

    xor_h = cv2.bitwise_xor(roi, flipped_h)
    xor_v = cv2.bitwise_xor(roi, flipped_v)

    total_area = np.count_nonzero(roi)
    if total_area == 0: return 0

    
    ratio_h = (np.count_nonzero(xor_h) / 2.0) / total_area
    ratio_v = (np.count_nonzero(xor_v) / 2.0) / total_area

    
    score = 0
    if ratio_h > 0.05: score += 1
    if ratio_v > 0.05: score += 1
    return score


def calculate_B(mask):
    """Tính chữ B (Border) dùng Compactness Index"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    cnt = max(contours, key=cv2.contourArea)
    A = cv2.contourArea(cnt)
    P = cv2.arcLength(cnt, True)
    if P == 0: return 0

    
    compactness = (4 * math.pi * A) / (P ** 2)

    
    
    score = int(round(20 * (1 - compactness)))
    return min(max(score, 0), 8)  


def calculate_C(image, mask):
    """Tính chữ C (Color Variegation) trong không gian LAB có kiểm soát ngưỡng"""
    rgb_colors = [
        [255, 255, 255], [255, 0, 0], [205, 133, 63],
        [101, 67, 33], [119, 136, 153], [0, 0, 0]
    ]
    bgr_colors = [[c[2], c[1], c[0]] for c in rgb_colors]
    bgr_array = np.uint8([bgr_colors])
    lab_colors = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2LAB)[0].astype(np.float32)

    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = img_lab[mask > 127].astype(np.float32)
    if len(pixels) == 0: return 1

    distances = np.zeros((len(pixels), 6), dtype=np.float32)
    for i in range(6):
        distances[:, i] = np.sqrt(np.sum((pixels - lab_colors[i]) ** 2, axis=1))

    closest_color_idx = np.argmin(distances, axis=1)

    score = 0
    total_pixels = len(pixels)
    for i in range(6):
        
        is_closest = (closest_color_idx == i)
        is_valid_distance = (distances[:, i] < 90.0)

        color_count = np.sum(is_closest & is_valid_distance)
        if (color_count / total_pixels) > 0.02:  
            score += 1

    return max(1, min(score, 6))


def calculate_D(mask, pixel_per_mm=50.0):
    """Tính D và trả về cụ thể ĐƯỜNG KÍNH THỰC (mm) và ĐIỂM SỐ (0-5)"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0, 0.0
    cnt = max(contours, key=cv2.contourArea)
    (_, _), radius = cv2.minEnclosingCircle(cnt)

    diameter_mm = (radius * 2) / pixel_per_mm
    score = 5.0 if diameter_mm > 6.0 else (diameter_mm / 6.0) * 5.0
    return diameter_mm, score


def evaluate_abcd_rule(image, snakes_mask, kmeans_mask, pixel_per_mm=50.0):
    """Hàm tổng hợp kết quả (Đã tách riêng việc dùng mask nào cho đặc trưng nào)"""
    A = calculate_A(snakes_mask)
    
    B = calculate_B(kmeans_mask)
    C = calculate_C(image, snakes_mask)
    diameter_mm, score_D = calculate_D(snakes_mask, pixel_per_mm)

    tds = (A * 1.3) + (B * 0.1) + (C * 0.5) + (score_D * 0.5)

    if tds < 4.75:
        diagnosis = "Benign"
    elif 4.75 <= tds <= 5.45:
        diagnosis = "Suspicious"
    else:
        diagnosis = "Malignant Melanoma"

    return {
        "A": int(A), "B": int(B), "C": int(C),
        "D_mm": round(diameter_mm, 2),  
        "D_score": round(score_D, 2),  
        "TDS": round(tds, 2),
        "Diagnosis": diagnosis
    }