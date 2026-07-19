import cv2
import numpy as np


def remove_hair(image, kernel_size=15, inpaint_rad=3):
    """
    Thuật toán DullRazor v3.1:
    Xử lý êm ái cho ảnh dày đặc lông, tránh phá nát kết cấu khối u.
    """
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    
    
    
    _, thresh_pre = cv2.threshold(blackhat, 10, 255, cv2.THRESH_TOZERO)
    _, hair_mask = cv2.threshold(thresh_pre, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_mask_dilated = cv2.dilate(hair_mask, kernel_dilate, iterations=1)

    
    
    
    
    clean_image = cv2.inpaint(image, hair_mask_dilated, inpaintRadius=inpaint_rad, flags=cv2.INPAINT_TELEA)

    return clean_image, hair_mask_dilated


def enhance_contrast_clahe(image , clip_limit=1.2 , grid_size=(8,8)):
    """
        Cân bằng sáng cục bộ bằng CLAHE trên không gian màu L*a*b*.

        Tham số:
        - clip_limit: Ngưỡng cắt (Clip Limit). Càng cao thì độ tương phản càng mạnh,
                      nhưng dễ làm nhiễu da bị phóng đại. (Chuẩn y tế thường dùng 2.0 - 3.0)
        - grid_size: Kích thước ô lưới cắt ảnh ra để xử lý cục bộ (8x8 là chuẩn mực).
        """

    
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    
    
    l, a, b = cv2.split(lab_image)

    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    
    
    cl = clahe.apply(l)

    
    merged_lab = cv2.merge((cl, a, b))

    
    final_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    
    
    
    
    return final_image, cl


def boost_faint_edges(l_channel):
    """
    Tăng cường sắc độ viền mờ (Thuật toán phát triển vùng đệm).
    Tìm lõi khối u, khoanh vùng viền mờ và tăng độ đậm cho vùng viền đó
    nhằm tạo "bức tường" chặn thuật toán Snakes.
    """
    
    _, core_mask = cv2.threshold(l_channel, 70, 255, cv2.THRESH_BINARY_INV)

    
    _, broad_mask = cv2.threshold(l_channel, 120, 255, cv2.THRESH_BINARY_INV)

    
    faint_zone = cv2.bitwise_xor(broad_mask, core_mask)

    
    boosted_l_channel = l_channel.copy()

    
    darkening_value = np.zeros_like(l_channel)
    darkening_value[faint_zone == 255] = 40

    
    boosted_l_channel = cv2.subtract(boosted_l_channel, darkening_value)

    
    boosted_l_channel = cv2.GaussianBlur(boosted_l_channel, (3, 3), 0)

    return boosted_l_channel, faint_zone


def apply_circular_mask(image, radius_reduction=0.9):
    """
    Loại bỏ hiệu ứng đen 4 góc (Vignetting) của ống kính máy soi da.
    Tẩy trắng các vùng bên ngoài vòng tròn trung tâm.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = int(min(h, w) / 2 * radius_reduction)

    
    circular_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circular_mask, center, radius, 255, -1)

    
    result = image.copy()
    if len(image.shape) == 3:  
        result[circular_mask == 0] = (255, 255, 255)
    else:  
        result[circular_mask == 0] = 255

    return result