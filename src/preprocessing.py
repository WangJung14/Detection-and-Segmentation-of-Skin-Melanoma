import cv2
import numpy as np


def remove_hair(image, kernel_size=15, inpaint_rad=3):
    """
    Thuật toán DullRazor v3.1:
    Xử lý êm ái cho ảnh dày đặc lông, tránh phá nát kết cấu khối u.
    """
    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Black-hat Morphology
    # GIẢM kernel_size xuống (vd: 15). Chổi nhỏ sẽ chỉ bắt sợi lông dài,
    # không ăn phạm vào các cục u to.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 3. Phân ngưỡng Otsu CÓ BẢO HIỂM
    # Otsu đôi khi bắt quá nhạy. Ta kết hợp THRESH_TOZERO để chặn các
    # nhiễu nhạt màu trước khi đưa cho Otsu quyết định.
    _, thresh_pre = cv2.threshold(blackhat, 10, 255, cv2.THRESH_TOZERO)
    _, hair_mask = cv2.threshold(thresh_pre, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Giãn nở RẤT NHẸ NHÀNG
    # Giảm chổi xuống 3x3 và chỉ quét 1 lần. Tránh làm các sợi lông dính chùm vào nhau.
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_mask_dilated = cv2.dilate(hair_mask, kernel_dilate, iterations=1)

    # 5. Inpainting: Quay xe về TELEA
    # Với lỗ hổng to và nhằng nhịt, thuật toán TELEA (nhân màu theo bán kính)
    # giữ lại vân da tự nhiên tốt hơn Navier-Stokes (thường tạo mảng màu bệt).
    # GIẢM inpaint_rad xuống 3 để không nội suy quá lố.
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

    # Step 1 : Chuyển không gian màu từ BGR sang L*a*b*
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Step 2 : Tách 3 kênh L, a , b ra thành 3 ma trận độc lập
    # l: ma trận độ sáng, a: ma trận màu Xanh-Đỏ, b: ma trận màu Xanh-Vàng
    l, a, b = cv2.split(lab_image)

    # Bước 3: Khởi tạo "Bộ máy" CLAHE với các tham số truyền vào
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # Bước 4: Áp dụng ma thuật CLAHE CHỈ LÊN KÊNH L (Độ sáng)
    # Lệnh này sẽ làm phẳng ánh sáng nền và đẩy khối u đen nổi bật lên
    cl = clahe.apply(l)

    # Bước 5: Lắp ráp lại thành bức ảnh hoàn chỉnh (Kênh L mới + Kênh a, b cũ)
    merged_lab = cv2.merge((cl, a, b))

    # Bước 6: Chuyển ngược lại về BGR để con người (và các hàm hiển thị) có thể nhìn được
    final_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # Lưu ý: Anh trả về 2 biến.
    # - final_image: Để em show lên màn hình xem màu sắc.
    # - cl (Kênh L đã xử lý): Kênh này có dạng ảnh xám (Grayscale), rất sắc nét,
    #   ta sẽ dùng chính kênh 'cl' này để đưa vào thuật toán phân đoạn (Otsu/Snakes) ở Giai đoạn 2.
    return final_image, cl


def boost_faint_edges(l_channel):
    """
    Tăng cường sắc độ viền mờ (Thuật toán phát triển vùng đệm).
    Tìm lõi khối u, khoanh vùng viền mờ và tăng độ đậm cho vùng viền đó
    nhằm tạo "bức tường" chặn thuật toán Snakes.
    """
    # Bước 1: Tìm vùng "Cực đại" (Chỉ bắt những pixel cực kỳ đen)
    _, core_mask = cv2.threshold(l_channel, 70, 255, cv2.THRESH_BINARY_INV)

    # Bước 2: Bắt luôn cả vùng "Mờ mờ" (Lấy lõi + viền mờ)
    _, broad_mask = cv2.threshold(l_channel, 120, 255, cv2.THRESH_BINARY_INV)

    # Bước 3: Tìm chính xác "Vùng đệm mờ mờ" (Dùng phép XOR)
    faint_zone = cv2.bitwise_xor(broad_mask, core_mask)

    # Bước 4: Nhuộm đen vùng đệm
    boosted_l_channel = l_channel.copy()

    # Dùng numpy array để chứa giá trị cần trừ đi (40 đơn vị độ sáng)
    darkening_value = np.zeros_like(l_channel)
    darkening_value[faint_zone == 255] = 40

    # Dùng cv2.subtract để tránh lỗi số âm (underflow)
    boosted_l_channel = cv2.subtract(boosted_l_channel, darkening_value)

    # Làm mượt nhẹ để vết nhuộm không bị vỡ hạt
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

    # Tạo mặt nạ hình tròn
    circular_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circular_mask, center, radius, 255, -1)

    # Ép 4 góc thành màu trắng (255) để K-Means không bao giờ nhầm là khối u
    result = image.copy()
    if len(image.shape) == 3:  # Nếu là ảnh màu
        result[circular_mask == 0] = (255, 255, 255)
    else:  # Nếu là ảnh xám
        result[circular_mask == 0] = 255

    return result