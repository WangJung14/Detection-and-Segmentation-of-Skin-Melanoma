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


if __name__ == "__main__":
    # Đọc ảnh gốc (thay đường dẫn bằng ảnh thực tế của em)
    img = cv2.imread("../data/toy_data/melanoma/ISIC_0000074.jpg")

    if img is not None:
        # 1. Chạy Pipeline
        clean_img, hair_mask = remove_hair(img, kernel_size=15, inpaint_rad=3)
        clahe_color_img, clahe_gray_l_channel = enhance_contrast_clahe(clean_img, clip_limit=1.2)
        smooth_l_channel = cv2.GaussianBlur(clahe_gray_l_channel, (5, 5), 0)

        # 2. Xử lý đồng bộ số kênh màu (Channel synchronization)
        # Chuyển L-Channel (1 kênh) thành 3 kênh (ảo) để có thể ghép với các ảnh màu
        clahe_gray_3_channels = cv2.cvtColor(smooth_l_channel, cv2.COLOR_GRAY2BGR)

        # 3. Hàm phụ trợ: Resize ảnh
        scale = 0.3


        def resize_img(image, scale_factor):
            width = int(image.shape[1] * scale_factor)
            height = int(image.shape[0] * scale_factor)
            return cv2.resize(image, (width, height))


        img_res = resize_img(img, scale)
        clean_res = resize_img(clean_img, scale)
        clahe_color_res = resize_img(clahe_color_img, scale)
        clahe_gray_res = resize_img(clahe_gray_3_channels, scale)

        # 4. Gắn nhãn (Text) lên từng ảnh đã resize
        # Tham số: (Ảnh, "Nội dung", (Tọa độ X, Y), Font, Size, (Màu BGR), Độ dày)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_res, "1. Original", (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(clean_res, "2. DullRazor", (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(clahe_color_res, "3. CLAHE Color", (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(clahe_gray_res, "4. Smoothed L-Channel", (10, 30), font, 0.8, (0, 255, 0), 2)

        # 5. Nối ảnh (Concatenate)
        # Cách 1: Nối ngang toàn bộ 4 ảnh (Dài thò lò)
        # combined_img = np.hstack((img_res, clean_res, clahe_color_res, clahe_gray_res))

        # Cách 2 (Khuyên dùng): Ghép thành lưới 2x2 để hiển thị đẹp nhất trên màn hình laptop
        top_row = np.hstack((img_res, clean_res))  # Hàng trên
        bottom_row = np.hstack((clahe_color_res, clahe_gray_res))  # Hàng dưới
        grid_combined = np.vstack((top_row, bottom_row))  # Ghép 2 hàng lại theo chiều dọc

        # 6. Hiển thị xem thành quả
        cv2.imshow("Preprocessing Pipeline Grid", grid_combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không tìm thấy ảnh! Hãy kiểm tra lại đường dẫn.")