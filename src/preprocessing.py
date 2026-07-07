import cv2
import numpy as np




def remove_hair(image, kernel_size=17, threshold_val=19, inpaint_rad=5):
    """
    Thuật toán DullRazor phiên bản nâng cấp:
    - Có thể tinh chỉnh tham số tùy theo độ dày của lông.
    - Bổ sung Dilation (Giãn nở) để xóa sạch bóng ma (ghosting).
    """
    """
    Điều chỉnh tham số 
    
    Nếu lông quá to và dàu : Tăng kernel_size lên 21 hoặc 25 , luôn là số lẻ
    Nếu mask có quá nhiều hạt nhiễu : Tăng threshold_val từ 15 lên 20 hoặc 25. Các chấm nhiễu sẽ biến mất chỉ gữ lại đường nét dài của sợi lông
    Nếu ảnh cuối cùng vẫn còn vệt mờ của lông : Tăng iterations trong func cv2.dilate lên 2( nghĩa là tăng size của mask lên 2 lần) hoặc tăng inpaint_rad lên 7
    
    """

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Black-hat Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 3. Phân ngưỡng (Thresholding)
    # Nâng threshold_val lên (mặc định 15 thay vì 10) để bớt bắt nhiễu vân da
    _, hair_mask = cv2.threshold(blackhat, threshold_val, 255, cv2.THRESH_BINARY)

    # 4. FIX CỐT LÕI: Phép toán Giãn nở (Dilation)
    # Tạo một "bút tô" kích thước 3x3
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Phình to các vệt trắng của sợi lông thêm 1-2 pixel để bao trùm hết rìa đen của sợi lông thật
    hair_mask_dilated = cv2.dilate(hair_mask, kernel_dilate, iterations=2)

    # 5. Inpainting
    # Tăng bán kính inpaint_rad lên (mặc định 5 thay vì 1) để nội suy màu rộng hơn
    clean_image = cv2.inpaint(image, hair_mask_dilated, inpaintRadius=inpaint_rad, flags=cv2.INPAINT_TELEA)

    return clean_image, hair_mask_dilated


# Test function
if __name__ == "__main__":
    # Đọc ảnh gốc (thay đường dẫn bằng ảnh thực tế của em)
    img = cv2.imread("../data/toy_data/melanoma/ISIC_0000138.jpg")

    if img is not None:
        clean_img, mask = remove_hair(img)

        scale = 0.3

        img_show = cv2.resize(img, None, fx=scale, fy=scale)
        mask_show = cv2.resize(mask, (img_show.shape[1], img_show.shape[0]))
        mask_show = cv2.cvtColor(mask_show, cv2.COLOR_GRAY2BGR)
        clean_show = cv2.resize(clean_img, None, fx=scale, fy=scale)

        result = cv2.hconcat([img_show, mask_show, clean_show])

        cv2.imshow("Hair Removal Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không tìm thấy ảnh! Hãy kiểm tra lại đường dẫn.")