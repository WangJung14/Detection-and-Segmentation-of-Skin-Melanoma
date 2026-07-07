import cv2
import numpy as np
from skimage.segmentation import morphological_chan_vese
from src.preprocessing import remove_hair, enhance_contrast_clahe

# 1. Đọc ảnh (Sửa lại đường dẫn của em)
img = cv2.imread("../data/toy_data/melanoma/ISIC_0000074.jpg")
img = cv2.resize(img, (400, 300))

# Đọc ảnh Ground Truth (Đáp án của bác sĩ) để đối chiếu
gt_mask = cv2.imread("../data/toy_data/ground_truth/ISIC_0000074_segmentation.png", cv2.IMREAD_GRAYSCALE)
gt_mask = cv2.resize(gt_mask, (400, 300))

# 2. CHẠY GIAI ĐOẠN 1 MỘT LẦN DUY NHẤT (Để tiết kiệm thời gian)
print("Đang chạy Tiền xử lý (Phase 1)...")
clean_img, _ = remove_hair(img, kernel_size=15, inpaint_rad=3)
_, clahe_gray = enhance_contrast_clahe(clean_img, clip_limit=1.2)
smooth_l = cv2.GaussianBlur(clahe_gray, (5, 5), 0)
print("Xong Tiền xử lý! Mở giao diện Tuning...")


def nothing(x):
    pass


cv2.namedWindow('Seg Tuning', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Seg Tuning', 1200, 600)

# Các thanh trượt điều khiển Hình thái học (Dọn rác Otsu)
cv2.createTrackbar('Open Size (Xoa nhieu)', 'Seg Tuning', 5, 31, nothing)
cv2.createTrackbar('Close Size (Lap lo)', 'Seg Tuning', 15, 51, nothing)

# Các thanh trượt điều khiển Rắn bò (Snakes)
cv2.createTrackbar('Snake Iterations', 'Seg Tuning', 20, 100, nothing)
cv2.createTrackbar('Snake Smooth', 'Seg Tuning', 1, 4, nothing)

while True:
    o_size = cv2.getTrackbarPos('Open Size (Xoa nhieu)', 'Seg Tuning')
    c_size = cv2.getTrackbarPos('Close Size (Lap lo)', 'Seg Tuning')
    iters = cv2.getTrackbarPos('Snake Iterations', 'Seg Tuning')
    smooth = cv2.getTrackbarPos('Snake Smooth', 'Seg Tuning')

    # Ép thành số lẻ
    if o_size % 2 == 0: o_size += 1
    if c_size % 2 == 0: c_size += 1
    if o_size < 3: o_size = 3
    if c_size < 3: c_size = 3
    if iters < 1: iters = 1

    # --- BƯỚC 1: OTSU & HÌNH THÁI HỌC ---
    _, thresh = cv2.threshold(smooth_l, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (o_size, o_size))
    mask_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_open)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c_size, c_size))
    otsu_mask = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, k_close)

    # --- THỦ THUẬT SENIOR: GIỮ LẠI CONTOUR LỚN NHẤT ---
    # Quét tìm tất cả các đảo trắng
    contours, _ = cv2.findContours(otsu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_otsu_mask = np.zeros_like(otsu_mask)
    if contours:
        # Tìm đảo có diện tích to nhất
        largest_contour = max(contours, key=cv2.contourArea)
        # Vẽ lại duy nhất đảo đó lên một nền đen mới
        cv2.drawContours(clean_otsu_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # --- BƯỚC 2: RẮN BÒ (SNAKES) ---
    img_float = smooth_l.astype(np.float32) / 255.0
    mask_float = (clean_otsu_mask > 0).astype(np.float32)

    snake_evo = morphological_chan_vese(
        img_float, num_iter=iters, init_level_set=mask_float, smoothing=smooth, lambda1=1, lambda2=1
    )
    snakes_mask = (snake_evo.astype(np.float64) * 255).astype(np.uint8)

    # --- HIỂN THỊ TRỰC QUAN ---
    # Đổi sang 3 kênh để vẽ viền đỏ/xanh lên cho đẹp
    otsu_colored = cv2.cvtColor(clean_otsu_mask, cv2.COLOR_GRAY2BGR)
    snake_colored = cv2.cvtColor(snakes_mask, cv2.COLOR_GRAY2BGR)
    gt_colored = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)

    cv2.putText(otsu_colored, "1. Otsu (Cleaned)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(snake_colored, "2. Snakes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(gt_colored, "3. Ground Truth (Doctor)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Ghép 3 ảnh nằm ngang để so sánh
    combined = np.hstack((otsu_colored, snake_colored, gt_colored))
    cv2.imshow('Seg Tuning', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print(
            f"THÔNG SỐ CHỐT SEGMENTATION:\nOpen: {o_size}\nClose: {c_size}\nSnake Iter: {iters}\nSnake Smooth: {smooth}")
        break

cv2.destroyAllWindows()