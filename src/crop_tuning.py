import cv2
import numpy as np

# Nhớ import hàm cắt góc của em vào nhé
from src.preprocessing import apply_circular_mask

# 1. Đọc ảnh (Vẫn giữ resize để ảnh không tràn màn hình nhé)
img = cv2.imread(r"D:\Computer Vision Final Project\Src code\data\toy_data\melanoma\ISIC_0000036.jpg")
img = cv2.resize(img, (600, 450))


def nothing(x):
    pass


# Tạo cửa sổ Tuning
cv2.namedWindow('Lens Crop Tuning', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Lens Crop Tuning', 1200, 500)

# 2. Tạo thanh trượt cho Bán kính vòng tròn (Tính bằng %)
# Giá trị mặc định là 85%, kéo từ 50% đến 100%
cv2.createTrackbar('Crop Radius (%)', 'Lens Crop Tuning', 95, 120, nothing)

# Thêm luôn thanh trượt cho CLAHE để vặn độ tương phản cho tiện
cv2.createTrackbar('CLAHE Limit', 'Lens Crop Tuning', 11, 30, nothing)

print("Đang mở Trạm điều khiển... Nhấn 'q' để chốt thông số và thoát.")

while True:
    # Lấy thông số từ thanh trượt
    radius_percent = cv2.getTrackbarPos('Crop Radius (%)', 'Lens Crop Tuning')
    clahe_val = cv2.getTrackbarPos('CLAHE Limit', 'Lens Crop Tuning')

    # Xử lý logic chống lỗi (Không cho cắt nhỏ hơn 50% ảnh)
    if radius_percent < 50: radius_percent = 50

    # Quy đổi ra số thập phân (Ví dụ 85 -> 0.85)
    radius_reduction = radius_percent / 100.0
    clip_limit = clahe_val / 10.0
    if clip_limit < 0.1: clip_limit = 0.1

    # --- BƯỚC 1: TRỰC QUAN HÓA LƯỠI DAO CẮT ---
    preview_img = img.copy()
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = int(min(h, w) / 2 * radius_reduction)

    # Vẽ vòng tròn ĐỎ lên ảnh gốc để xem nó cắt trúng khối u không
    cv2.circle(preview_img, center, radius, (0, 0, 255), 2)
    cv2.putText(preview_img, f"Radius: {radius_reduction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # --- BƯỚC 2: KẾT QUẢ THỰC TẾ ---
    # Chạy hàm apply_circular_mask để xem ảnh sau khi bị tẩy trắng 4 góc
    masked_img = apply_circular_mask(img, radius_reduction=radius_reduction)

    # (Tùy chọn) Chạy demo thử CLAHE để xem viền mờ có bị lấp không
    lab = cv2.cvtColor(masked_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    demo_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    cv2.putText(demo_clahe, f"CLAHE: {clip_limit}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # --- BƯỚC 3: GHÉP ẢNH HIỂN THỊ ---
    # Ghép 3 tấm: Ảnh gốc có viền đỏ -> Ảnh đã tẩy trắng -> Ảnh đã nâng sáng
    combined = np.hstack((preview_img, masked_img, demo_clahe))

    cv2.imshow('Lens Crop Tuning', combined)

    # Nhấn phím 'q' hoặc 'Esc' để thoát
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print(f"\n[THÔNG SỐ CHỐT]")
        print(f"=> apply_circular_mask(img, radius_reduction={radius_reduction})")
        print(f"=> enhance_contrast_clahe(img, clip_limit={clip_limit})")
        break

cv2.destroyAllWindows()