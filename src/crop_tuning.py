import cv2
import numpy as np


from src.preprocessing import apply_circular_mask


img = cv2.imread(r"D:\Computer Vision Final Project\Src code\data\toy_data\melanoma\ISIC_0000036.jpg")
img = cv2.resize(img, (600, 450))


def nothing(x):
    pass



cv2.namedWindow('Lens Crop Tuning', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Lens Crop Tuning', 1200, 500)



cv2.createTrackbar('Crop Radius (%)', 'Lens Crop Tuning', 95, 120, nothing)


cv2.createTrackbar('CLAHE Limit', 'Lens Crop Tuning', 11, 30, nothing)

print("Đang mở Trạm điều khiển... Nhấn 'q' để chốt thông số và thoát.")

while True:
    
    radius_percent = cv2.getTrackbarPos('Crop Radius (%)', 'Lens Crop Tuning')
    clahe_val = cv2.getTrackbarPos('CLAHE Limit', 'Lens Crop Tuning')

    
    if radius_percent < 50: radius_percent = 50

    
    radius_reduction = radius_percent / 100.0
    clip_limit = clahe_val / 10.0
    if clip_limit < 0.1: clip_limit = 0.1

    
    preview_img = img.copy()
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = int(min(h, w) / 2 * radius_reduction)

    
    cv2.circle(preview_img, center, radius, (0, 0, 255), 2)
    cv2.putText(preview_img, f"Radius: {radius_reduction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    
    
    masked_img = apply_circular_mask(img, radius_reduction=radius_reduction)

    
    lab = cv2.cvtColor(masked_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    demo_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    cv2.putText(demo_clahe, f"CLAHE: {clip_limit}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    
    
    combined = np.hstack((preview_img, masked_img, demo_clahe))

    cv2.imshow('Lens Crop Tuning', combined)

    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print(f"\n[THÔNG SỐ CHỐT]")
        print(f"=> apply_circular_mask(img, radius_reduction={radius_reduction})")
        print(f"=> enhance_contrast_clahe(img, clip_limit={clip_limit})")
        break

cv2.destroyAllWindows()