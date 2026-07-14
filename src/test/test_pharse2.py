import cv2
import numpy as np

from src.preprocessing import remove_hair, enhance_contrast_clahe, boost_faint_edges, apply_circular_mask
from src.segmentation import get_otsu_mask, get_snakes_mask, get_kmeans_mask
from src.evaluation import calculate_iou
# IMPORT ĐẦY ĐỦ BỘ TỨ ABCD
from src.features import (calculate_asymmetry, calculate_border_irregularity,
                               calculate_color_variegation, calculate_diameter)

# 1. Đọc ảnh gốc
img = cv2.imread("../../data/toy_data/melanoma/ISIC_0000029.jpg")
img = cv2.resize(img, (600, 450))

# Tẩy góc ống kính (Dùng thông số tối ưu em vừa tìm được)
img_no_corners = apply_circular_mask(img, radius_reduction=0.85)

# ----------------- GIAI ĐOẠN 1 (Tiền xử lý) -----------------
clean_img, _ = remove_hair(img_no_corners, kernel_size=15, inpaint_rad=3)
clahe_color_img, clahe_gray = enhance_contrast_clahe(clean_img, clip_limit=1.1)

boosted_l, faint_mask = boost_faint_edges(clahe_gray)
smooth_l_channel = cv2.GaussianBlur(boosted_l, (5, 5), 0)

smooth_l_channel = apply_circular_mask(smooth_l_channel, radius_reduction=0.85)
clahe_color_img = apply_circular_mask(clahe_color_img, radius_reduction=0.85)

# ----------------- GIAI ĐOẠN 2 (Phân đoạn) ------------------
kmeans_mask = get_kmeans_mask(clahe_color_img, k=4)
snakes_mask = get_snakes_mask(smooth_l_channel, kmeans_mask, num_iter=35)

# ----------------- CHẤM ĐIỂM IOU ----------------------------
gt_path = "../../data/toy_data/ground_truth/ISIC_0000029_segmentation.png"
gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

if gt_mask is not None:
    gt_mask_resized = cv2.resize(gt_mask, (600, 450))
    iou_score = calculate_iou(snakes_mask, gt_mask_resized)
    score_text = f"IoU: {iou_score * 100:.2f}%"
else:
    score_text = "IoU: N/A"

# ----------------- GIAI ĐOẠN 3: TRIẾT XUẤT ABCD -----------------
asymmetry_score = calculate_asymmetry(snakes_mask)
border_score = calculate_border_irregularity(snakes_mask)
color_score = calculate_color_variegation(img, snakes_mask) # Lấy màu trên ảnh gốc
diameter_mm = calculate_diameter(snakes_mask, mm_per_pixel=0.02)

# IN BÁO CÁO CHẨN ĐOÁN LÊN TERMINAL
print("\n" + "═"*50)
print("🩺 HỆ THỐNG TRÍ CHẨN ĐOÁN LÂM SÀNG TỰ ĐỘNG (ABCD REPORT)")
print("═"*50)
print(f"  [A] Asymmetry (Bất đối xứng): {asymmetry_score:.4f}  " + ("=> ⚠️ NGUY CƠ" if asymmetry_score > 0.4 else "=> ✅ An toàn"))
print(f"  [B] Border (Độ tròn viền):    {border_score:.4f}  " + ("=> ⚠️ NGUY CƠ (Nham nhở)" if border_score < 0.55 else "=> ✅ An toàn"))
print(f"  [C] Color (Biến thiên màu):   {color_score:.4f} " + ("=> ⚠️ NGUY CƠ (Loang lổ)" if color_score > 22.0 else "=> ✅ An toàn"))
print(f"  [D] Diameter (Đường kính):    {diameter_mm:.2f} mm " + ("=> ⚠️ NGUY CƠ (>6mm)" if diameter_mm > 6.0 else "=> ✅ An toàn"))
print("═"*50)
if (asymmetry_score > 0.4) or (border_score < 0.55) or (color_score > 22.0) or (diameter_mm > 6.0):
    print("🚨 KẾT LUẬN SƠ BỘ: Có dấu hiệu bất thường. Khuyến nghị sinh thiết!")
else:
    print("🟢 KẾT LUẬN SƠ BỘ: Nốt ruồi lành tính.")
print("═"*50)


# ----------------- HIỂN THỊ KẾT QUẢ -------------------------
def draw_contour_on_image(image, mask, color=(0, 0, 255)):
    result = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    return result

img_kmeans_result = draw_contour_on_image(img, kmeans_mask, color=(0, 255, 0))
img_snakes_result = draw_contour_on_image(img, snakes_mask, color=(0, 0, 255))

smooth_l_channel_3c = cv2.cvtColor(smooth_l_channel, cv2.COLOR_GRAY2BGR)
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(smooth_l_channel_3c, "1. L-Channel (Boosted+Masked)", (10, 30), font, 0.7, (0, 255, 0), 2)
cv2.putText(kmeans_mask_3c := cv2.cvtColor(kmeans_mask, cv2.COLOR_GRAY2BGR), "2. K-Means Mask", (10, 30), font, 0.7, (0, 255, 0), 2)
cv2.putText(img_kmeans_result, "3. K-Means Contour", (10, 30), font, 0.7, (0, 255, 0), 2)

# IN THÔNG SỐ LÊN ẢNH ĐỂ ĐI THUYẾT TRÌNH
cv2.putText(img_snakes_result, score_text, (10, 30), font, 0.7, (0, 255, 255), 2)
cv2.putText(img_snakes_result, f"A: {asymmetry_score:.2f}", (10, 60), font, 0.6, (255, 255, 255), 1)
cv2.putText(img_snakes_result, f"B: {border_score:.2f}", (10, 80), font, 0.6, (255, 255, 255), 1)
cv2.putText(img_snakes_result, f"C: {color_score:.1f}", (10, 100), font, 0.6, (255, 255, 255), 1)
cv2.putText(img_snakes_result, f"D: {diameter_mm:.1f}mm", (10, 120), font, 0.6, (255, 255, 255), 1)

top = np.hstack((smooth_l_channel_3c, kmeans_mask_3c))
bottom = np.hstack((img_kmeans_result, img_snakes_result))

cv2.imshow("Segmentation and ABCD Extraction", np.vstack((top, bottom)))
cv2.waitKey(0)
cv2.destroyAllWindows()