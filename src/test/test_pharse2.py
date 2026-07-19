import cv2
import numpy as np

from src.preprocessing import remove_hair, enhance_contrast_clahe, boost_faint_edges, apply_circular_mask
from src.segmentation import get_otsu_mask, get_snakes_mask, get_kmeans_mask
from src.evaluation import calculate_iou
from src.features import evaluate_abcd_rule


img = cv2.imread(r"D:\Computer Vision Final Project\Src code\data\toy_data\melanoma\ISIC_0000029.jpg")
img = cv2.resize(img, (600, 450))


img_no_corners = apply_circular_mask(img, radius_reduction=0.93)


clean_img, _ = remove_hair(img_no_corners, kernel_size=15, inpaint_rad=3)
clahe_color_img, clahe_gray = enhance_contrast_clahe(clean_img, clip_limit=1.3)

boosted_l, faint_mask = boost_faint_edges(clahe_gray)
smooth_l_channel = cv2.GaussianBlur(boosted_l, (5, 5), 0)

smooth_l_channel = apply_circular_mask(smooth_l_channel, radius_reduction=0.93)
clahe_color_img = apply_circular_mask(clahe_color_img, radius_reduction=0.93)


kmeans_mask = get_kmeans_mask(clahe_color_img, k=4)
snakes_mask = get_snakes_mask(smooth_l_channel, kmeans_mask, num_iter=35)


gt_path = r"D:\Computer Vision Final Project\Src code\data\toy_data\ground_truth\ISIC_0000029_segmentation.png"
gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

if gt_mask is not None:
    gt_mask_resized = cv2.resize(gt_mask, (600, 450))
    iou_score = calculate_iou(snakes_mask, gt_mask_resized)
    score_text = f"IoU: {iou_score * 100:.2f}%"
else:
    score_text = "IoU: N/A"


abcd_results = evaluate_abcd_rule(img, snakes_mask, kmeans_mask, pixel_per_mm=50.0)
asymmetry_score = abcd_results["A"]
border_score = abcd_results["B"]
color_score = abcd_results["C"]
diameter_mm = abcd_results["D_mm"]
tds = abcd_results["TDS"]
diagnosis = abcd_results["Diagnosis"]


print("\n" + "═"*50)
print("🩺 HỆ THỐNG CHẨN ĐOÁN LÂM SÀNG TỰ ĐỘNG (ABCD REPORT)")
print("═"*50)
print(f"  [A] Asymmetry (Bất đối xứng): {asymmetry_score}/2")
print(f"  [B] Border (Độ nham nhở):     {border_score}/8")
print(f"  [C] Color (Đa dạng màu):      {color_score}/6")
print(f"  [D] Diameter (Đường kính):    {diameter_mm:.2f} mm")
print(f"  [TDS] Total Score:            {tds:.2f}")
print("═"*50)
if diagnosis == "Benign":
    print(f"🟢 KẾT LUẬN SƠ BỘ: {diagnosis} (Lành tính)")
elif diagnosis == "Suspicious":
    print(f"🟡 KẾT LUẬN SƠ BỘ: {diagnosis} (Đáng ngờ - Cần theo dõi)")
else:
    print(f"🚨 KẾT LUẬN SƠ BỘ: {diagnosis} (Ung thư hắc tố. Khuyến nghị sinh thiết!)")
print("═"*50)



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


cv2.putText(img_snakes_result, score_text, (10, 30), font, 0.7, (0, 255, 255), 2)
cv2.putText(img_snakes_result, f"A:{asymmetry_score} B:{border_score} C:{color_score} D:{diameter_mm}mm", (10, 60), font, 0.6, (255, 255, 255), 1)
cv2.putText(img_snakes_result, f"TDS: {tds:.2f} ({diagnosis})", (10, 90), font, 0.6, (0, 255, 255), 2)

top = np.hstack((smooth_l_channel_3c, kmeans_mask_3c))
bottom = np.hstack((img_kmeans_result, img_snakes_result))

cv2.imshow("Segmentation and ABCD Extraction", np.vstack((top, bottom)))
cv2.waitKey(0)
cv2.destroyAllWindows()