import cv2
import numpy as np

from src.preprocessing import apply_circular_mask, remove_hair, enhance_contrast_clahe, boost_faint_edges
from src.segmentation import get_kmeans_mask, get_snakes_mask


def draw_contour(image, mask, color=(0, 0, 255)):
    result = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    return result

# 1. LOAD ẢNH VÀ CHẠY TIỀN XỬ LÝ (Chỉ chạy 1 lần)
img = cv2.imread("../data/toy_data/melanoma/ISIC_0000074.jpg")
img = cv2.resize(img, (600, 450))

# Preprocessing Pipeline
img_no_corners = apply_circular_mask(img, radius_reduction=0.85)
clean_img, _ = remove_hair(img_no_corners, kernel_size=15, inpaint_rad=3)
clahe_color_img, clahe_gray = enhance_contrast_clahe(clean_img, clip_limit=1.1)

boosted_l, _ = boost_faint_edges(clahe_gray)
smooth_l_channel = cv2.GaussianBlur(boosted_l, (5, 5), 0)

smooth_l_channel = apply_circular_mask(smooth_l_channel, radius_reduction=0.85)
clahe_color_img = apply_circular_mask(clahe_color_img, radius_reduction=0.85)

font = cv2.FONT_HERSHEY_SIMPLEX

# =====================================================================
# THÍ NGHIỆM 1: KHẢO SÁT THAM SỐ K TRONG K-MEANS
# =====================================================================
print("Đang chạy Thí nghiệm 1: K-Means Sweep...")
k_values = [2, 3, 4] # Các giá trị K cần test
kmeans_results = []

for k in k_values:
    # Chạy K-means với K tương ứng
    mask = get_kmeans_mask(clahe_color_img, k=k)
    # Vẽ viền xanh lá lên ảnh gốc
    result_img = draw_contour(img, mask, color=(0, 255, 0))
    # Dán nhãn thông số
    cv2.putText(result_img, f"K-Means: K={k}", (10, 30), font, 0.8, (0, 255, 255), 2)
    kmeans_results.append(result_img)

# Ghép 3 ảnh theo chiều ngang và lưu lại
final_kmeans_sweep = np.hstack(kmeans_results)
cv2.imwrite("kmeans_sweep_report.jpg", final_kmeans_sweep)
cv2.imshow("K-Means Parameter Sweep", final_kmeans_sweep)

# =====================================================================
# THÍ NGHIỆM 2: KHẢO SÁT VÒNG LẶP CỦA SNAKES (ĐƯỜNG VIỀN ĐỘNG)
# =====================================================================
print("Đang chạy Thí nghiệm 2: Snakes Iterations Sweep...")
# Lấy mask tối ưu từ K=4 để làm đầu vào cho Snakes
optimal_kmeans_mask = get_kmeans_mask(clahe_color_img, k=4)

iter_values = [5, 35, 100] # Thử Rắn bò 5 bước, 35 bước và 100 bước
snakes_results = []

for iters in iter_values:
    # Chạy Snakes với số vòng lặp tương ứng
    s_mask = get_snakes_mask(smooth_l_channel, optimal_kmeans_mask, num_iter=iters)
    # Vẽ viền đỏ
    result_img = draw_contour(img, s_mask, color=(0, 0, 255))
    cv2.putText(result_img, f"Snakes: iter={iters}", (10, 30), font, 0.8, (0, 255, 255), 2)
    snakes_results.append(result_img)

# Ghép 3 ảnh theo chiều ngang và lưu lại
final_snakes_sweep = np.hstack(snakes_results)
cv2.imwrite("snakes_sweep_report.jpg", final_snakes_sweep)
cv2.imshow("Snakes Iterations Sweep", final_snakes_sweep)

print("Đã lưu thành công 2 file ảnh: kmeans_sweep_report.jpg và snakes_sweep_report.jpg")
cv2.waitKey(0)
cv2.destroyAllWindows()