import cv2
import numpy as np
import sys
import io

# Đảm bảo in được tiếng Việt trên console Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.preprocessing import apply_circular_mask, remove_hair, enhance_contrast_clahe, boost_faint_edges
from src.segmentation import get_otsu_mask, get_kmeans_mask, get_snakes_mask
from src.evaluation import calculate_iou

def draw_contour(image, mask, color, thickness=2):
    res = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(res, contours, -1, color, thickness)
    return res

def run_ablation_study(img_path, gt_path):
    print(f"\n--- ĐANG XỬ LÝ: {img_path} ---")
    
    # 1. Đọc ảnh
    img = cv2.imread(img_path)
    img = cv2.resize(img, (600, 450))
    
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print("Lỗi: Không tìm thấy ảnh Ground Truth!")
        return
    gt_mask = cv2.resize(gt_mask, (600, 450))
    
    # 2. Tiền xử lý (Giữ nguyên cho cả 3 kịch bản)
    img_no_corners = apply_circular_mask(img, radius_reduction=0.85)
    clean_img, _ = remove_hair(img_no_corners, kernel_size=15, inpaint_rad=3)
    clahe_color_img, clahe_gray = enhance_contrast_clahe(clean_img, clip_limit=1.1)
    
    boosted_l, _ = boost_faint_edges(clahe_gray)
    smooth_l_channel = cv2.GaussianBlur(boosted_l, (5, 5), 0)
    smooth_l_channel = apply_circular_mask(smooth_l_channel, radius_reduction=0.85)
    clahe_color_img = apply_circular_mask(clahe_color_img, radius_reduction=0.85)
    
    # 3. Kịch bản 1: Thuật toán Cơ sở (Otsu Thresholding)
    otsu_mask = get_otsu_mask(smooth_l_channel)
    iou_otsu = calculate_iou(otsu_mask, gt_mask)
    
    # 4. Kịch bản 2: Nâng cấp Không gian màu (K-Means Clustering)
    kmeans_mask = get_kmeans_mask(clahe_color_img, k=4)
    iou_kmeans = calculate_iou(kmeans_mask, gt_mask)
    
    # 5. Kịch bản 3: Kiến trúc Hoàn chỉnh (K-Means + Morphological Snakes)
    snakes_mask = get_snakes_mask(smooth_l_channel, kmeans_mask, num_iter=35)
    iou_snakes = calculate_iou(snakes_mask, gt_mask)
    
    # --- OUTPUT 1: Bảng Log Số liệu ---
    print("\n[BÁO CÁO BÓC TÁCH KIẾN TRÚC (ABLATION STUDY)]")
    print(f"1. Baseline (Otsu):         {iou_otsu * 100:.2f}%")
    print(f"2. Nâng cấp 1 (K-Means):    {iou_kmeans * 100:.2f}%")
    print(f"3. Hoàn chỉnh (Snakes):     {iou_snakes * 100:.2f}%")
    
    print("\n[PHÂN TÍCH HIỆU NĂNG]")
    diff_kmeans_otsu = (iou_kmeans - iou_otsu) * 100
    print(f"=> K-Means cải thiện {diff_kmeans_otsu:+.2f}% so với Otsu (Khắc phục nhiễu sáng).")
    
    diff_snakes_kmeans = (iou_snakes - iou_kmeans) * 100
    print(f"=> Việc thêm Snakes giúp cải thiện thêm {diff_snakes_kmeans:+.2f}% so với chỉ dùng K-Means (Bám khít đường viền nham nhở).")
    
    diff_total = (iou_snakes - iou_otsu) * 100
    print(f"=> TỔNG CỘNG: Kiến trúc đề xuất tăng {diff_total:+.2f}% so với thuật toán cơ sở.")
    
    # --- OUTPUT 2: Bức ảnh Ghép Trực quan ---
    # Ảnh 1: Ảnh gốc + Viền GT (Trắng)
    img_gt = draw_contour(img, gt_mask, color=(255, 255, 255))
    
    # Ảnh 2: Kịch bản 1 + Viền Đỏ
    img_otsu = draw_contour(img, otsu_mask, color=(0, 0, 255)) # BGR: Đỏ
    
    # Ảnh 3: Kịch bản 2 + Viền Xanh lá
    img_kmeans = draw_contour(img, kmeans_mask, color=(0, 255, 0)) # BGR: Xanh lá
    
    # Ảnh 4: Kịch bản 3 + Viền Vàng
    img_snakes = draw_contour(img, snakes_mask, color=(0, 255, 255)) # BGR: Vàng
    
    # Vẽ text lên ảnh
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_gt, "1. Goc + GT (Trang)", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(img_otsu, f"2. Otsu (IoU: {iou_otsu*100:.1f}%)", (10, 30), font, 0.7, (0, 0, 255), 2)
    cv2.putText(img_kmeans, f"3. K-Means (IoU: {iou_kmeans*100:.1f}%)", (10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(img_snakes, f"4. Snakes (IoU: {iou_snakes*100:.1f}%)", (10, 30), font, 0.7, (0, 255, 255), 2)
    
    # Ghép 4 ảnh thành 1 lưới 2x2
    top_row = np.hstack((img_gt, img_otsu))
    bottom_row = np.hstack((img_kmeans, img_snakes))
    grid_img = np.vstack((top_row, bottom_row))
    
    save_path = "ablation_study_result.jpg"
    cv2.imwrite(save_path, grid_img)
    print(f"\nĐã xuất ảnh trực quan ra file: {save_path}")

if __name__ == "__main__":
    train_img_path = "data/train/images/ISIC_0000002.jpg" 
    train_gt_path = "data/train/masks/ISIC_0000002_segmentation.png"
    run_ablation_study(train_img_path, train_gt_path)
