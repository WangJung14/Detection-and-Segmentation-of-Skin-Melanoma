import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import io

# Đảm bảo in được tiếng Việt trên console Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.preprocessing import apply_circular_mask, remove_hair, enhance_contrast_clahe, boost_faint_edges
from src.segmentation import get_otsu_mask, get_kmeans_mask, get_snakes_mask
from src.u_net.inference import UNetInferencer
from src.ensemble import majority_voting
from src.evaluation import calculate_iou
from src.features import evaluate_abcd_rule

def run_pipeline(img_path, gt_path, unet_inferencer):
    print(f"\n--- ĐANG XỬ LÝ: {img_path} ---")
    
    if not os.path.exists(img_path):
        print(f"Lỗi: Không tìm thấy ảnh {img_path}")
        return

    # 1. Đọc ảnh
    img = cv2.imread(img_path)
    img = cv2.resize(img, (600, 450))
    
    gt_mask = None
    if os.path.exists(gt_path):
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            gt_mask = cv2.resize(gt_mask, (600, 450))
    
    # 2. Tiền xử lý
    print("Đang chạy Tiền xử lý (DullRazor + CLAHE)...")
    img_no_corners = apply_circular_mask(img, radius_reduction=0.85)
    clean_img, _ = remove_hair(img_no_corners, kernel_size=15, inpaint_rad=3)
    clahe_color_img, clahe_gray = enhance_contrast_clahe(clean_img, clip_limit=1.1)
    
    boosted_l, _ = boost_faint_edges(clahe_gray)
    smooth_l_channel = cv2.GaussianBlur(boosted_l, (5, 5), 0)
    smooth_l_channel = apply_circular_mask(smooth_l_channel, radius_reduction=0.85)
    clahe_color_img = apply_circular_mask(clahe_color_img, radius_reduction=0.85)
    
    # 3. Phân đoạn độc lập (Voters)
    print("Đang chạy phân đoạn CV truyền thống (Otsu, K-Means, Snakes)...")
    otsu_mask = get_otsu_mask(smooth_l_channel)
    kmeans_mask = get_kmeans_mask(clahe_color_img, k=4)
    snakes_mask = get_snakes_mask(smooth_l_channel, kmeans_mask, num_iter=35)
    
    print("Đang chạy phân đoạn Học sâu (U-Net)...")
    unet_mask = unet_inferencer.predict(clahe_color_img)
    
    # 4. Majority Voting (Ensemble)
    print("Đang kết hợp kết quả (Majority Voting)...")
    # Thử nghiệm với 3 voters (K-Means, Snakes, U-Net) - Threshold = 2
    ensemble3_mask = majority_voting(kmeans_mask, snakes_mask, unet_mask, threshold=2)
    
    # Thử nghiệm với 4 voters (Otsu, K-Means, Snakes, U-Net) - Threshold = 3
    ensemble4_mask = majority_voting(otsu_mask, kmeans_mask, snakes_mask, unet_mask, threshold=3)
    
    # 5. Đánh giá IoU
    if gt_mask is not None:
        iou_otsu = calculate_iou(otsu_mask, gt_mask)
        iou_kmeans = calculate_iou(kmeans_mask, gt_mask)
        iou_snakes = calculate_iou(snakes_mask, gt_mask)
        iou_unet = calculate_iou(unet_mask, gt_mask)
        iou_ens3 = calculate_iou(ensemble3_mask, gt_mask)
        iou_ens4 = calculate_iou(ensemble4_mask, gt_mask)
        
        print("\n[BẢNG XẾP HẠNG IOU]")
        print(f"Otsu:       {iou_otsu:.4f}")
        print(f"K-Means:    {iou_kmeans:.4f}")
        print(f"Snakes:     {iou_snakes:.4f}")
        print(f"U-Net:      {iou_unet:.4f}")
        print(f"Ensemble 3: {iou_ens3:.4f} (K-Means, Snakes, U-Net)")
        print(f"Ensemble 4: {iou_ens4:.4f} (Otsu, K-Means, Snakes, U-Net)")
        
        # Chọn mask tốt nhất để trích xuất lâm sàng
        best_mask = ensemble3_mask if iou_ens3 >= iou_ens4 else ensemble4_mask
        best_name = "Ensemble 3" if iou_ens3 >= iou_ens4 else "Ensemble 4"
    else:
        print("\nKhông tìm thấy Ground Truth để tính IoU. Sử dụng Ensemble 3 làm mặc định.")
        best_mask = ensemble3_mask
        best_name = "Ensemble 3"
        iou_otsu = iou_kmeans = iou_snakes = iou_unet = iou_ens3 = iou_ens4 = 0.0

    print(f"\n=> Đã chọn {best_name} làm Mask cuối cùng.")
    
    # 6. Đánh giá lâm sàng ABCD
    print("\n[ĐÁNH GIÁ ABCD RULE TRÊN MASK TỐT NHẤT]")
    abcd = evaluate_abcd_rule(img, best_mask, kmeans_mask, pixel_per_mm=50.0)
    print(f"A: {abcd['A']}, B: {abcd['B']}, C: {abcd['C']}, D: {abcd['D_mm']}mm")
    print(f"TDS: {abcd['TDS']} -> Kết luận: {abcd['Diagnosis']}")
    
    # 7. Trực quan hóa
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    masks_to_show = [
        ("1. Anh goc", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None),
        ("2. Ground Truth", gt_mask if gt_mask is not None else np.zeros_like(otsu_mask), None),
        (f"3. Otsu (IoU: {iou_otsu:.2f})", otsu_mask, "gray"),
        (f"4. K-Means (IoU: {iou_kmeans:.2f})", kmeans_mask, "gray"),
        (f"5. Snakes (IoU: {iou_snakes:.2f})", snakes_mask, "gray"),
        (f"6. U-Net (IoU: {iou_unet:.2f})", unet_mask, "gray"),
        (f"7. Ens 3 (IoU: {iou_ens3:.2f})", ensemble3_mask, "gray"),
        (f"8. Ens 4 (IoU: {iou_ens4:.2f})", ensemble4_mask, "gray")
    ]
    
    for ax, (title, img_data, cmap) in zip(axes, masks_to_show):
        if img_data is not None:
            if cmap:
                ax.imshow(img_data, cmap=cmap)
            else:
                ax.imshow(img_data)
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()
    save_path = "ensemble_comparison.png"
    plt.savefig(save_path)
    print(f"\nĐã xuất ảnh so sánh ra file: {save_path}")
    
if __name__ == "__main__":
    # 1. Khởi tạo U-Net Inferencer 1 lần để tải trọng số (Dùng best_model.pth theo ý người dùng)
    print("Đang nạp mô hình U-Net vào bộ nhớ...")
    unet_inferencer = UNetInferencer(weight_path="best_model.pth")
    
    # 2. Chạy nghiệm thu trên 1 ảnh trong tập train
    train_img_path = "data/train/images/ISIC_0000002.jpg" 
    train_gt_path = "data/train/masks/ISIC_0000002_segmentation.png"
    
    run_pipeline(train_img_path, train_gt_path, unet_inferencer)
