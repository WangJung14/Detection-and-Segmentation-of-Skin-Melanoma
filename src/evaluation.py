import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.u_net.inference import UNetInferencer
from src.segmentation import get_otsu_mask, get_kmeans_mask, get_snakes_mask

def calculate_metrics(mask_pred, mask_gt):
    """
    Tính toán các chỉ số đánh giá bằng Confusion Matrix (TP, TN, FP, FN).
    """
    # 1. Đảm bảo 2 mask có cùng kích thước
    if mask_pred.shape != mask_gt.shape:
        mask_pred = cv2.resize(mask_pred, (mask_gt.shape[1], mask_gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 2. Nhị phân hóa (Chỉ giữ lại 0 và 1)
    pred_bin = (mask_pred > 127).astype(np.uint8)
    gt_bin = (mask_gt > 127).astype(np.uint8)

    # 3. Tính TP, TN, FP, FN
    TP = np.sum(np.logical_and(pred_bin == 1, gt_bin == 1))
    TN = np.sum(np.logical_and(pred_bin == 0, gt_bin == 0))
    FP = np.sum(np.logical_and(pred_bin == 1, gt_bin == 0))
    FN = np.sum(np.logical_and(pred_bin == 0, gt_bin == 1))

    # 4. Tính toán 5 chỉ số (Thêm epsilon để tránh chia cho 0)
    epsilon = 1e-7
    
    iou = TP / (TP + FP + FN + epsilon)
    dice = (2 * TP) / (2 * TP + FP + FN + epsilon)
    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    sensitivity = TP / (TP + FN + epsilon)
    specificity = TN / (TN + FP + epsilon)

    return {
        "IoU": iou,
        "Dice": dice,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }


def evaluate_pipeline(test_img_dir, test_gt_dir, model_type="unet", output_dir="evaluation_results", weight_path="best_model.pth"):
    """
    Vòng lặp Quét Toàn bộ Tập Test (The Evaluation Loop)
    - model_type: 'unet' hoặc 'traditional' (K-Means + Snakes)
    """
    print(f"Start evaluation for model: {model_type.upper()}")
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    error_cases_dir = os.path.join(output_dir, "Error_Cases")
    os.makedirs(error_cases_dir, exist_ok=True)

    # Chuẩn bị mô hình
    inferencer = None
    if model_type == "unet":
        inferencer = UNetInferencer(weight_path=weight_path)

    results_list = []
    
    # Lọc danh sách ảnh
    image_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith(".jpg")])
    
    for img_name in tqdm(image_files, desc="Evaluating Images"):
        # Đường dẫn
        img_path = os.path.join(test_img_dir, img_name)
        gt_name = img_name.replace(".jpg", "_segmentation.png")
        gt_path = os.path.join(test_gt_dir, gt_name)
        
        # Đọc ảnh
        bgr_image = cv2.imread(img_path)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if bgr_image is None or gt_mask is None:
            continue
            
        # Dự đoán Mask
        pred_mask = None
        if model_type == "unet":
            pred_mask = inferencer.predict(bgr_image)
        elif model_type == "traditional":
            h_orig, w_orig = bgr_image.shape[:2]
            small_bgr = cv2.resize(bgr_image, (256, 256))
            small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
            # Khởi tạo K-Means rồi chạy Snakes trên ảnh nhỏ
            init_mask = get_kmeans_mask(small_bgr, k=3)
            small_pred_mask = get_snakes_mask(small_gray, init_mask)
            # Đưa về kích thước ban đầu
            pred_mask = cv2.resize(small_pred_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
        # Đánh giá
        metrics = calculate_metrics(pred_mask, gt_mask)
        
        # Lưu dòng dữ liệu
        row = {"Image_ID": img_name}
        row.update(metrics)
        results_list.append(row)
        
    # Chuyển thành DataFrame
    df = pd.DataFrame(results_list)
    
    # Giai đoạn 3: Kết xuất Báo cáo & Phân tích Lỗi
    
    # 1. Bảng Thống kê
    stats = df.mean(numeric_only=True).to_frame("Mean").join(df.std(numeric_only=True).to_frame("Std"))
    stats_csv_path = os.path.join(output_dir, f"{model_type}_statistics.csv")
    stats.to_csv(stats_csv_path)
    print(f"\n[Statistics Mean ± Std saved to: {stats_csv_path}]")
    print(stats)
    
    # Toàn bộ dữ liệu
    full_csv_path = os.path.join(output_dir, f"{model_type}_full_results.csv")
    df.to_csv(full_csv_path, index=False)
    
    # 2. Vẽ Biểu đồ Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(df['IoU'], bins=20, kde=True, color="blue")
    plt.title(f"IoU Distribution - {model_type.upper()}")
    plt.xlabel("Intersection over Union (IoU)")
    plt.ylabel("Frequency (Number of Images)")
    plt.grid(True, alpha=0.3)
    
    hist_path = os.path.join(output_dir, f"{model_type}_iou_histogram.png")
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Histogram chart saved to: {hist_path}]")
    
    # 3. Bóc tách 5 "Tội đồ" (Error Analysis)
    worst_5 = df.sort_values(by='IoU').head(5)
    print("\n[Top 5 worst cases (lowest IoU)]:")
    for index, row in worst_5.iterrows():
        print(f" - {row['Image_ID']}: IoU = {row['IoU']:.4f}")
        
        # Lưu 5 cặp ảnh
        img_name = row['Image_ID']
        img_path = os.path.join(test_img_dir, img_name)
        gt_name = img_name.replace(".jpg", "_segmentation.png")
        gt_path = os.path.join(test_gt_dir, gt_name)
        
        bgr_image = cv2.imread(img_path)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Chạy lại predict để lấy mask trực quan
        pred_mask = None
        if model_type == "unet":
            pred_mask = inferencer.predict(bgr_image)
        elif model_type == "traditional":
            h_orig, w_orig = bgr_image.shape[:2]
            small_bgr = cv2.resize(bgr_image, (256, 256))
            small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
            init_mask = get_kmeans_mask(small_bgr, k=3)
            small_pred_mask = get_snakes_mask(small_gray, init_mask)
            pred_mask = cv2.resize(small_pred_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
        # Ghép ảnh ngang (Ảnh gốc, Mask Dự đoán, Mask Chuẩn)
        # Chuyển mask sang BGR để ghép
        pred_mask_bgr = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
        gt_mask_bgr = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
        
        # Viết chữ lên ảnh
        cv2.putText(bgr_image, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(pred_mask_bgr, f"Pred ({model_type})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(gt_mask_bgr, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Thay đổi kích thước cùng chiều cao để ghép
        h = bgr_image.shape[0]
        pred_mask_bgr = cv2.resize(pred_mask_bgr, (pred_mask_bgr.shape[1], h))
        gt_mask_bgr = cv2.resize(gt_mask_bgr, (gt_mask_bgr.shape[1], h))
        
        concat_img = np.hstack((bgr_image, pred_mask_bgr, gt_mask_bgr))
        save_path = os.path.join(error_cases_dir, f"{model_type}_error_{img_name}")
        cv2.imwrite(save_path, concat_img)
        
    print(f"\n[5 error illustrative images saved to: {error_cases_dir}]")

if __name__ == "__main__":
    TEST_IMG_DIR = r"D:\Computer Vision Final Project\Src code\data\data_classification\ISIC2018_Task1-2_Test_Input\ISIC2018_Task1-2_Test_Input"
    TEST_GT_DIR = r"D:\Computer Vision Final Project\Src code\data\data_classification\ISIC2018_Task1_Test_GroundTruth\ISIC2018_Task1_Test_GroundTruth"
    
    print("==================================================")
    print("1. EVALUATING U-NET MODEL")
    print("==================================================")
    evaluate_pipeline(TEST_IMG_DIR, TEST_GT_DIR, model_type="unet")
    
    print("\n==================================================")
    print("2. EVALUATING TRADITIONAL PIPELINE (K-MEANS + SNAKES)")
    print("==================================================")
    evaluate_pipeline(TEST_IMG_DIR, TEST_GT_DIR, model_type="traditional")