import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from src.u_net.inference import UNetInferencer
from src.segmentation import get_kmeans_mask, get_snakes_mask

def plot_bar_chart(unet_csv, trad_csv, output_dir):
    df_unet = pd.read_csv(unet_csv)
    df_trad = pd.read_csv(trad_csv)
    
    metrics = ['IoU', 'Dice', 'Accuracy', 'Sensitivity', 'Specificity']
    
    
    unet_means = df_unet[metrics].mean().values
    trad_means = df_trad[metrics].mean().values
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, unet_means, width, label='U-Net (Deep Learning)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, trad_means, width, label='K-Means + Snakes (Traditional)', color='#ff7f0e')
    
    ax.set_ylabel('Scores')
    ax.set_title('Performance Comparison: U-Net vs Traditional Pipeline')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "grouped_bar_chart.png"), dpi=300)
    plt.close()
    print("[Module 1] Grouped Bar Chart saved successfully.")

def plot_boxplot(unet_csv, trad_csv, output_dir):
    df_unet = pd.read_csv(unet_csv)
    df_trad = pd.read_csv(trad_csv)
    
    df_unet['Model'] = 'U-Net'
    df_trad['Model'] = 'K-Means + Snakes'
    
    combined_df = pd.concat([df_unet[['IoU', 'Model']], df_trad[['IoU', 'Model']]])
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Model', y='IoU', hue='Model', data=combined_df, palette=['#1f77b4', '#ff7f0e'], legend=False)
    plt.title('IoU Distribution Boxplot (Stability Analysis)')
    plt.ylabel('IoU Score')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "iou_boxplot.png"), dpi=300)
    plt.close()
    print("[Module 1] Boxplot saved successfully.")

def run_error_analysis(trad_csv, img_dir, gt_dir, output_dir):
    error_cases_dir = os.path.join(output_dir, "Error_Cases")
    os.makedirs(error_cases_dir, exist_ok=True)
    
    df_trad = pd.read_csv(trad_csv)
    
    worst_3 = df_trad.sort_values(by='IoU', ascending=True).head(3)
    
    
    unet_weight = r"D:\Computer Vision Final Project\Src code\best_model.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inferencer = UNetInferencer(unet_weight, device)
    
    print("[Module 2] Extracting top 3 worst cases...")
    for idx, row in worst_3.iterrows():
        img_name = row['Image_ID']
        print(f"Processing {img_name} (IoU = {row['IoU']:.4f})")
        
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(gt_dir, img_name.replace(".jpg", "_segmentation.png"))
        
        bgr_image = cv2.imread(img_path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if bgr_image is None or gt_mask is None:
            continue
            
        h_orig, w_orig = bgr_image.shape[:2]
        
        
        unet_mask = inferencer.predict(bgr_image)
        
        
        small_bgr = cv2.resize(bgr_image, (256, 256))
        small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
        init_mask = get_kmeans_mask(small_bgr, k=3)
        small_pred_mask = get_snakes_mask(small_gray, init_mask)
        trad_mask = cv2.resize(small_pred_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        
        axs[0].imshow(rgb_image)
        axs[0].set_title('Input Image')
        axs[0].axis('off')
        
        axs[1].imshow(gt_mask, cmap='gray')
        axs[1].set_title('Ground Truth (Doctor)')
        axs[1].axis('off')
        
        axs[2].imshow(trad_mask, cmap='gray')
        axs[2].set_title(f'Traditional Mask\n(IoU = {row["IoU"]:.3f})')
        axs[2].axis('off')
        
        axs[3].imshow(unet_mask, cmap='gray')
        axs[3].set_title('U-Net Prediction')
        axs[3].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(error_cases_dir, f"Comparison_Worst_{img_name.replace('.jpg', '')}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    print(f"[Module 2] Error images saved to {error_cases_dir}")

if __name__ == "__main__":
    RESULTS_DIR = r"D:\Computer Vision Final Project\Src code\evaluation_results"
    UNET_CSV = os.path.join(RESULTS_DIR, "unet_full_results.csv")
    TRAD_CSV = os.path.join(RESULTS_DIR, "traditional_full_results.csv")
    
    TEST_IMG_DIR = r"D:\Computer Vision Final Project\Src code\data\data_classification\ISIC2018_Task1-2_Test_Input\ISIC2018_Task1-2_Test_Input"
    TEST_GT_DIR = r"D:\Computer Vision Final Project\Src code\data\data_classification\ISIC2018_Task1_Test_GroundTruth\ISIC2018_Task1_Test_GroundTruth"
    
    
    plot_bar_chart(UNET_CSV, TRAD_CSV, RESULTS_DIR)
    plot_boxplot(UNET_CSV, TRAD_CSV, RESULTS_DIR)
    
    
    run_error_analysis(TRAD_CSV, TEST_IMG_DIR, TEST_GT_DIR, RESULTS_DIR)
