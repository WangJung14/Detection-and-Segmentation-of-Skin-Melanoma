import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_comparison_chart():
    # File paths
    eval_dir = r"d:\Computer Vision Final Project\Src code\evaluation_results"
    trad_file = os.path.join(eval_dir, "traditional_full_results.csv")
    unet_file = os.path.join(eval_dir, "unet_full_results.csv")
    output_file = os.path.join(eval_dir, "iou_comparison_10_90.png")

    # Load data
    df_trad = pd.read_csv(trad_file)
    df_unet = pd.read_csv(unet_file)

    # Filter data for 10% < IoU < 90%
    trad_iou = df_trad[(df_trad['IoU'] > 0.1) & (df_trad['IoU'] < 0.9)]['IoU']
    unet_iou = df_unet[(df_unet['IoU'] > 0.1) & (df_unet['IoU'] < 0.9)]['IoU']

    # Define bins
    bins = np.linspace(0.1, 0.9, 9) # 0.1 to 0.9 with 0.1 step
    
    # Calculate histograms
    trad_hist, _ = np.histogram(trad_iou, bins=bins)
    unet_hist, _ = np.histogram(unet_iou, bins=bins)

    # Plot
    x = np.arange(len(bins) - 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, trad_hist, width, label='Traditional')
    rects2 = ax.bar(x + width/2, unet_hist, width, label='U-Net')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of Images')
    ax.set_xlabel('IoU Range')
    ax.set_title('IoU Comparison (10% < IoU < 90%)')
    
    # Format x-ticks
    bin_labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(len(bins)-1)]
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend()

    # Add labels on top of bars
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    # Save plot
    plt.savefig(output_file)
    print(f"Chart saved to {output_file}")

if __name__ == "__main__":
    generate_comparison_chart()
