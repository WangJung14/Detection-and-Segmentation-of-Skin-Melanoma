import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from src.preprocessing import apply_circular_mask, remove_hair, enhance_contrast_clahe, boost_faint_edges
from src.segmentation import get_kmeans_mask, get_snakes_mask
from src.evaluation import calculate_iou

def draw_contour(image, mask, color):
    res = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(res, contours, -1, color, 2)
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

def sweep_dullrazor(img):
    """Khảo sát kích thước chổi xóa lông (kernel_size)"""
    kernels = [5, 15, 25]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    img_no_corners = apply_circular_mask(img, 0.85)
    
    for i, k in enumerate(kernels):
        clean_img, _ = remove_hair(img_no_corners, kernel_size=k, inpaint_rad=3)
        axs[i].imshow(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))
        axs[i].set_title(f"DullRazor (kernel_size = {k})")
        axs[i].axis('off')
    return fig

def sweep_clahe(img):
    """Khảo sát độ gắt tương phản CLAHE (clip_limit)"""
    limits = [1.0, 1.1, 3.0]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    img_no_corners = apply_circular_mask(img, 0.85)
    clean_img, _ = remove_hair(img_no_corners, kernel_size=15, inpaint_rad=3)
    
    for i, clip in enumerate(limits):
        _, clahe_gray = enhance_contrast_clahe(clean_img, clip_limit=clip)
        axs[i].imshow(clahe_gray, cmap='gray')
        axs[i].set_title(f"CLAHE (clip_limit = {clip})")
        axs[i].axis('off')
    return fig

def sweep_otsu(img, gt_mask):
    """Khảo sát Morphological dọn rác cho mặt nạ Otsu"""
    open_kernels = [0, 5, 11]
    close_kernels = [0, 15, 31]
    
    img_no_corners = apply_circular_mask(img, 0.85)
    clean_img, _ = remove_hair(img_no_corners, kernel_size=15, inpaint_rad=3)
    _, clahe_gray = enhance_contrast_clahe(clean_img, clip_limit=1.1)
    boosted_l, _ = boost_faint_edges(clahe_gray)
    smooth_l = cv2.GaussianBlur(boosted_l, (5, 5), 0)
    smooth_l = apply_circular_mask(smooth_l, 0.85)
    
    _, base_otsu = cv2.threshold(smooth_l, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    fig, axs = plt.subplots(len(open_kernels), len(close_kernels), figsize=(15, 12))
    
    for i, k_open in enumerate(open_kernels):
        for j, k_close in enumerate(close_kernels):
            mask = base_otsu.copy()
            if k_open > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            if k_close > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
                
            iou = calculate_iou(mask, gt_mask)
            ax = axs[i, j]
            ax.imshow(draw_contour(img, mask, (255, 0, 0)))
            ax.set_title(f"Open:{k_open}, Close:{k_close} (IoU:{iou*100:.1f}%)")
            ax.axis('off')
    return fig

def sweep_kmeans(img, gt_mask):
    """Khảo sát số cụm màu (K) trong K-Means"""
    k_values = [2, 3, 4, 6]
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    img_no_corners = apply_circular_mask(img, 0.85)
    clean_img, _ = remove_hair(img_no_corners, kernel_size=15, inpaint_rad=3)
    clahe_color, _ = enhance_contrast_clahe(clean_img, clip_limit=1.1)
    clahe_color = apply_circular_mask(clahe_color, 0.85)
    
    for i, k in enumerate(k_values):
        mask = get_kmeans_mask(clahe_color, k=k)
        iou = calculate_iou(mask, gt_mask)
        axs[i].imshow(draw_contour(img, mask, (0, 255, 0)))
        axs[i].set_title(f"K = {k} (IoU: {iou*100:.1f}%)")
        axs[i].axis('off')
    return fig

def sweep_snakes(img, gt_mask):
    """Khảo sát số vòng lặp (num_iter) của Snakes"""
    iters = [5, 35, 100]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    img_no_corners = apply_circular_mask(img, 0.85)
    clean_img, _ = remove_hair(img_no_corners, kernel_size=15, inpaint_rad=3)
    clahe_color, clahe_gray = enhance_contrast_clahe(clean_img, clip_limit=1.1)
    boosted_l, _ = boost_faint_edges(clahe_gray)
    smooth_l = cv2.GaussianBlur(boosted_l, (5, 5), 0)
    smooth_l = apply_circular_mask(smooth_l, 0.85)
    clahe_color = apply_circular_mask(clahe_color, 0.85)
    
    kmeans_mask = get_kmeans_mask(clahe_color, k=4)
    
    for i, it in enumerate(iters):
        mask = get_snakes_mask(smooth_l, kmeans_mask, num_iter=it)
        iou = calculate_iou(mask, gt_mask)
        axs[i].imshow(draw_contour(img, mask, (0, 255, 255)))
        axs[i].set_title(f"Snakes Iterations = {it} (IoU: {iou*100:.1f}%)")
        axs[i].axis('off')
    return fig

def sweep_abcd_compactness(img, mask):
    """Khảo sát độ nhạy của thuật toán chấm điểm B (Border / Compactness)"""
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    A = cv2.contourArea(cnt)
    P = cv2.arcLength(cnt, True)
    compactness = (4 * math.pi * A) / (P ** 2)
    
    multipliers = [5, 20, 50]
    print(f"Hệ số Compactness thực tế của khối u này là: {compactness:.3f}")
    
    for m in multipliers:
        score = int(round(m * (1 - compactness)))
        score = min(max(score, 0), 8)
        print(f"-> Nếu dùng hệ số nhạy = {m}: Điểm chữ B sẽ là {score}/8 điểm.")