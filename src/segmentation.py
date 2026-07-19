import cv2
import numpy as np
from skimage.segmentation import morphological_chan_vese

def get_otsu_mask(gray_image):
    """
    Tạo mặt nạ thô bằng thuật toán Otsu và làm sạch bằng hình thái học.
    Đây sẽ là điểm xuất phát (Initial Level Set) cho thuật toán Snakes.
    """
    
    
    _, mask_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel_open)

    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_close)

    return mask_cleaned


def get_snakes_mask(gray_image, init_mask, num_iter=35):
    image_float = gray_image.astype(np.float32) / 255.0
    mask_float = (init_mask > 0).astype(np.float32)

    
    snake_evolution = morphological_chan_vese(
        image_float,
        num_iter=num_iter,
        init_level_set=mask_float,
        smoothing=1,
        lambda1=1,
        lambda2=1
    )

    
    res = snake_evolution.astype(np.float64)
    
    res = res * 255
    
    final_mask = res.astype(np.uint8)

    return final_mask


def get_kmeans_mask(color_image, k=3):
    """
    Sử dụng Machine Learning (K-Means Clustering) để gom nhóm màu sắc.
    Sử dụng Spatial Prior (Ưu tiên Không gian) thay vì Độ sáng:
    Khối u luôn nằm ở trung tâm bức ảnh. Ta sẽ lấy các cụm gần tâm nhất.
    Bao gồm màng lọc "Đảo lớn nhất" để loại bỏ các nhiễu rác xung quanh.
    """
    h, w = color_image.shape[:2]
    
    pixel_values = color_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()

    
    center_x, center_y = w // 2, h // 2
    Y, X = np.indices((h, w))
    X = X.flatten()
    Y = Y.flatten()
    
    cluster_dist = []
    for i in range(k):
        idx = (labels == i)
        
        if np.mean(centers[i]) > 240:
            cluster_dist.append(float('inf'))
            continue
            
        if np.sum(idx) == 0:
            cluster_dist.append(float('inf'))
            continue
            
        
        dist = np.sqrt((X[idx] - center_x)**2 + (Y[idx] - center_y)**2)
        cluster_dist.append(np.mean(dist))

    
    mask = np.zeros_like(labels, dtype=np.uint8)

    
    core_cluster = np.argmin(cluster_dist)
    mask[labels == core_cluster] = 255

    
    if k >= 3:
        cluster_dist[core_cluster] = float('inf') 
        margin_cluster = np.argmin(cluster_dist)
        mask[labels == margin_cluster] = 255

    
    mask = mask.reshape((h, w))

    
    
    
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    clean_mask = np.zeros_like(mask)

    if contours:
        
        largest_contour = max(contours, key=cv2.contourArea)

        
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    

    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    return final_mask