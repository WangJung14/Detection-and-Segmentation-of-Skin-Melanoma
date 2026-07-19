import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

def compute_asymmetry(mask):
    """
    Computes Asymmetry (A) score.
    Returns bucketized score: 0 (symmetric), 1 (asymmetric in 1 axis), 2 (asymmetric in 2 axes).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return 0
    
    (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
    
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return 0
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    mask_rotated = cv2.warpAffine(mask, cv2.getRotationMatrix2D((cx, cy), 180, 1.0), mask.shape[::-1])
    diff = cv2.absdiff(mask, mask_rotated)
    non_overlap_ratio = np.sum(diff > 0) / (np.sum(mask > 0) + 1e-5)
    
    if non_overlap_ratio < 0.1:
        return 0
    elif non_overlap_ratio < 0.3:
        return 1
    else:
        return 2

def compute_border(mask):
    """
    Computes Border (B) score based on compactness.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    if area == 0:
        return 0
        
    compactness = (perimeter ** 2) / (4 * np.pi * area)
    
    b_score = int(min((compactness - 1.0), 8.0))
    return max(0, b_score)

def compute_color(image, mask):
    """
    Computes Color (C) score based on standard deviation across RGB channels.
    """
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked_img[mask > 0]
    
    if len(pixels) == 0:
        return 1
        
    std_r = np.std(pixels[:, 0])
    std_g = np.std(pixels[:, 1])
    std_b = np.std(pixels[:, 2])
    
    total_std = std_r + std_g + std_b
    c_score = int((total_std / 150.0) * 6) + 1
    c_score = min(max(c_score, 1), 6)
    return c_score

def compute_diameter(mask):
    """
    Computes Diameter (D) score based on bounding box diagonal.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    diameter = np.sqrt(w**2 + h**2)
    
    d_score = int((diameter / 200.0)) + 1
    d_score = min(max(d_score, 1), 5)
    return d_score

def extract_features(img_dir, mask_dir, output_csv):
    """
    Iterates through image and mask directories to extract ABCD features.
    """
    results = []
    
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    print(f"Found {len(img_files)} images in {img_dir}. Starting extraction...")
    
    for img_name in tqdm(img_files):
        img_id = img_name.replace('.jpg', '')
        img_path = os.path.join(img_dir, img_name)
        mask_name = f"{img_id}_segmentation.png"
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            continue
            
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        a_score = compute_asymmetry(mask)
        b_score = compute_border(mask)
        c_score = compute_color(img, mask)
        d_score = compute_diameter(mask)
        
        results.append({
            'image_id': img_id,
            'A_score': a_score,
            'B_score': b_score,
            'C_score': c_score,
            'D_score': d_score
        })
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--mask_dir', required=True)
    parser.add_argument('--output_csv', required=True)
    args = parser.parse_args()
    
    extract_features(args.img_dir, args.mask_dir, args.output_csv)
