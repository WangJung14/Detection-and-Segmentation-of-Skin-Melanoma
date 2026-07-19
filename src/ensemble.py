import numpy as np

def majority_voting(*masks, threshold=None):
    """
    Kết hợp nhiều mặt nạ (masks) dựa trên cơ chế Bầu chọn đa số (Majority Voting).
    
    Parameters:
    - masks: danh sách các numpy array (H, W) mang giá trị 0 hoặc 255.
    - threshold: số phiếu tối thiểu để pixel được coi là u (Foreground). 
    """
    if len(masks) == 0:
        raise ValueError("Yêu cầu ít nhất 1 mask để thực hiện voting.")
        
    binary_masks = [(m > 127).astype(np.uint8) for m in masks]
    stacked = np.stack(binary_masks, axis=0)
    votes = np.sum(stacked, axis=0)
    
    if threshold is None:
        threshold = len(masks) // 2 + 1 if len(masks) % 2 != 0 else len(masks) // 2
        
    final_mask = (votes >= threshold).astype(np.uint8) * 255
    return final_mask
