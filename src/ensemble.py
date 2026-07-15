import numpy as np

def majority_voting(*masks, threshold=None):
    """
    Kết hợp nhiều mặt nạ (masks) dựa trên cơ chế Bầu chọn đa số (Majority Voting).
    
    Parameters:
    - masks: danh sách các numpy array (H, W) mang giá trị 0 hoặc 255.
    - threshold: số phiếu tối thiểu để pixel được coi là u (Foreground). 
                 Nếu None, tự động tính bằng ceil(len(masks) / 2).
                 Ví dụ: 3 voters -> threshold = 2.
                        4 voters -> threshold = 3.
    """
    if len(masks) == 0:
        raise ValueError("Yêu cầu ít nhất 1 mask để thực hiện voting.")
        
    # Chuẩn hóa về dạng 0 và 1 để dễ cộng
    binary_masks = [(m > 127).astype(np.uint8) for m in masks]
    
    # Chồng các mask lại thành mảng 3D
    stacked = np.stack(binary_masks, axis=0)
    
    # Cộng dồn số phiếu theo trục Z
    votes = np.sum(stacked, axis=0)
    
    # Xác định số phiếu cần thiết để thắng
    if threshold is None:
        threshold = len(masks) // 2 + 1 if len(masks) % 2 != 0 else len(masks) // 2
        
    # Tạo mask cuối cùng
    final_mask = (votes >= threshold).astype(np.uint8) * 255
    return final_mask
