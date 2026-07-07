import numpy as np


def calculate_iou(mask_pred, mask_gt):
    """
    Tính toán chỉ số Intersection over Union (IoU) giữa ảnh dự đoán và đáp án.
    Trả về giá trị từ 0.0 đến 1.0.
    """
    # 1. Đảm bảo 2 mask có cùng kích thước
    if mask_pred.shape != mask_gt.shape:
        raise ValueError(f"Kích thước không khớp! Pred: {mask_pred.shape} khác GT: {mask_gt.shape}")

    # 2. Nhị phân hóa an toàn (Chỉ giữ lại True và False)
    # Lớn hơn 127 là vùng trắng (khối u), nhỏ hơn là nền đen
    pred_bin = (mask_pred > 127).astype(np.bool_)
    gt_bin = (mask_gt > 127).astype(np.bool_)

    # 3. Tính phần Giao (Intersection - Điểm đồng thuận) bằng logic AND
    intersection = np.logical_and(pred_bin, gt_bin).sum()

    # 4. Tính phần Hợp (Union - Tổng diện tích bao phủ) bằng logic OR
    union = np.logical_or(pred_bin, gt_bin).sum()

    # 5. Xử lý chia cho 0 (Trường hợp ảnh hoàn toàn không có khối u)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    # 6. Ra kết quả
    iou = intersection / union
    return iou