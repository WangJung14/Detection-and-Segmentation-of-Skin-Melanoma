import os
import sys
import torch
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath('.'))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.u_net.dataset import SkinCancerDataset

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Hàm khôi phục ảnh về dạng chuẩn [0, 1] từ dạng đã chuẩn hóa ImageNet
    để matplotlib có thể hiển thị màu sắc chính xác.
    """
    t = tensor.clone()
    for channel, m, s in zip(t, mean, std):
        channel.mul_(s).add_(m)
    return torch.clamp(t, 0, 1)

def main():
    image_dir = r"D:\Computer Vision Final Project\Src code\data\data_classification\ISIC2018_Task1-2_Training_Input\ISIC2018_Task1-2_Training_Input"
    mask_dir = r"D:\Computer Vision Final Project\Src code\data\data_classification\ISIC2018_Task1_Training_GroundTruth\ISIC2018_Task1_Training_GroundTruth"
    
    
    dataset = SkinCancerDataset(image_dir=image_dir, mask_dir=mask_dir, train=True, image_size=256)
    
    print(f"Tổng số ảnh trong dataset: {len(dataset)}")
    if len(dataset) == 0:
        print("Lỗi: Không tìm thấy ảnh")
        return

    
    num_samples = min(10, len(dataset))
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3 * num_samples))
    fig.suptitle("Unit Test: Khảo sát 10 Cặp Ảnh & Mask Sau Khi Data Augmentation (Albumentations)", fontsize=14, y=0.99)

    for i in range(num_samples):
        
        image, mask = dataset[i]
        
        
        image_img = denormalize(image).permute(1, 2, 0).numpy()
        mask_img = mask.squeeze(0).numpy()
        
        
        unique_vals = torch.unique(mask)
        print(f"Mẫu {i+1} | Giá trị duy nhất trong Mask: {unique_vals.tolist()} | Kiểu dữ liệu: {mask.dtype}")
        
        
        axes[i, 0].imshow(image_img)
        axes[i, 0].set_title(f"Ảnh Augmented {i+1}")
        axes[i, 0].axis("off")
        
        
        axes[i, 1].imshow(mask_img, cmap="gray")
        axes[i, 1].set_title(f"Mask Augmented {i+1} (Nhị phân 0-255)")
        axes[i, 1].axis("off")
        
    plt.tight_layout()
    output_path = "augmented_samples_test.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"\n[PASS] Đã tạo thành công biểu đồ so sánh và lưu tại: {os.path.abspath(output_path)}")
    print("Vui lòng mở file 'augmented_samples_test.png' để kiểm tra độ khớp khít của ảnh và mask!")

if __name__ == "__main__":
    main()
