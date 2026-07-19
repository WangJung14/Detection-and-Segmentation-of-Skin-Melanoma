import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SkinCancerDataset(Dataset):
    """
    Dataset dành cho Skin Lesion Segmentation (ISIC)
    Sử dụng thư viện Albumentations để thực hiện data augmentation đồng bộ
    giữa ảnh (RGB) và mặt nạ (Ground Truth Mask).
    
    Parameters
    ----------
    image_dir : str
        Thư mục chứa ảnh RGB (.jpg)
    mask_dir : str
        Thư mục chứa Ground Truth (.png)
    train : bool
        True -> Bật Data Augmentation (Tập Training)
        False -> Chỉ Resize và Chuẩn hóa (Tập Validation/Test)
    image_size : int
        Kích thước ảnh đầu ra (mặc định 256x256)
    """

    def __init__(self, image_dir, mask_dir, train=True, image_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train = train
        self.image_size = image_size

        valid_images = []
        for file in sorted(os.listdir(image_dir)):
            if file.endswith(".jpg"):
                mask_name = file.replace(".jpg", "_segmentation.png")
                if os.path.exists(os.path.join(mask_dir, mask_name)):
                    valid_images.append(file)
                else:
                    print(f"⚠️ Cảnh báo: Ảnh {file} không có Mask đi kèm. Đã bỏ qua!")
        self.images = valid_images

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.train_transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=180, p=0.8),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
        
        self.val_transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        image_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace(".jpg", "_segmentation.png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        transform = self.train_transform if self.train else self.val_transform
        augmented = transform(image=image, mask=mask)
        
        image = augmented['image']
        mask = augmented['mask']

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
            
        if mask.max() > 1:
            mask = (mask > 127).float()
        else:
            mask = (mask > 0.5).float()

        return image, mask