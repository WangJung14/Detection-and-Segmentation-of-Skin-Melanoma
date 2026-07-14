import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image


class SkinCancerDataset(Dataset):
    """
    Dataset dành cho Skin Lesion Segmentation (ISIC)

    Parameters
    ----------
    image_dir : str
        Thư mục chứa ảnh RGB

    mask_dir : str
        Thư mục chứa Ground Truth

    train : bool
        True -> bật Data Augmentation
        False -> chỉ Resize + Normalize
    """

    def __init__(self,
                 image_dir,
                 mask_dir,
                 train=True,
                 image_size=256):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train = train
        self.image_size = image_size

        self.images = sorted([
            file for file in os.listdir(image_dir)
            if file.endswith(".jpg")
        ])

        # Mean / Std của ImageNet
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_name = self.images[index]

        image_path = os.path.join(self.image_dir, img_name)

        mask_name = img_name.replace(
            ".jpg",
            "_segmentation.png"
        )

        mask_path = os.path.join(
            self.mask_dir,
            mask_name
        )

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        ############################################
        # Resize
        ############################################

        image = TF.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.BILINEAR
        )

        mask = TF.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.NEAREST
        )

        ############################################
        # Data Augmentation
        ############################################

        if self.train:

            # Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Vertical Flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random Rotation
            angle = random.uniform(-20, 20)

            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR
            )

            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST
            )

            # Brightness
            brightness = random.uniform(0.9, 1.1)
            image = TF.adjust_brightness(image, brightness)

            # Contrast
            contrast = random.uniform(0.9, 1.1)
            image = TF.adjust_contrast(image, contrast)

        ############################################
        # To Tensor
        ############################################

        image = TF.to_tensor(image)

        ############################################
        # Normalize
        ############################################

        image = TF.normalize(
            image,
            mean=self.mean,
            std=self.std
        )

        ############################################
        # Mask
        ############################################

        mask = TF.to_tensor(mask)

        mask = (mask > 0.5).float()

        return image, mask