import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from src.u_net.unet_model import UNet

class UNetInferencer:
    def __init__(self, weight_path="best_model.pth", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model = UNet(in_channels=3, out_channels=1).to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()
        
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def predict(self, bgr_image):
        """
        Nhận vào ảnh BGR từ OpenCV, dự đoán và trả về mask nhị phân (0 và 255) kích thước gốc.
        """
        
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)
        
        
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.sigmoid(output)
            
            mask_256 = (probs > 0.5).float().cpu().squeeze().numpy()
            
        
        h, w = bgr_image.shape[:2]
        mask_original_size = cv2.resize(mask_256, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return (mask_original_size * 255).astype(np.uint8)
