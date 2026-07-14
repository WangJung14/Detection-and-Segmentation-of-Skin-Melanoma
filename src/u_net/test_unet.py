import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os

# Khai báo kiến trúc mô hình
from src.u_net.unet_model import UNet


def test_single_image():
    # 1. Cấu hình thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Khởi tạo mô hình và tải trọng số (bộ não) đã train
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load("unet_toy_weights.pth", map_location=device))
    model.eval()  # Chuyển mô hình sang chế độ Đánh giá (Không học thêm nữa)

    # 3. Chọn 1 tấm ảnh để test (Em trỏ đường dẫn tuyệt đối vào 1 tấm ảnh bất kỳ trong tập Toy Data)
    img_path = "../../data/toy_data/melanoma/ISIC_0000074.jpg"
    mask_path = "../../data/toy_data/ground_truth/ISIC_0000074_segmentation.png"

    # Đọc ảnh
    original_img = Image.open(img_path).convert("RGB")
    ground_truth = Image.open(mask_path).convert("L")

    # Ép kiểu cho giống hệt lúc train
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    input_tensor = transform(original_img).unsqueeze(0).to(device)  # Thêm 1 chiều (batch_size=1) ở đầu

    # 4. Yêu cầu AI dự đoán (Không tính toán đạo hàm cho nhẹ máy)
    with torch.no_grad():
        output = model(input_tensor)

        # Hàm Sigmoid nén giá trị về khoảng [0, 1]
        probs = torch.sigmoid(output)

        # Cắt ngưỡng 0.5: Điểm ảnh nào > 0.5 thì cho là U (1), ngược lại là Da (0)
        predicted_mask = (probs > 0.5).float().cpu().squeeze().numpy()

    # 5. Trực quan hóa kết quả để đưa lên Slide
    fig, arr = plt.subplots(1, 3, figsize=(15, 5))

    arr[0].imshow(original_img.resize((256, 256)))
    arr[0].set_title("1. Ảnh gốc (Input)")
    arr[0].axis("off")

    arr[1].imshow(ground_truth.resize((256, 256)), cmap="gray")
    arr[1].set_title("2. Đáp án của Bác sĩ (Ground Truth)")
    arr[1].axis("off")

    arr[2].imshow(predicted_mask, cmap="gray")
    arr[2].set_title("3. U-Net Dự đoán (Predicted Mask)")
    arr[2].axis("off")

    plt.tight_layout()
    plt.savefig("unet_inference_result.png")
    print("✅ Đã xuất ảnh so sánh thành công: unet_inference_result.png")
    plt.show()


if __name__ == "__main__":
    test_single_image()