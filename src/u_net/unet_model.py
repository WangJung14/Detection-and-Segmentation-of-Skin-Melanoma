import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Khối tích chập lặp lại 2 lần (Double Convolution).
    Mỗi bước bao gồm: Conv2d -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Kiến trúc U-Net hoàn chỉnh cho phân đoạn ảnh y tế.
    """

    def __init__(self, in_channels=3, out_channels=1):
        # in_channels=3 (Ảnh màu RGB)
        # out_channels=1 (Mặt nạ nhị phân: 0 là da nền, 1 là khối u)
        super(UNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1. Nhánh xuống (Encoder)
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        # Lõi đáy (Bottleneck)
        self.bottleneck = DoubleConv(512, 1024)

        # 2. Nhánh lên (Decoder) + Phép phóng to (Up-convolution)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)  # 1024 = 512 (từ dưới lên) + 512 (từ Skip Connection)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)

        # 3. Lớp đầu ra (Output Layer)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # MẠN TRÁI: ĐI XUỐNG
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        # ĐÁY CHỮ U
        x_bottleneck = self.bottleneck(self.pool(x4))

        # MẠN PHẢI: ĐI LÊN (Kết hợp Skip Connections)
        x = self.up1(x_bottleneck)
        x = torch.cat([x, x4], dim=1)  # Cầu nối dữ liệu
        x = self.up_conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv4(x)

        # ĐẦU RA
        return self.out_conv(x)