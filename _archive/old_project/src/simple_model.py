"""
Simple Siamese UNet for Change Detection
Author: Ninh Hai Dang (21021411)
Date: 2025-10-17

A lightweight UNet-based change detection model that works with 9-channel inputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, skip_channels=None):
        super().__init__()
        if skip_channels is None:
            skip_channels = in_channels
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SimpleSiameseUNet(nn.Module):
    """
    Simple Siamese UNet for change detection

    Takes 9-channel images from T1 and T2, concatenates them into 18 channels,
    and outputs binary change mask.

    Args:
        in_channels (int): Number of input channels per time step (default: 9)
        num_classes (int): Number of output classes (default: 2 for binary)
    """
    def __init__(self, in_channels=9, num_classes=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Input: 18 channels (9 from T1 + 9 from T2)
        self.inc = DoubleConv(in_channels * 2, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(512, 256, skip_channels=512)  # x5: 512, x4: 512
        self.up2 = Up(256, 128, skip_channels=256)  # from up1: 256, x3: 256
        self.up3 = Up(128, 64, skip_channels=128)   # from up2: 128, x2: 128
        self.up4 = Up(64, 64, skip_channels=64)     # from up3: 64, x1: 64

        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 18, H, W)
                where 18 = 9 channels from T1 + 9 channels from T2

        Returns:
            torch.Tensor: Output logits of shape (B, num_classes, H, W)
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


def test_model():
    """Test the model with dummy data"""
    model = SimpleSiameseUNet(in_channels=9, num_classes=2)

    # Dummy input: batch_size=2, 18 channels, 256x256
    x = torch.randn(2, 18, 256, 256)

    # Forward pass
    output = model(x)

    print("Model test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert output.shape == (2, 2, 256, 256), f"Expected (2, 2, 256, 256), got {output.shape}"
    print("  ✓ Model test passed!")


if __name__ == '__main__':
    test_model()
