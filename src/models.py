"""
CNN architectures for deforestation detection

This module implements 3 shallow CNN architectures:
1. SpatialContextCNN (~30K parameters) - Simple spatial smoothing
2. MultiScaleCNN (~80K parameters) - Multi-scale feature learning
3. ShallowUNet (~120K parameters) - Encoder-decoder with skip connections
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpatialContextCNN(nn.Module):
    """
    Spatial Context CNN - Simplest architecture

    Architecture:
        - Conv 3x3 (32 filters)
        - Conv 3x3 (32 filters)
        - Conv 1x1 (1 filter) + Sigmoid

    Parameters: ~30,000
    Receptive field: 5x5 pixels (50m x 50m)

    Args:
        in_channels: Number of input channels (default: 16 - VH only from S1)

    Example:
        >>> model = SpatialContextCNN(in_channels=16)
        >>> x = torch.randn(1, 16, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([1, 1, 128, 128])
    """

    def __init__(self, in_channels: int = 16):
        super(SpatialContextCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, 16, 128, 128)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)  # (B, 32, 128, 128)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)  # (B, 32, 128, 128)

        # Output layer (logits, no sigmoid - will use BCEWithLogitsLoss)
        x = self.conv3(x)  # (B, 1, 128, 128)

        return x


class MultiScaleCNN(nn.Module):
    """
    Multi-Scale CNN - Learns features at multiple scales

    Architecture:
        - Branch 1: Conv 3x3 (32 filters)
        - Branch 2: Conv 5x5 (32 filters)
        - Concatenate: 64 channels
        - Conv 3x3 (64 filters)
        - Conv 3x3 (64 filters)
        - Conv 1x1 (1 filter) + Sigmoid

    Parameters: ~80,000
    Receptive field: Branch 1 (7x7), Branch 2 (9x9)

    Args:
        in_channels: Number of input channels (default: 16 - VH only from S1)

    Example:
        >>> model = MultiScaleCNN(in_channels=16)
        >>> x = torch.randn(1, 16, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([1, 1, 128, 128])
    """

    def __init__(self, in_channels: int = 16):
        super(MultiScaleCNN, self).__init__()

        # Branch 1: Small receptive field (3x3)
        self.branch1_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.branch1_bn = nn.BatchNorm2d(32)

        # Branch 2: Large receptive field (5x5)
        self.branch2_conv = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.branch2_bn = nn.BatchNorm2d(32)

        # Fusion layers (after concatenation: 64 channels)
        self.fusion_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fusion_bn1 = nn.BatchNorm2d(64)

        self.fusion_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fusion_bn2 = nn.BatchNorm2d(64)

        # Output layer
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, 16, 128, 128)

        # Branch 1: 3x3 convolution
        branch1 = self.branch1_conv(x)
        branch1 = self.branch1_bn(branch1)
        branch1 = F.relu(branch1)  # (B, 32, 128, 128)

        # Branch 2: 5x5 convolution
        branch2 = self.branch2_conv(x)
        branch2 = self.branch2_bn(branch2)
        branch2 = F.relu(branch2)  # (B, 32, 128, 128)

        # Concatenate branches
        x = torch.cat([branch1, branch2], dim=1)  # (B, 64, 128, 128)

        # Fusion layer 1
        x = self.fusion_conv1(x)
        x = self.fusion_bn1(x)
        x = F.relu(x)  # (B, 64, 128, 128)

        # Fusion layer 2
        x = self.fusion_conv2(x)
        x = self.fusion_bn2(x)
        x = F.relu(x)  # (B, 64, 128, 128)

        # Output layer (logits, no sigmoid - will use BCEWithLogitsLoss)
        x = self.output_conv(x)  # (B, 1, 128, 128)

        return x


class ShallowUNet(nn.Module):
    """
    Shallow U-Net - Encoder-decoder with skip connections

    Architecture:
        Encoder:
            - Conv 3x3 x2 (32 filters) -> MaxPool 2x2
            - Conv 3x3 x2 (64 filters) -> MaxPool 2x2
        Bottleneck:
            - Conv 3x3 x2 (128 filters)
        Decoder:
            - Upsample 2x2 -> Concat -> Conv 3x3 x2 (64 filters)
            - Upsample 2x2 -> Concat -> Conv 3x3 x2 (32 filters)
            - Conv 1x1 (1 filter) + Sigmoid

    Parameters: ~120,000
    Receptive field: 13x13 pixels (130m x 130m)

    Args:
        in_channels: Number of input channels (default: 16 - VH only from S1)

    Example:
        >>> model = ShallowUNet(in_channels=16)
        >>> x = torch.randn(1, 16, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([1, 1, 128, 128])
    """

    def __init__(self, in_channels: int = 16):
        super(ShallowUNet, self).__init__()

        # Encoder
        self.enc1_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.enc1_bn1 = nn.BatchNorm2d(32)
        self.enc1_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.enc1_bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2_conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc2_bn1 = nn.BatchNorm2d(64)
        self.enc2_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.enc2_bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bottleneck_bn1 = nn.BatchNorm2d(128)
        self.bottleneck_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bottleneck_bn2 = nn.BatchNorm2d(128)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1_conv1 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.dec1_bn1 = nn.BatchNorm2d(64)
        self.dec1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec1_bn2 = nn.BatchNorm2d(64)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2_conv1 = nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1)
        self.dec2_bn1 = nn.BatchNorm2d(32)
        self.dec2_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dec2_bn2 = nn.BatchNorm2d(32)

        # Output layer
        self.output_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, 16, 128, 128)

        # Encoder 1
        enc1 = self.enc1_conv1(x)
        enc1 = self.enc1_bn1(enc1)
        enc1 = F.relu(enc1)
        enc1 = self.enc1_conv2(enc1)
        enc1 = self.enc1_bn2(enc1)
        enc1 = F.relu(enc1)  # (B, 32, 128, 128) - save for skip connection

        x = self.pool1(enc1)  # (B, 32, 64, 64)

        # Encoder 2
        enc2 = self.enc2_conv1(x)
        enc2 = self.enc2_bn1(enc2)
        enc2 = F.relu(enc2)
        enc2 = self.enc2_conv2(enc2)
        enc2 = self.enc2_bn2(enc2)
        enc2 = F.relu(enc2)  # (B, 64, 64, 64) - save for skip connection

        x = self.pool2(enc2)  # (B, 64, 32, 32)

        # Bottleneck
        x = self.bottleneck_conv1(x)
        x = self.bottleneck_bn1(x)
        x = F.relu(x)
        x = self.bottleneck_conv2(x)
        x = self.bottleneck_bn2(x)
        x = F.relu(x)  # (B, 128, 32, 32)

        # Decoder 1
        x = self.up1(x)  # (B, 128, 64, 64)
        x = torch.cat([x, enc2], dim=1)  # (B, 128+64=192, 64, 64)
        x = self.dec1_conv1(x)
        x = self.dec1_bn1(x)
        x = F.relu(x)
        x = self.dec1_conv2(x)
        x = self.dec1_bn2(x)
        x = F.relu(x)  # (B, 64, 64, 64)

        # Decoder 2
        x = self.up2(x)  # (B, 64, 128, 128)
        x = torch.cat([x, enc1], dim=1)  # (B, 64+32=96, 128, 128)
        x = self.dec2_conv1(x)
        x = self.dec2_bn1(x)
        x = F.relu(x)
        x = self.dec2_conv2(x)
        x = self.dec2_bn2(x)
        x = F.relu(x)  # (B, 32, 128, 128)

        # Output layer (logits, no sigmoid - will use BCEWithLogitsLoss)
        x = self.output_conv(x)  # (B, 1, 128, 128)

        return x


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters

    Example:
        >>> model = SpatialContextCNN()
        >>> n_params = count_parameters(model)
        >>> print(f"Parameters: {n_params:,}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(model_name: str, in_channels: int = 16) -> nn.Module:
    """
    Factory function to get model by name

    Args:
        model_name: Name of model ('spatial_cnn', 'multiscale_cnn', 'shallow_unet')
        in_channels: Number of input channels

    Returns:
        Model instance

    Example:
        >>> model = get_model('spatial_cnn', in_channels=16)
        >>> model = get_model('shallow_unet')
    """
    models = {
        'spatial_cnn': SpatialContextCNN,
        'multiscale_cnn': MultiScaleCNN,
        'shallow_unet': ShallowUNet
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    return models[model_name](in_channels=in_channels)


def print_model_summary(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 16, 128, 128)):
    """
    Print model summary

    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)

    Example:
        >>> model = get_model('spatial_cnn')
        >>> print_model_summary(model)
    """
    print("\n" + "="*80)
    print(f"MODEL: {model.__class__.__name__}")
    print("="*80)

    # Count parameters
    total_params = count_parameters(model)
    print(f"\n📊 Parameters: {total_params:,}")

    # Test forward pass
    device = next(model.parameters()).device
    x = torch.randn(input_size).to(device)

    with torch.no_grad():
        y = model(x)

    print(f"\n🔄 Input/Output:")
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(y.shape)}")

    print("\n" + "="*80)


if __name__ == "__main__":
    print("Testing models module...")

    # Test all 3 models
    models = ['spatial_cnn', 'multiscale_cnn', 'shallow_unet']

    for model_name in models:
        print(f"\n{'='*80}")
        print(f"Testing {model_name.upper()}")
        print(f"{'='*80}")

        model = get_model(model_name, in_channels=16)
        print_model_summary(model)

        # Test forward pass
        x = torch.randn(2, 16, 128, 128)  # Batch of 2
        y = model(x)

        print(f"\n✅ Forward pass successful")
        print(f"   Input:  {tuple(x.shape)}")
        print(f"   Output: {tuple(y.shape)}")
        print(f"   Output range: [{y.min():.4f}, {y.max():.4f}]")

    print("\n" + "="*80)
    print("✅ All models tested successfully")
    print("="*80)
