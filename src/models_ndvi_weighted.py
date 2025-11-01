"""
NDVI-Weighted CNN Models for Deforestation Detection

This module implements CNN architectures that emphasize NDVI change:
1. Channel Attention - Learn importance of each channel
2. NDVI Difference Branch - Explicit NDVI change modeling
3. Feature Weighting - Manual weight boost for NDVI channels
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation style)

    Learns importance weights for each input channel.
    This allows the model to focus more on NDVI channels.
    """
    def __init__(self, channels, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: Learn channel weights
        y = self.fc(y).view(b, c, 1, 1)
        # Scale input by learned weights
        return x * y.expand_as(x)


class NDVIDifferenceBranch(nn.Module):
    """
    Explicit NDVI Change Branch

    Computes NDVI difference and processes it separately.

    Input channels (S2 only - 14 channels):
        0-6:   Sentinel-2 2024 [B, G, R, NIR, NDVI, NBR, NDMI]
        7-13:  Sentinel-2 2025 [B, G, R, NIR, NDVI, NBR, NDMI]

    NDVI indices: 4 (2024), 11 (2025)
    """
    def __init__(self):
        super(NDVIDifferenceBranch, self).__init__()
        # Process NDVI difference with small CNN
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        """
        Args:
            x: (B, 14, H, W) - Full input tensor

        Returns:
            ndvi_features: (B, 16, H, W) - Processed NDVI change features
            ndvi_change: (B, 1, H, W) - Raw NDVI change map
        """
        # Extract NDVI channels
        ndvi_2024 = x[:, 4:5, :, :]  # (B, 1, H, W)
        ndvi_2025 = x[:, 11:12, :, :]  # (B, 1, H, W)

        # Compute NDVI change
        ndvi_change = ndvi_2025 - ndvi_2024  # (B, 1, H, W)

        # Process NDVI change
        ndvi_features = F.relu(self.bn1(self.conv1(ndvi_change)))
        ndvi_features = F.relu(self.bn2(self.conv2(ndvi_features)))

        return ndvi_features, ndvi_change


class MultiScaleCNN_NDVIWeighted(nn.Module):
    """
    Multi-Scale CNN with NDVI Emphasis

    Improvements:
    1. Channel attention to learn importance of each channel
    2. Separate NDVI difference branch
    3. Feature fusion from multiple paths

    Architecture:
        - Main branch: Multi-scale convolutions on all channels
        - NDVI branch: Dedicated processing for NDVI change
        - Channel attention: Learn channel importance weights
        - Fusion: Combine all features

    Args:
        in_channels: Number of input channels (default: 14 - S2 only)
        use_channel_attention: Enable channel attention (default: True)
        use_ndvi_branch: Enable NDVI difference branch (default: True)
    """

    def __init__(self, in_channels=14, use_channel_attention=True, use_ndvi_branch=True):
        super(MultiScaleCNN_NDVIWeighted, self).__init__()

        self.use_channel_attention = use_channel_attention
        self.use_ndvi_branch = use_ndvi_branch

        # Channel attention (optional)
        if use_channel_attention:
            self.channel_attention = ChannelAttention(in_channels, reduction=4)

        # Main multi-scale branch (same as original)
        self.conv1a = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)

        # NDVI difference branch (optional)
        if use_ndvi_branch:
            self.ndvi_branch = NDVIDifferenceBranch()
            # Adjust fusion channels
            fusion_channels = 128 + 16  # Main features + NDVI features
        else:
            fusion_channels = 128

        # Fusion and output
        self.conv3 = nn.Conv2d(fusion_channels, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, 14, 128, 128)

        Returns:
            output: (B, 1, 128, 128) - Logits (before sigmoid)
        """
        # Optional: Channel attention
        if self.use_channel_attention:
            x = self.channel_attention(x)

        # Main multi-scale branch
        x1a = F.relu(self.conv1a(x))
        x1b = F.relu(self.conv1b(x))
        x1 = torch.cat([x1a, x1b], dim=1)
        x1 = self.bn1(x1)

        x2a = F.relu(self.conv2a(x1))
        x2b = F.relu(self.conv2b(x1))
        x2 = torch.cat([x2a, x2b], dim=1)
        x2 = self.bn2(x2)

        # Optional: NDVI branch
        if self.use_ndvi_branch:
            ndvi_features, ndvi_change = self.ndvi_branch(x)
            # Fuse main features with NDVI features
            x2 = torch.cat([x2, ndvi_features], dim=1)

        # Final layers
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        output = self.conv5(x4)

        return output


class ShallowUNet_NDVIWeighted(nn.Module):
    """
    Shallow U-Net with NDVI Emphasis

    Same improvements as MultiScaleCNN:
    1. Channel attention
    2. NDVI difference branch
    3. Skip connections (U-Net style)
    """

    def __init__(self, in_channels=14, use_channel_attention=True, use_ndvi_branch=True):
        super(ShallowUNet_NDVIWeighted, self).__init__()

        self.use_channel_attention = use_channel_attention
        self.use_ndvi_branch = use_ndvi_branch

        # Channel attention
        if use_channel_attention:
            self.channel_attention = ChannelAttention(in_channels, reduction=4)

        # Encoder
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)

        self.pool = nn.MaxPool2d(2)

        # NDVI branch
        if use_ndvi_branch:
            self.ndvi_branch = NDVIDifferenceBranch()
            bottleneck_channels = 128 + 16
        else:
            bottleneck_channels = 128

        # Bottleneck
        self.bottleneck = self._conv_block(bottleneck_channels, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)

        # Output
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Channel attention
        if self.use_channel_attention:
            x = self.channel_attention(x)

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # NDVI branch
        if self.use_ndvi_branch:
            ndvi_features, _ = self.ndvi_branch(x)
            # Downsample NDVI features to match enc3
            ndvi_features_down = F.adaptive_avg_pool2d(ndvi_features, enc3.shape[2:])
            enc3 = torch.cat([enc3, ndvi_features_down], dim=1)

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))

        # Decoder
        dec3 = self.up3(bottleneck)
        dec3 = torch.cat([dec3, enc3[:, :128, :, :]], dim=1)  # Skip connection
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output
        output = self.out(dec1)
        return output


def get_ndvi_weighted_model(model_type, in_channels=14, **kwargs):
    """
    Get NDVI-weighted model by type

    Args:
        model_type: 'multiscale_cnn' or 'shallow_unet'
        in_channels: Number of input channels (default: 14)
        **kwargs: Additional arguments (use_channel_attention, use_ndvi_branch)

    Returns:
        model: PyTorch model

    Example:
        >>> # Full NDVI weighting
        >>> model = get_ndvi_weighted_model('multiscale_cnn', in_channels=14)
        >>>
        >>> # Only channel attention
        >>> model = get_ndvi_weighted_model('multiscale_cnn',
        ...                                  use_ndvi_branch=False)
        >>>
        >>> # Only NDVI branch
        >>> model = get_ndvi_weighted_model('multiscale_cnn',
        ...                                  use_channel_attention=False)
    """
    if model_type == 'multiscale_cnn':
        return MultiScaleCNN_NDVIWeighted(in_channels, **kwargs)
    elif model_type == 'shallow_unet':
        return ShallowUNet_NDVIWeighted(in_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models
    print("Testing NDVI-Weighted Models\n")

    models = {
        'MultiScale CNN (Full)': get_ndvi_weighted_model('multiscale_cnn'),
        'MultiScale CNN (Attention only)': get_ndvi_weighted_model(
            'multiscale_cnn', use_ndvi_branch=False
        ),
        'MultiScale CNN (NDVI branch only)': get_ndvi_weighted_model(
            'multiscale_cnn', use_channel_attention=False
        ),
        'Shallow U-Net (Full)': get_ndvi_weighted_model('shallow_unet'),
    }

    x = torch.randn(2, 14, 128, 128)

    for name, model in models.items():
        y = model(x)
        params = count_parameters(model)
        print(f"{name}:")
        print(f"  Output shape: {y.shape}")
        print(f"  Parameters: {params:,}")
        print()
