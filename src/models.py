"""
Deep learning models for deforestation detection
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision import models


class SimpleCNN(nn.Module):
    """Simple CNN for binary classification"""

    def __init__(self, in_channels=18, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8

            # Conv block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8 -> 4
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNet18Classifier(nn.Module):
    """ResNet-18 based classifier for binary classification"""

    def __init__(self, in_channels=18, num_classes=2, pretrained=False):
        super(ResNet18Classifier, self).__init__()

        # Load ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)

        # Modify first conv layer to accept custom input channels
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify final fc layer for binary classification
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)


class UNetClassifier(nn.Module):
    """U-Net encoder for classification (using segmentation_models_pytorch)"""

    def __init__(self, in_channels=18, num_classes=2, encoder_name='resnet34'):
        super(UNetClassifier, self).__init__()

        # Create U-Net model (we'll use only the encoder part)
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,  # Random init for custom input channels
            in_channels=in_channels,
            classes=num_classes
        )

        # Replace decoder with global pooling + classifier
        # Get encoder output channels
        encoder_channels = self.unet.encoder.out_channels[-1]

        self.encoder = self.unet.encoder
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(encoder_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features using encoder
        features = self.encoder(x)

        # Use the last (deepest) feature map
        x = features[-1]

        # Classify
        x = self.classifier(x)
        return x


def get_model(model_name='simple_cnn', in_channels=18, num_classes=2):
    """
    Factory function to get model by name

    Args:
        model_name: 'simple_cnn', 'resnet18', or 'unet'
        in_channels: Number of input channels
        num_classes: Number of output classes

    Returns:
        PyTorch model
    """
    if model_name == 'simple_cnn':
        return SimpleCNN(in_channels, num_classes)
    elif model_name == 'resnet18':
        return ResNet18Classifier(in_channels, num_classes)
    elif model_name == 'unet':
        return UNetClassifier(in_channels, num_classes, encoder_name='resnet34')
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    # Test models
    batch_size = 2
    in_channels = 18  # Updated: 2 time periods × (7 S2 + 2 S1) = 18 channels
    patch_size = 64

    x = torch.randn(batch_size, in_channels, patch_size, patch_size)

    print("Testing SimpleCNN...")
    model = SimpleCNN(in_channels=in_channels)
    out = model(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")

    print("\nTesting ResNet18Classifier...")
    model = ResNet18Classifier(in_channels=in_channels)
    out = model(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")

    print("\nTesting UNetClassifier...")
    model = UNetClassifier(in_channels=in_channels)
    out = model(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
