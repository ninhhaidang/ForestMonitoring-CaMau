"""
Deep learning models for deforestation detection - Phase 1
"""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN for binary deforestation classification.

    Architecture (from README):
    - Conv Block 1: 18 → 32 channels
    - Conv Block 2: 32 → 64 channels
    - Conv Block 3: 64 → 128 channels
    - Conv Block 4: 128 → 256 channels
    - Global Average Pooling
    - FC: 256 → 128 → 2

    Regularization:
    - BatchNorm after each conv
    - Dropout (progressive: 0.3 → 0.5)
    - Global Average Pooling (reduce parameters)

    Parameters: ~1.2M
    """

    def __init__(self, in_channels=18, num_classes=2):
        super(SimpleCNN, self).__init__()

        # Conv Block 1: 18 → 32
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 → 32
            nn.Dropout(0.3)
        )

        # Conv Block 2: 32 → 64
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 → 16
            nn.Dropout(0.3)
        )

        # Conv Block 3: 64 → 128
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 → 8
            nn.Dropout(0.4)
        )

        # Conv Block 4: 128 → 256
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8 → 4
            nn.Dropout(0.5)
        )

        # Classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 18, 64, 64)

        Returns:
            Output logits of shape (batch, 2)
        """
        x = self.conv_block1(x)  # (batch, 32, 32, 32)
        x = self.conv_block2(x)  # (batch, 64, 16, 16)
        x = self.conv_block3(x)  # (batch, 128, 8, 8)
        x = self.conv_block4(x)  # (batch, 256, 4, 4)

        x = self.global_avg_pool(x)  # (batch, 256, 1, 1)
        x = self.classifier(x)        # (batch, 2)

        return x


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        total_params: Total number of trainable parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # Test SimpleCNN
    print("=" * 60)
    print("Testing SimpleCNN")
    print("=" * 60)

    # Create model
    model = SimpleCNN(in_channels=18, num_classes=2)

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameters in millions: {total_params / 1e6:.2f}M")

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 18, 64, 64)
    print(f"\nInput shape: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Test with different batch sizes
    print("\n" + "=" * 60)
    print("Testing different batch sizes")
    print("=" * 60)
    for bs in [1, 8, 16, 24]:
        dummy_input = torch.randn(bs, 18, 64, 64)
        output = model(dummy_input)
        print(f"Batch size {bs:2d}: Input {tuple(dummy_input.shape)} → Output {tuple(output.shape)}")

    print("\n✓ SimpleCNN test passed!")
