"""
2D CNN Model for Patch-based Deforestation Detection
Lightweight architecture designed for small datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DeforestationCNN(nn.Module):
    """
    Lightweight 2D CNN for deforestation detection
    Input: (batch, patch_size, patch_size, n_features)
    Output: (batch, 2) - logits for binary classification
    """

    def __init__(
        self,
        patch_size: int = 3,
        n_features: int = 27,
        n_classes: int = 2,
        dropout_rate: float = 0.5
    ):
        """
        Initialize DeforestationCNN

        Args:
            patch_size: Size of input patch (e.g., 3 for 3x3)
            n_features: Number of feature channels (e.g., 27)
            n_classes: Number of output classes (2 for binary)
            dropout_rate: Dropout probability
        """
        super(DeforestationCNN, self).__init__()

        self.patch_size = patch_size
        self.n_features = n_features
        self.n_classes = n_classes

        # Convolutional layers
        # Input: (batch, n_features, patch_size, patch_size)

        # Conv Block 1
        self.conv1 = nn.Conv2d(
            in_channels=n_features,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(p=0.3)

        # Conv Block 2
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout2d(p=0.3)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(32, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout_fc = nn.Dropout(p=dropout_rate)

        self.fc2 = nn.Linear(64, n_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, patch_size, patch_size, n_features)

        Returns:
            Logits tensor (batch, n_classes)
        """
        # Permute to (batch, n_features, patch_size, patch_size)
        x = x.permute(0, 3, 1, 2)

        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Global Average Pooling
        x = self.global_pool(x)  # (batch, 32, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 32)

        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)

        x = self.fc2(x)  # Logits

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities

        Args:
            x: Input tensor

        Returns:
            Probabilities tensor (batch, n_classes)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels

        Args:
            x: Input tensor

        Returns:
            Class labels (batch,)
        """
        probs = self.predict_proba(x)
        preds = torch.argmax(probs, dim=1)
        return preds

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        summary = []
        summary.append("="*70)
        summary.append("DeforestationCNN Model Architecture")
        summary.append("="*70)
        summary.append(f"Input shape: (batch, {self.patch_size}, {self.patch_size}, {self.n_features})")
        summary.append(f"Output shape: (batch, {self.n_classes})")
        summary.append(f"\nTotal parameters: {self.count_parameters():,}")
        summary.append("\nLayer details:")
        summary.append(f"  Conv1: {self.n_features} -> 64 channels (3x3)")
        summary.append(f"  Conv2: 64 -> 32 channels (3x3)")
        summary.append(f"  Global Avg Pool")
        summary.append(f"  FC1: 32 -> 64")
        summary.append(f"  FC2: 64 -> {self.n_classes}")
        summary.append("="*70)
        return "\n".join(summary)


class DeforestationCNNDeeper(nn.Module):
    """
    Deeper variant with 3 conv blocks (alternative architecture)
    """

    def __init__(
        self,
        patch_size: int = 3,
        n_features: int = 27,
        n_classes: int = 2,
        dropout_rate: float = 0.5
    ):
        super(DeforestationCNNDeeper, self).__init__()

        self.patch_size = patch_size
        self.n_features = n_features
        self.n_classes = n_classes

        # Conv Block 1
        self.conv1 = nn.Conv2d(n_features, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(p=0.3)

        # Conv Block 2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(p=0.3)

        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout3 = nn.Dropout2d(p=0.3)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # FC layers
        self.fc1 = nn.Linear(32, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout_fc = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, n_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    model_type: str = 'standard',
    patch_size: int = 3,
    n_features: int = 27,
    n_classes: int = 2,
    dropout_rate: float = 0.5
) -> nn.Module:
    """
    Factory function to create model

    Args:
        model_type: 'standard' or 'deeper'
        patch_size: Input patch size
        n_features: Number of feature channels
        n_classes: Number of output classes
        dropout_rate: Dropout rate

    Returns:
        CNN model
    """
    if model_type == 'standard':
        model = DeforestationCNN(
            patch_size=patch_size,
            n_features=n_features,
            n_classes=n_classes,
            dropout_rate=dropout_rate
        )
    elif model_type == 'deeper':
        model = DeforestationCNNDeeper(
            patch_size=patch_size,
            n_features=n_features,
            n_classes=n_classes,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


if __name__ == '__main__':
    # Test model
    model = create_model(model_type='standard', patch_size=3, n_features=27)
    print(model.get_model_summary())

    # Test forward pass
    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 3, 27)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    # Test prediction
    probs = model.predict_proba(dummy_input)
    preds = model.predict(dummy_input)
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Predictions shape: {preds.shape}")
