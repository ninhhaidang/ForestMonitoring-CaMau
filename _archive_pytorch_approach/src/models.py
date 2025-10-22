"""
Model definitions using segmentation_models_pytorch
3 lightweight models for forest change detection
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def get_model(model_name, in_channels=18, num_classes=2, pretrained='imagenet'):
    """
    Get segmentation model from segmentation_models_pytorch

    Args:
        model_name: One of ['unet_efficientnet', 'unet_mobilenet', 'fpn_efficientnet']
        in_channels: Number of input channels (default: 18)
        num_classes: Number of output classes (default: 2 for binary)
        pretrained: Pretrained weights ('imagenet' or None)

    Returns:
        PyTorch model
    """

    if model_name == 'unet_efficientnet':
        model = smp.Unet(
            encoder_name='efficientnet-b0',
            encoder_weights=pretrained,
            in_channels=in_channels,
            classes=num_classes,
            activation=None  # Use raw logits
        )
        print(f"✅ UNet-EfficientNet-B0 initialized")
        print(f"   Encoder: EfficientNet-B0 (pretrained: {pretrained})")
        print(f"   Params: ~5M")
        print(f"   Best for: Balanced performance")

    elif model_name == 'unet_mobilenet':
        model = smp.Unet(
            encoder_name='mobilenet_v2',
            encoder_weights=pretrained,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        print(f"✅ UNet-MobileNetV2 initialized")
        print(f"   Encoder: MobileNetV2 (pretrained: {pretrained})")
        print(f"   Params: ~2M")
        print(f"   Best for: Fastest inference, mobile deployment")

    elif model_name == 'fpn_efficientnet':
        model = smp.FPN(
            encoder_name='efficientnet-b0',
            encoder_weights=pretrained,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        print(f"✅ FPN-EfficientNet-B0 initialized")
        print(f"   Encoder: EfficientNet-B0 (pretrained: {pretrained})")
        print(f"   Params: ~6M")
        print(f"   Best for: Highest accuracy")

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: "
                         f"['unet_efficientnet', 'unet_mobilenet', 'fpn_efficientnet']")

    return model


class ForestChangeModel(nn.Module):
    """
    Wrapper for forest change detection model
    Adds additional utilities like prediction, metrics, etc.
    """

    def __init__(self, model_name='unet_efficientnet', in_channels=18, num_classes=2):
        super().__init__()
        self.model = get_model(model_name, in_channels, num_classes)
        self.model_name = model_name

    def forward(self, x):
        """Forward pass"""
        return self.model(x)

    def predict(self, x):
        """
        Predict with softmax probabilities

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            probs: Probability tensor (B, H, W) - probability of class 1
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)  # (B, 2, H, W)
            probs = torch.softmax(logits, dim=1)  # (B, 2, H, W)
            probs_class1 = probs[:, 1, :, :]  # (B, H, W) - probability of change
        return probs_class1

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_loss_function(loss_name='ce', class_weights=None):
    """
    Get loss function

    Args:
        loss_name: 'ce' (CrossEntropy), 'focal', 'dice'
        class_weights: Tensor of class weights (optional)

    Returns:
        Loss function
    """
    if loss_name == 'ce':
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        print(f"✅ Loss: CrossEntropyLoss")

    elif loss_name == 'focal':
        # Focal loss for imbalanced data
        from segmentation_models_pytorch.losses import FocalLoss
        criterion = FocalLoss(mode='multiclass')
        print(f"✅ Loss: FocalLoss")

    elif loss_name == 'dice':
        # Dice loss for segmentation
        from segmentation_models_pytorch.losses import DiceLoss
        criterion = DiceLoss(mode='multiclass')
        print(f"✅ Loss: DiceLoss")

    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    return criterion


def get_optimizer(model, optimizer_name='adam', lr=1e-4, weight_decay=1e-5):
    """
    Get optimizer

    Args:
        model: PyTorch model
        optimizer_name: 'adam', 'adamw', 'sgd'
        lr: Learning rate
        weight_decay: Weight decay

    Returns:
        Optimizer
    """
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        print(f"✅ Optimizer: Adam (lr={lr}, weight_decay={weight_decay})")

    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        print(f"✅ Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")

    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
        print(f"✅ Optimizer: SGD (lr={lr}, momentum=0.9)")

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def get_scheduler(optimizer, scheduler_name='cosine', epochs=50):
    """
    Get learning rate scheduler

    Args:
        optimizer: PyTorch optimizer
        scheduler_name: 'cosine', 'step', 'reduce_on_plateau'
        epochs: Total number of epochs

    Returns:
        Scheduler
    """
    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6
        )
        print(f"✅ Scheduler: CosineAnnealingLR (T_max={epochs})")

    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )
        print(f"✅ Scheduler: StepLR (step=10, gamma=0.5)")

    elif scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        print(f"✅ Scheduler: ReduceLROnPlateau (patience=5)")

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler


# Model comparison table
MODEL_INFO = {
    'unet_efficientnet': {
        'name': 'UNet-EfficientNet-B0',
        'encoder': 'EfficientNet-B0',
        'params': '~5M',
        'speed': '⚡⚡⚡',
        'accuracy': '⭐⭐⭐⭐',
        'memory': '~4GB VRAM',
        'best_for': 'Balanced performance'
    },
    'unet_mobilenet': {
        'name': 'UNet-MobileNetV2',
        'encoder': 'MobileNetV2',
        'params': '~2M',
        'speed': '⚡⚡⚡⚡',
        'accuracy': '⭐⭐⭐',
        'memory': '~2GB VRAM',
        'best_for': 'Fastest, mobile deployment'
    },
    'fpn_efficientnet': {
        'name': 'FPN-EfficientNet-B0',
        'encoder': 'EfficientNet-B0',
        'params': '~6M',
        'speed': '⚡⚡',
        'accuracy': '⭐⭐⭐⭐⭐',
        'memory': '~6GB VRAM',
        'best_for': 'Highest accuracy'
    }
}


def print_model_comparison():
    """Print comparison table of all 3 models"""
    print("\n" + "="*80)
    print("🔬 MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<25} {'Params':<10} {'Speed':<12} {'Accuracy':<12} {'Memory':<12}")
    print("-"*80)
    for model_key, info in MODEL_INFO.items():
        print(f"{info['name']:<25} {info['params']:<10} {info['speed']:<12} "
              f"{info['accuracy']:<12} {info['memory']:<12}")
    print("="*80)
    print("Best for:")
    for model_key, info in MODEL_INFO.items():
        print(f"  • {info['name']}: {info['best_for']}")
    print("="*80 + "\n")


if __name__ == '__main__':
    # Test model creation
    print_model_comparison()

    print("\n🧪 Testing model creation...\n")
    for model_name in ['unet_efficientnet', 'unet_mobilenet', 'fpn_efficientnet']:
        model = ForestChangeModel(model_name)
        print(f"   Total parameters: {model.count_parameters():,}")
        print()
