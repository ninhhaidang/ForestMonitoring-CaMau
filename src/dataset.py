"""
PyTorch Dataset classes for deforestation detection
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import random


class DeforestationDataset(Dataset):
    """
    PyTorch Dataset for deforestation patches

    Args:
        patches_dir: Directory containing .npy patch files
        transform: Optional transform to apply to patches
        augment: Apply data augmentation (default: False)

    Example:
        >>> from src.dataset import DeforestationDataset
        >>> from torch.utils.data import DataLoader
        >>>
        >>> # Create dataset
        >>> train_dataset = DeforestationDataset('data/patches/train', augment=True)
        >>> print(f"Dataset size: {len(train_dataset)}")
        >>>
        >>> # Create dataloader
        >>> train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        >>>
        >>> # Get a batch
        >>> for patches, labels in train_loader:
        >>>     print(f"Batch: patches {patches.shape}, labels {labels.shape}")
        >>>     break
    """

    def __init__(self,
                 patches_dir: str,
                 transform: Optional[Callable] = None,
                 augment: bool = False):
        super().__init__()

        self.patches_dir = Path(patches_dir)
        if not self.patches_dir.exists():
            raise ValueError(f"Patches directory not found: {self.patches_dir}")

        # Get all .npy files
        self.files = sorted(list(self.patches_dir.glob('*.npy')))
        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in: {self.patches_dir}")

        self.transform = transform
        self.augment = augment

        # Count classes
        self.labels = [self._extract_label(f) for f in self.files]
        self.class_counts = {
            0: self.labels.count(0),
            1: self.labels.count(1)
        }

        print(f"✅ Dataset loaded: {len(self.files)} patches")
        print(f"   Class 0 (No deforestation): {self.class_counts[0]}")
        print(f"   Class 1 (Deforestation): {self.class_counts[1]}")

    def __len__(self) -> int:
        return len(self.files)

    def _extract_label(self, filepath: Path) -> int:
        """Extract label from filename (e.g., train_0001_label1.npy -> 1)"""
        filename = filepath.stem
        label_part = filename.split('_label')[-1]
        return int(label_part)

    def _apply_augmentation(self, patch: np.ndarray) -> np.ndarray:
        """Apply random augmentation to patch (H, W, C)"""
        # Random rotation (90, 180, 270 degrees)
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            patch = np.rot90(patch, k=k, axes=(0, 1))

        # Random horizontal flip
        if random.random() < 0.5:
            patch = np.flip(patch, axis=1)

        # Random vertical flip
        if random.random() < 0.5:
            patch = np.flip(patch, axis=0)

        # Add small Gaussian noise (very small to not destroy normalized data)
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, patch.shape)
            patch = patch + noise

        return patch.copy()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single patch and label

        Returns:
            patch: Tensor of shape (C, H, W) - 18 channels, 128x128 pixels
            label: Tensor of shape (1,) - binary label (0 or 1)
        """
        # Load patch
        filepath = self.files[idx]
        patch = np.load(filepath)  # Shape: (H, W, C) = (128, 128, 18)

        # Extract label
        label = self._extract_label(filepath)

        # Apply augmentation if enabled
        if self.augment:
            patch = self._apply_augmentation(patch)

        # Convert to tensor: (H, W, C) -> (C, H, W)
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()

        # Apply custom transform if provided
        if self.transform:
            patch = self.transform(patch)

        # Label as tensor
        label = torch.tensor(label, dtype=torch.float32)

        return patch, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets

        Returns:
            Tensor of shape (2,) with weights for class 0 and 1

        Example:
            >>> dataset = DeforestationDataset('data/patches/train')
            >>> weights = dataset.get_class_weights()
            >>> criterion = nn.BCELoss(weight=weights)
        """
        total = len(self.files)
        weights = torch.tensor([
            total / (2 * self.class_counts[0]),
            total / (2 * self.class_counts[1])
        ])
        return weights

    def get_sample_info(self, idx: int) -> dict:
        """
        Get information about a specific sample

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample information

        Example:
            >>> dataset = DeforestationDataset('data/patches/train')
            >>> info = dataset.get_sample_info(0)
            >>> print(info)
        """
        filepath = self.files[idx]
        patch = np.load(filepath)
        label = self._extract_label(filepath)

        return {
            'filename': filepath.name,
            'index': idx,
            'label': label,
            'shape': patch.shape,
            'dtype': patch.dtype,
            'range': (float(patch.min()), float(patch.max())),
            'has_nan': bool(np.isnan(patch).any())
        }


def create_dataloaders(train_dir: str,
                      val_dir: str,
                      test_dir: str,
                      batch_size: int = 16,
                      num_workers: int = 4,
                      augment_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders

    Args:
        train_dir: Directory with training patches
        val_dir: Directory with validation patches
        test_dir: Directory with test patches
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        augment_train: Apply augmentation to training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        >>>     train_dir='data/patches/train',
        >>>     val_dir='data/patches/val',
        >>>     test_dir='data/patches/test',
        >>>     batch_size=16
        >>> )
        >>>
        >>> # Use in training loop
        >>> for patches, labels in train_loader:
        >>>     # Training code here
        >>>     pass
    """
    # Create datasets
    train_dataset = DeforestationDataset(train_dir, augment=augment_train)
    val_dataset = DeforestationDataset(val_dir, augment=False)
    test_dataset = DeforestationDataset(test_dir, augment=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("\n✅ DataLoaders created:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test:  {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def print_dataset_info(dataset: DeforestationDataset):
    """
    Print detailed information about dataset

    Args:
        dataset: DeforestationDataset instance

    Example:
        >>> dataset = DeforestationDataset('data/patches/train')
        >>> print_dataset_info(dataset)
    """
    print("\n" + "="*80)
    print("DATASET INFORMATION")
    print("="*80)

    print(f"\n📁 Directory: {dataset.patches_dir}")
    print(f"📊 Total samples: {len(dataset)}")

    print(f"\n🏷️ Class distribution:")
    for label, count in dataset.class_counts.items():
        pct = 100 * count / len(dataset)
        label_name = "No deforestation" if label == 0 else "Deforestation"
        print(f"   Class {label} ({label_name}): {count} ({pct:.1f}%)")

    print(f"\n⚖️ Class weights:")
    weights = dataset.get_class_weights()
    print(f"   Class 0: {weights[0]:.4f}")
    print(f"   Class 1: {weights[1]:.4f}")

    print(f"\n🔧 Augmentation: {'Enabled' if dataset.augment else 'Disabled'}")

    # Sample info
    if len(dataset) > 0:
        sample_info = dataset.get_sample_info(0)
        print(f"\n📦 Sample patch info:")
        print(f"   Shape: {sample_info['shape']}")
        print(f"   Data type: {sample_info['dtype']}")
        print(f"   Value range: [{sample_info['range'][0]:.3f}, {sample_info['range'][1]:.3f}]")
        print(f"   Has NaN: {sample_info['has_nan']}")

    print("\n" + "="*80)


if __name__ == "__main__":
    print("Testing dataset module...")

    # Test with actual data if available
    import sys
    from pathlib import Path

    # Try to load dataset
    test_dirs = [
        Path("../data/patches/train"),
        Path("data/patches/train")
    ]

    dataset = None
    for test_dir in test_dirs:
        if test_dir.exists():
            print(f"\n✅ Found patches directory: {test_dir}")

            try:
                # Create dataset
                dataset = DeforestationDataset(test_dir, augment=False)

                # Print info
                print_dataset_info(dataset)

                # Test __getitem__
                print("\n🧪 Testing __getitem__:")
                patch, label = dataset[0]
                print(f"   Patch shape: {patch.shape}")
                print(f"   Patch dtype: {patch.dtype}")
                print(f"   Label: {label.item()}")

                # Test dataloader
                print("\n🧪 Testing DataLoader:")
                from torch.utils.data import DataLoader
                loader = DataLoader(dataset, batch_size=4, shuffle=True)

                for batch_patches, batch_labels in loader:
                    print(f"   Batch patches shape: {batch_patches.shape}")
                    print(f"   Batch labels shape: {batch_labels.shape}")
                    break

                print("\n✅ All tests passed!")
                break

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                break
    else:
        print("\n⚠️ No patches directory found. Please create patches first using:")
        print("   python -c \"from src.preprocessing import create_patches_dataset; create_patches_dataset(...)\"")
