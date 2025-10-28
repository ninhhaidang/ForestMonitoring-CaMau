"""
Machine Learning models for deforestation detection

This module implements traditional ML algorithms:
1. Random Forest - Ensemble of decision trees
2. Support Vector Machine (SVM) - For future implementation
3. Gradient Boosting - For future implementation
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm.auto import tqdm


class RandomForestModel:
    """
    Random Forest classifier for deforestation detection

    Args:
        n_estimators: Number of trees in the forest (default: 100)
        max_depth: Maximum depth of trees (default: None)
        min_samples_split: Minimum samples required to split a node (default: 2)
        min_samples_leaf: Minimum samples required at a leaf node (default: 1)
        random_state: Random seed for reproducibility (default: 42)
        n_jobs: Number of parallel jobs (-1 = use all cores)

    Example:
        >>> model = RandomForestModel(n_estimators=100, max_depth=20)
        >>> model.train(X_train, y_train)
        >>> y_pred = model.predict(X_test)
        >>> metrics = model.evaluate(X_test, y_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0
        )
        self.is_trained = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Train the Random Forest model

        Args:
            X_train: Training features (N, num_features)
            y_train: Training labels (N,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary with training and validation metrics
        """
        print(f"Training Random Forest with {len(X_train)} samples...")

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)[:, 1]

        metrics = {
            'train_acc': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'train_f1': f1_score(y_train, y_train_pred),
            'train_auc': roc_auc_score(y_train, y_train_proba)
        }

        # Calculate validation metrics if provided
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            y_val_proba = self.model.predict_proba(X_val)[:, 1]

            metrics.update({
                'val_acc': accuracy_score(y_val, y_val_pred),
                'val_precision': precision_score(y_val, y_val_pred),
                'val_recall': recall_score(y_val, y_val_pred),
                'val_f1': f1_score(y_val, y_val_pred),
                'val_auc': roc_auc_score(y_val, y_val_proba)
            })

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels

        Args:
            X: Features (N, num_features)

        Returns:
            Predicted labels (N,)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Features (N, num_features)

        Returns:
            Predicted probabilities (N, 2) - [prob_class_0, prob_class_1]
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            X: Features (N, num_features)
            y: True labels (N,)

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_proba)
        }

        return metrics

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores

        Returns:
            Feature importance array (num_features,)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained to get feature importance")

        return self.model.feature_importances_

    def save(self, save_path: Union[str, Path]):
        """
        Save model to disk

        Args:
            save_path: Path to save the model (.pkl file)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)

        print(f"Model saved to: {save_path}")

    def load(self, load_path: Union[str, Path]):
        """
        Load model from disk

        Args:
            load_path: Path to load the model from (.pkl file)
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        with open(load_path, 'rb') as f:
            self.model = pickle.load(f)

        self.is_trained = True
        print(f"Model loaded from: {load_path}")


def load_patches_for_ml(
    patches_dir: Union[str, Path],
    split: str = 'train'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and flatten patches for machine learning models

    Args:
        patches_dir: Directory containing patches
        split: Data split ('train', 'val', or 'test')

    Returns:
        X: Flattened features (N, num_features)
        y: Labels (N,)

    Example:
        >>> X_train, y_train = load_patches_for_ml('data/patches', 'train')
        >>> print(X_train.shape)  # (899, 14*128*128)
    """
    patches_dir = Path(patches_dir) / split

    if not patches_dir.exists():
        raise FileNotFoundError(f"Patches directory not found: {patches_dir}")

    # Get all patch files
    patch_files = sorted(patches_dir.glob('*.npy'))

    if len(patch_files) == 0:
        raise ValueError(f"No patches found in {patches_dir}")

    # Load patches
    X_list = []
    y_list = []

    for patch_file in tqdm(patch_files, desc=f"Loading {split}", unit="patch"):
        # Load patch
        patch = np.load(patch_file)  # (128, 128, 14)

        # Flatten to 1D vector
        patch_flat = patch.flatten()  # (128*128*14,)

        # Extract label from filename
        # Format: {split}_{id:04d}_label{label}.npy
        label = int(patch_file.stem.split('_label')[-1])

        X_list.append(patch_flat)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"Loaded {len(X)} patches from {split} set")
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Class distribution: {np.bincount(y)}")

    return X, y


if __name__ == "__main__":
    # Test the module
    print("Testing Random Forest model...")

    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.randn(100, 20)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(20, 20)
    y_test = np.random.randint(0, 2, 20)

    # Train model
    model = RandomForestModel(n_estimators=10, max_depth=5)
    metrics = model.train(X_train, y_train, X_test, y_test)

    print("\nTraining metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Test predictions
    y_pred = model.predict(X_test)
    print(f"\nPredictions: {y_pred[:10]}")

    # Test feature importance
    importance = model.get_feature_importance()
    print(f"\nFeature importance shape: {importance.shape}")

    print("\nRandom Forest model test completed!")
