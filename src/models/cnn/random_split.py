"""
Random data splitting for CNN training
Simple stratified random split for train/val/test sets
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomKFoldWithFixedTest:
    """
    Random K-Fold CV with Fixed Test Set
    
    Workflow:
    1. Split off a fixed test set (e.g., 20%)
    2. Perform K-Fold CV on the remaining data (e.g., 80%)
    3. Train final model on all remaining data (80%)
    4. Evaluate final model on fixed test set (20%)
    
    This ensures:
    - Test set is never seen during CV
    - Proper evaluation on truly held-out data
    - Stratified splits to maintain class balance
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_splits: int = 5,
        random_state: int = 42,
        shuffle: bool = True
    ):
        """
        Initialize RandomKFoldWithFixedTest

        Args:
            test_size: Proportion of data for fixed test set (e.g., 0.2 = 20%)
            n_splits: Number of folds for CV on remaining data
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle data before splitting
        """
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle

    def split_fixed_test(
        self,
        ground_truth: pd.DataFrame,
        stratify_by_class: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Step 1: Split off a fixed test set

        Args:
            ground_truth: DataFrame with ['x', 'y', 'label']
            stratify_by_class: Maintain class balance

        Returns:
            Tuple of (trainval_indices, test_indices, metadata)
        """
        logger.info(f"\n{'='*70}")
        logger.info("STEP 1: SPLIT OFF FIXED TEST SET")
        logger.info(f"{'='*70}")
        logger.info(f"Test size: {self.test_size*100:.1f}%")
        logger.info("Using RANDOM SPLIT (stratified by class)")
        
        if stratify_by_class:
            stratify = ground_truth['label']
        else:
            stratify = None
        
        indices = np.arange(len(ground_truth))
        trainval_indices, test_indices = train_test_split(
            indices,
            test_size=self.test_size,
            stratify=stratify,
            random_state=self.random_state,
            shuffle=self.shuffle
        )
        
        metadata = {
            'method': 'random_split',
            'test_size': self.test_size,
            'stratified': stratify_by_class
        }

        logger.info(f"\n✅ Fixed test set created:")
        logger.info(f"   Train+Val: {len(trainval_indices)} samples ({len(trainval_indices)/len(ground_truth)*100:.1f}%)")
        logger.info(f"   Test (FIXED): {len(test_indices)} samples ({len(test_indices)/len(ground_truth)*100:.1f}%)")
        
        # Log class distribution
        logger.info(f"\n   Test set class distribution:")
        test_labels = ground_truth.iloc[test_indices]['label']
        for label in sorted(test_labels.unique()):
            count = (test_labels == label).sum()
            logger.info(f"     Class {label}: {count} ({count/len(test_labels)*100:.1f}%)")

        return trainval_indices, test_indices, metadata

    def cross_validate(
        self,
        ground_truth_trainval: pd.DataFrame,
        stratify_by_class: bool = True
    ):
        """
        Step 2: Perform K-Fold CV on train+val data

        Args:
            ground_truth_trainval: DataFrame with train+val data only
            stratify_by_class: Maintain class balance

        Yields:
            Tuple of (fold_idx, train_indices, val_indices) for each fold
        """
        logger.info(f"\n{'='*70}")
        logger.info("STEP 2: K-FOLD CV ON TRAIN+VAL DATA")
        logger.info(f"{'='*70}")
        logger.info(f"Number of folds: {self.n_splits}")
        logger.info(f"Data size: {len(ground_truth_trainval)} samples")
        logger.info("Using RANDOM K-FOLD (stratified by class)")
        
        if stratify_by_class:
            kfold = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(
                ground_truth_trainval,
                ground_truth_trainval['label']
            )):
                logger.info(f"\nFold {fold_idx + 1}/{self.n_splits}:")
                logger.info(f"  Train: {len(train_indices)} samples ({len(train_indices)/len(ground_truth_trainval)*100:.1f}%)")
                logger.info(f"  Val: {len(val_indices)} samples ({len(val_indices)/len(ground_truth_trainval)*100:.1f}%)")
                
                # Log class distribution for this fold
                train_labels = ground_truth_trainval.iloc[train_indices]['label']
                val_labels = ground_truth_trainval.iloc[val_indices]['label']
                
                logger.info(f"  Train classes: {dict(train_labels.value_counts().sort_index())}")
                logger.info(f"  Val classes: {dict(val_labels.value_counts().sort_index())}")
                
                yield fold_idx, train_indices, val_indices
        else:
            kfold = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(
                ground_truth_trainval
            )):
                logger.info(f"\nFold {fold_idx + 1}/{self.n_splits}:")
                logger.info(f"  Train: {len(train_indices)} samples")
                logger.info(f"  Val: {len(val_indices)} samples")
                yield fold_idx, train_indices, val_indices

    def get_full_trainval_data(
        self,
        trainval_indices: np.ndarray
    ) -> np.ndarray:
        """
        Step 3: Get all train+val indices for final model training

        Args:
            trainval_indices: All train+val indices from step 1

        Returns:
            trainval_indices (same as input, for clarity)
        """
        logger.info(f"\n{'='*70}")
        logger.info("STEP 3: PREPARE FULL TRAIN+VAL DATA FOR FINAL MODEL")
        logger.info(f"{'='*70}")
        logger.info(f"Using all {len(trainval_indices)} samples for final training")
        logger.info(f"This is the model that will be used for prediction")

        return trainval_indices
