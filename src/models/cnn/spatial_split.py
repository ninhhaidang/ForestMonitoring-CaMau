"""
Spatial-aware data splitting
Ensures no spatial overlap between train/validation/test sets
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialSplitter:
    """
    Spatial-aware data splitter to avoid data leakage
    """

    def __init__(
        self,
        cluster_distance: float = 50.0,
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ):
        """
        Initialize SpatialSplitter

        Args:
            cluster_distance: Distance threshold for clustering (meters)
            train_size: Proportion for training
            val_size: Proportion for validation
            test_size: Proportion for testing
            random_state: Random seed for reproducibility
        """
        self.cluster_distance = cluster_distance
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

        # Verify proportions sum to 1
        total = train_size + val_size + test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"Proportions must sum to 1.0, got {total}")

    def cluster_points(
        self,
        ground_truth: pd.DataFrame
    ) -> np.ndarray:
        """
        Cluster nearby points using hierarchical clustering

        Args:
            ground_truth: DataFrame with columns ['x', 'y', 'label']

        Returns:
            Array of cluster IDs for each point
        """
        logger.info(f"\n{'='*70}")
        logger.info("SPATIAL CLUSTERING")
        logger.info(f"{'='*70}")
        logger.info(f"Cluster distance threshold: {self.cluster_distance}m")

        coords = ground_truth[['x', 'y']].values
        distances = pdist(coords, metric='euclidean')

        # Hierarchical clustering with single linkage
        Z = linkage(distances, method='single')
        clusters = fcluster(Z, t=self.cluster_distance, criterion='distance')

        n_clusters = len(np.unique(clusters))
        cluster_sizes = np.bincount(clusters)

        logger.info(f"\nClustering results:")
        logger.info(f"  Total points: {len(ground_truth)}")
        logger.info(f"  Number of clusters: {n_clusters}")
        logger.info(f"  Singleton clusters (size=1): {(cluster_sizes == 1).sum()}")
        logger.info(f"  Multi-point clusters (size>1): {(cluster_sizes > 1).sum()}")
        logger.info(f"  Max cluster size: {cluster_sizes.max()}")
        logger.info(f"  Mean cluster size: {cluster_sizes.mean():.2f}")

        return clusters

    def split_by_clusters(
        self,
        ground_truth: pd.DataFrame,
        clusters: np.ndarray,
        stratify_by_class: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data by clusters (not by individual points)

        Args:
            ground_truth: DataFrame with 'label' column
            clusters: Cluster ID for each point
            stratify_by_class: Try to maintain class balance

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        logger.info(f"\n{'='*70}")
        logger.info("CLUSTER-BASED SPLITTING")
        logger.info(f"{'='*70}")

        # Create cluster DataFrame
        unique_clusters = np.unique(clusters)
        cluster_info = []

        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_points = ground_truth[cluster_mask]

            # Get majority class for this cluster
            majority_class = cluster_points['label'].mode()[0]
            n_points = len(cluster_points)

            cluster_info.append({
                'cluster_id': cluster_id,
                'n_points': n_points,
                'majority_class': majority_class
            })

        cluster_df = pd.DataFrame(cluster_info)

        logger.info(f"Created {len(cluster_df)} clusters for splitting")

        # Split clusters into train/val/test
        if stratify_by_class:
            stratify = cluster_df['majority_class']
        else:
            stratify = None

        # First split: train vs (val+test)
        train_clusters, temp_clusters = train_test_split(
            cluster_df['cluster_id'].values,
            test_size=(self.val_size + self.test_size),
            stratify=stratify,
            random_state=self.random_state
        )

        # Second split: val vs test
        # Handle edge case: if val_size=0, skip second split
        if self.val_size == 0:
            val_clusters = np.array([])  # Empty array
            test_clusters = temp_clusters  # All temp goes to test
        else:
            val_proportion = self.val_size / (self.val_size + self.test_size)

            if stratify_by_class:
                temp_mask = cluster_df['cluster_id'].isin(temp_clusters)
                temp_stratify = cluster_df[temp_mask]['majority_class']
            else:
                temp_stratify = None

            val_clusters, test_clusters = train_test_split(
                temp_clusters,
                test_size=(1 - val_proportion),
                stratify=temp_stratify,
                random_state=self.random_state
            )

        # Get point indices for each split
        train_indices = np.where(np.isin(clusters, train_clusters))[0]
        val_indices = np.where(np.isin(clusters, val_clusters))[0]
        test_indices = np.where(np.isin(clusters, test_clusters))[0]

        # Log split statistics
        logger.info(f"\n{'='*70}")
        logger.info("SPLIT STATISTICS")
        logger.info(f"{'='*70}")

        total_points = len(ground_truth)
        logger.info(f"Total points: {total_points}")

        logger.info(f"\nTrain set:")
        logger.info(f"  Points: {len(train_indices)} ({len(train_indices)/total_points*100:.2f}%)")
        logger.info(f"  Clusters: {len(train_clusters)}")
        train_labels = ground_truth.iloc[train_indices]['label']
        for label in sorted(train_labels.unique()):
            logger.info(f"  Class {label}: {(train_labels == label).sum()}")

        logger.info(f"\nValidation set:")
        logger.info(f"  Points: {len(val_indices)} ({len(val_indices)/total_points*100:.2f}%)")
        logger.info(f"  Clusters: {len(val_clusters)}")
        if len(val_indices) > 0:
            val_labels = ground_truth.iloc[val_indices]['label']
            for label in sorted(val_labels.unique()):
                logger.info(f"  Class {label}: {(val_labels == label).sum()}")
        else:
            logger.info(f"  (Empty - no validation set)")

        logger.info(f"\nTest set:")
        logger.info(f"  Points: {len(test_indices)} ({len(test_indices)/total_points*100:.2f}%)")
        logger.info(f"  Clusters: {len(test_clusters)}")
        test_labels = ground_truth.iloc[test_indices]['label']
        for label in sorted(test_labels.unique()):
            logger.info(f"  Class {label}: {(test_labels == label).sum()}")

        logger.info(f"{'='*70}\n")

        return train_indices, val_indices, test_indices

    def verify_no_overlap(
        self,
        ground_truth: pd.DataFrame,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
        min_distance: float = None
    ) -> dict:
        """
        Verify there's no spatial overlap between splits

        Args:
            ground_truth: DataFrame with ['x', 'y']
            train_indices: Training indices
            val_indices: Validation indices
            test_indices: Test indices
            min_distance: Minimum required distance (default: cluster_distance)

        Returns:
            Dictionary with verification results
        """
        if min_distance is None:
            min_distance = self.cluster_distance

        logger.info(f"\n{'='*70}")
        logger.info("VERIFYING SPATIAL SEPARATION")
        logger.info(f"{'='*70}")
        logger.info(f"Minimum required distance: {min_distance}m")

        train_coords = ground_truth.iloc[train_indices][['x', 'y']].values
        val_coords = ground_truth.iloc[val_indices][['x', 'y']].values
        test_coords = ground_truth.iloc[test_indices][['x', 'y']].values

        def compute_min_distance(coords1, coords2):
            """Compute minimum distance between two sets of points"""
            min_dist = float('inf')
            for c1 in coords1:
                for c2 in coords2:
                    dist = np.sqrt(((c1 - c2) ** 2).sum())
                    if dist < min_dist:
                        min_dist = dist
            return min_dist

        train_val_dist = compute_min_distance(train_coords, val_coords)
        train_test_dist = compute_min_distance(train_coords, test_coords)
        val_test_dist = compute_min_distance(val_coords, test_coords)

        logger.info(f"\nMinimum distances between splits:")
        logger.info(f"  Train <-> Val:  {train_val_dist:.2f}m")
        logger.info(f"  Train <-> Test: {train_test_dist:.2f}m")
        logger.info(f"  Val <-> Test:   {val_test_dist:.2f}m")

        all_ok = (train_val_dist >= min_distance and
                  train_test_dist >= min_distance and
                  val_test_dist >= min_distance)

        if all_ok:
            logger.info(f"\n✅ SUCCESS: All splits are spatially separated (>= {min_distance}m)")
        else:
            logger.warning(f"\n⚠️  WARNING: Some splits are closer than {min_distance}m")

        logger.info(f"{'='*70}\n")

        return {
            'train_val_distance': train_val_dist,
            'train_test_distance': train_test_dist,
            'val_test_distance': val_test_dist,
            'all_separated': all_ok,
            'min_required_distance': min_distance
        }

    def spatial_split(
        self,
        ground_truth: pd.DataFrame,
        stratify_by_class: bool = True,
        verify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Perform complete spatial-aware splitting

        Args:
            ground_truth: DataFrame with ['x', 'y', 'label']
            stratify_by_class: Maintain class balance
            verify: Verify spatial separation

        Returns:
            Tuple of (train_indices, val_indices, test_indices, metadata)
        """
        # Step 1: Cluster points
        clusters = self.cluster_points(ground_truth)

        # Step 2: Split by clusters
        train_indices, val_indices, test_indices = self.split_by_clusters(
            ground_truth, clusters, stratify_by_class
        )

        # Step 3: Verify (optional)
        metadata = {'clusters': clusters}
        if verify:
            verification = self.verify_no_overlap(
                ground_truth, train_indices, val_indices, test_indices
            )
            metadata['verification'] = verification

        return train_indices, val_indices, test_indices, metadata


class SpatialKFold:
    """
    Spatial-aware K-Fold Cross-Validation
    Ensures no spatial overlap between train and test sets across folds
    """

    def __init__(
        self,
        n_splits: int = 5,
        cluster_distance: float = 50.0,
        random_state: int = 42,
        shuffle: bool = True
    ):
        """
        Initialize SpatialKFold

        Args:
            n_splits: Number of folds
            cluster_distance: Distance threshold for clustering (meters)
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle clusters before splitting
        """
        self.n_splits = n_splits
        self.cluster_distance = cluster_distance
        self.random_state = random_state
        self.shuffle = shuffle

    def cluster_points(
        self,
        ground_truth: pd.DataFrame
    ) -> np.ndarray:
        """
        Cluster nearby points using hierarchical clustering

        Args:
            ground_truth: DataFrame with columns ['x', 'y', 'label']

        Returns:
            Array of cluster IDs for each point
        """
        logger.info(f"\n{'='*70}")
        logger.info("SPATIAL CLUSTERING FOR K-FOLD CV")
        logger.info(f"{'='*70}")
        logger.info(f"Cluster distance threshold: {self.cluster_distance}m")

        coords = ground_truth[['x', 'y']].values
        distances = pdist(coords, metric='euclidean')

        # Hierarchical clustering with single linkage
        Z = linkage(distances, method='single')
        clusters = fcluster(Z, t=self.cluster_distance, criterion='distance')

        n_clusters = len(np.unique(clusters))
        cluster_sizes = np.bincount(clusters)

        logger.info(f"\nClustering results:")
        logger.info(f"  Total points: {len(ground_truth)}")
        logger.info(f"  Number of clusters: {n_clusters}")
        logger.info(f"  Singleton clusters (size=1): {(cluster_sizes == 1).sum()}")
        logger.info(f"  Multi-point clusters (size>1): {(cluster_sizes > 1).sum()}")
        logger.info(f"  Max cluster size: {cluster_sizes.max()}")
        logger.info(f"  Mean cluster size: {cluster_sizes.mean():.2f}")

        return clusters

    def split(
        self,
        ground_truth: pd.DataFrame,
        stratify_by_class: bool = True
    ):
        """
        Generate K-Fold splits based on spatial clusters

        Args:
            ground_truth: DataFrame with ['x', 'y', 'label']
            stratify_by_class: Try to maintain class balance

        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"SPATIAL {self.n_splits}-FOLD CROSS-VALIDATION")
        logger.info(f"{'='*70}")

        # Step 1: Cluster points
        clusters = self.cluster_points(ground_truth)

        # Step 2: Create cluster DataFrame
        unique_clusters = np.unique(clusters)
        cluster_info = []

        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_points = ground_truth[cluster_mask]

            # Get majority class for this cluster
            majority_class = cluster_points['label'].mode()[0]
            n_points = len(cluster_points)

            cluster_info.append({
                'cluster_id': cluster_id,
                'n_points': n_points,
                'majority_class': majority_class
            })

        cluster_df = pd.DataFrame(cluster_info)
        logger.info(f"Created {len(cluster_df)} clusters for K-Fold splitting")

        # Step 3: Shuffle clusters if requested
        if self.shuffle:
            cluster_ids = cluster_df['cluster_id'].values.copy()
            np.random.seed(self.random_state)
            np.random.shuffle(cluster_ids)
        else:
            cluster_ids = cluster_df['cluster_id'].values

        # Step 4: Split clusters into K folds (stratified by majority class if requested)
        if stratify_by_class:
            # Stratified K-Fold on clusters
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

            # Get majority class for each cluster (in order of cluster_ids)
            cluster_id_to_class = dict(zip(cluster_df['cluster_id'], cluster_df['majority_class']))
            cluster_classes = np.array([cluster_id_to_class[cid] for cid in cluster_ids])

            fold_iterator = skf.split(cluster_ids, cluster_classes)
        else:
            # Simple K-Fold on clusters
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            fold_iterator = kf.split(cluster_ids)

        # Step 5: For each fold, get point indices
        for fold_idx, (train_cluster_idx, test_cluster_idx) in enumerate(fold_iterator):
            train_clusters = cluster_ids[train_cluster_idx]
            test_clusters = cluster_ids[test_cluster_idx]

            # Get point indices for this fold
            train_indices = np.where(np.isin(clusters, train_clusters))[0]
            test_indices = np.where(np.isin(clusters, test_clusters))[0]

            # Log fold statistics
            logger.info(f"\n{'='*70}")
            logger.info(f"FOLD {fold_idx + 1}/{self.n_splits} - SPLIT STATISTICS")
            logger.info(f"{'='*70}")

            total_points = len(ground_truth)
            logger.info(f"Total points: {total_points}")

            logger.info(f"\nTrain set:")
            logger.info(f"  Points: {len(train_indices)} ({len(train_indices)/total_points*100:.2f}%)")
            logger.info(f"  Clusters: {len(train_clusters)}")
            train_labels = ground_truth.iloc[train_indices]['label']
            unique_train_labels = np.unique(train_labels)
            for label in unique_train_labels:
                logger.info(f"  Class {label}: {(train_labels == label).sum()}")

            logger.info(f"\nTest set:")
            logger.info(f"  Points: {len(test_indices)} ({len(test_indices)/total_points*100:.2f}%)")
            logger.info(f"  Clusters: {len(test_clusters)}")
            test_labels = ground_truth.iloc[test_indices]['label']
            unique_test_labels = np.unique(test_labels)
            for label in unique_test_labels:
                logger.info(f"  Class {label}: {(test_labels == label).sum()}")

            logger.info(f"{'='*70}")

            yield train_indices, test_indices


class SpatialKFoldWithFixedTest:
    """
    Spatial-aware K-Fold CV with Fixed Test Set
    
    Workflow:
    1. Split off a fixed test set (e.g., 15%)
    2. Perform K-Fold CV on the remaining data (e.g., 85%)
    3. Train final model on all remaining data (85%)
    4. Evaluate final model on fixed test set (15%)
    
    This ensures:
    - Test set is never seen during CV
    - No spatial overlap between any splits
    - Proper evaluation on truly held-out data
    """

    def __init__(
        self,
        test_size: float = 0.15,
        n_splits: int = 5,
        cluster_distance: float = 50.0,
        random_state: int = 42,
        shuffle: bool = True
    ):
        """
        Initialize SpatialKFoldWithFixedTest

        Args:
            test_size: Proportion of data for fixed test set (e.g., 0.15 = 15%)
            n_splits: Number of folds for CV on remaining data
            cluster_distance: Distance threshold for clustering (meters)
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle clusters before splitting
        """
        self.test_size = test_size
        self.n_splits = n_splits
        self.cluster_distance = cluster_distance
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

        # Check if using random split or spatial clustering
        if self.cluster_distance is None:
            # Use simple random split (stratified)
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
                random_state=self.random_state
            )
            
            metadata = {'method': 'random_split', 'clusters': None}
            
        else:
            # Use spatial clustering
            logger.info(f"Using SPATIAL CLUSTERING (cluster_distance={self.cluster_distance}m)")
            
            splitter = SpatialSplitter(
                cluster_distance=self.cluster_distance,
                train_size=(1 - self.test_size),
                val_size=0.0,  # No validation set in this step
                test_size=self.test_size,
                random_state=self.random_state
            )

            trainval_indices, _, test_indices, metadata = splitter.spatial_split(
                ground_truth,
                stratify_by_class=stratify_by_class,
                verify=True
            )

        logger.info(f"\n✅ Fixed test set created:")
        logger.info(f"   Train+Val: {len(trainval_indices)} samples ({len(trainval_indices)/len(ground_truth)*100:.1f}%)")
        logger.info(f"   Test (FIXED): {len(test_indices)} samples ({len(test_indices)/len(ground_truth)*100:.1f}%)")

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
            Tuple of (train_indices, val_indices) for each fold
        """
        logger.info(f"\n{'='*70}")
        logger.info("STEP 2: K-FOLD CV ON TRAIN+VAL DATA")
        logger.info(f"{'='*70}")
        logger.info(f"Number of folds: {self.n_splits}")
        logger.info(f"Data size: {len(ground_truth_trainval)} samples")

        # Check if using random split or spatial clustering
        if self.cluster_distance is None:
            # Use simple StratifiedKFold or KFold
            logger.info("Using RANDOM K-FOLD (stratified by class)")
            
            if stratify_by_class:
                from sklearn.model_selection import StratifiedKFold
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
                    logger.info(f"  Train: {len(train_indices)} samples")
                    logger.info(f"  Val: {len(val_indices)} samples")
                    yield fold_idx, train_indices, val_indices
            else:
                from sklearn.model_selection import KFold
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
        else:
            # Use SpatialKFold for CV
            logger.info(f"Using SPATIAL K-FOLD (cluster_distance={self.cluster_distance}m)")
            
            kfold = SpatialKFold(
                n_splits=self.n_splits,
                cluster_distance=self.cluster_distance,
                random_state=self.random_state,
                shuffle=self.shuffle
            )

            for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(
                ground_truth_trainval,
                stratify_by_class=stratify_by_class
            )):
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