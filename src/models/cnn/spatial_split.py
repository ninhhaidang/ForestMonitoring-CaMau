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
        logger.info(f"  Class 0: {(train_labels == 0).sum()}")
        logger.info(f"  Class 1: {(train_labels == 1).sum()}")

        logger.info(f"\nValidation set:")
        logger.info(f"  Points: {len(val_indices)} ({len(val_indices)/total_points*100:.2f}%)")
        logger.info(f"  Clusters: {len(val_clusters)}")
        val_labels = ground_truth.iloc[val_indices]['label']
        logger.info(f"  Class 0: {(val_labels == 0).sum()}")
        logger.info(f"  Class 1: {(val_labels == 1).sum()}")

        logger.info(f"\nTest set:")
        logger.info(f"  Points: {len(test_indices)} ({len(test_indices)/total_points*100:.2f}%)")
        logger.info(f"  Clusters: {len(test_clusters)}")
        test_labels = ground_truth.iloc[test_indices]['label']
        logger.info(f"  Class 0: {(test_labels == 0).sum()}")
        logger.info(f"  Class 1: {(test_labels == 1).sum()}")

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
