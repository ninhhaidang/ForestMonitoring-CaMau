"""
Analyze spatial clustering of ground truth points
To determine optimal splitting strategy for avoiding data leakage
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

from config import GROUND_TRUTH_CSV, RESULTS_DIR


def analyze_point_spacing():
    """Analyze spacing between ground truth points"""

    # Load ground truth
    df = pd.read_csv(GROUND_TRUTH_CSV)
    print(f'\n{"="*70}')
    print("SPATIAL CLUSTERING ANALYSIS")
    print("="*70)
    print(f'\nTotal points: {len(df)}')
    print(f'Class distribution:')
    print(df['label'].value_counts())

    # Calculate pairwise distances
    coords = df[['x', 'y']].values
    distances = pdist(coords, metric='euclidean')

    print(f'\n{"="*70}')
    print("DISTANCE STATISTICS (meters)")
    print("="*70)
    print(f'Min distance:        {distances.min():8.2f}m')
    print(f'Max distance:        {distances.max():8.2f}m')
    print(f'Mean distance:       {distances.mean():8.2f}m')
    print(f'Median distance:     {np.median(distances):8.2f}m')
    print(f'25th percentile:     {np.percentile(distances, 25):8.2f}m')
    print(f'10th percentile:     {np.percentile(distances, 10):8.2f}m')
    print(f'5th percentile:      {np.percentile(distances, 5):8.2f}m')

    # Proximity analysis
    dist_matrix = squareform(distances)
    print(f'\n{"="*70}')
    print("PROXIMITY ANALYSIS")
    print("="*70)
    print(f'Pairs within 10m (1 pixel):   {(distances < 10).sum():6d}')
    print(f'Pairs within 30m (3 pixels):  {(distances < 30).sum():6d}')
    print(f'Pairs within 50m (5 pixels):  {(distances < 50).sum():6d}')
    print(f'Pairs within 100m (10 pixels): {(distances < 100).sum():6d}')
    print(f'Pairs within 150m (15 pixels): {(distances < 150).sum():6d}')

    # Closest neighbor analysis
    np.fill_diagonal(dist_matrix, np.inf)
    min_distances = dist_matrix.min(axis=1)

    print(f'\n{"="*70}')
    print("CLOSEST NEIGHBOR ANALYSIS")
    print("="*70)
    print(f'Points with neighbor < 30m:  {(min_distances < 30).sum():6d} ({(min_distances < 30).sum()/len(df)*100:5.2f}%)')
    print(f'Points with neighbor < 50m:  {(min_distances < 50).sum():6d} ({(min_distances < 50).sum()/len(df)*100:5.2f}%)')
    print(f'Points with neighbor < 100m: {(min_distances < 100).sum():6d} ({(min_distances < 100).sum()/len(df)*100:5.2f}%)')
    print(f'Points with neighbor < 150m: {(min_distances < 150).sum():6d} ({(min_distances < 150).sum()/len(df)*100:5.2f}%)')
    print(f'Minimum neighbor distance:   {min_distances.min():8.2f}m')
    print(f'Mean neighbor distance:      {min_distances.mean():8.2f}m')

    # Hierarchical clustering analysis
    print(f'\n{"="*70}')
    print("HIERARCHICAL CLUSTERING (eps=50m)")
    print("="*70)

    # Using hierarchical clustering
    Z = linkage(distances, method='single')
    clusters = fcluster(Z, t=50, criterion='distance')

    n_clusters = len(np.unique(clusters))
    print(f'Number of clusters: {n_clusters}')

    cluster_sizes = np.bincount(clusters)
    print(f'\nCluster size distribution:')
    print(f'  Singleton (size=1):     {(cluster_sizes == 1).sum()} clusters')
    print(f'  Small (size=2-5):       {((cluster_sizes >= 2) & (cluster_sizes <= 5)).sum()} clusters')
    print(f'  Medium (size=6-10):     {((cluster_sizes >= 6) & (cluster_sizes <= 10)).sum()} clusters')
    print(f'  Large (size>10):        {(cluster_sizes > 10).sum()} clusters')
    print(f'  Max cluster size:       {cluster_sizes.max()}')

    # Add cluster info to dataframe
    df['cluster_id'] = clusters

    # Analyze class distribution in clusters
    print(f'\n{"="*70}')
    print("CLASS DISTRIBUTION IN CLUSTERS")
    print("="*70)

    mixed_clusters = 0
    for cluster_id in range(1, n_clusters + 1):
        cluster_df = df[df['cluster_id'] == cluster_id]
        if len(cluster_df) > 1:
            n_class0 = (cluster_df['label'] == 0).sum()
            n_class1 = (cluster_df['label'] == 1).sum()
            if n_class0 > 0 and n_class1 > 0:
                mixed_clusters += 1

    print(f'Clusters with mixed classes: {mixed_clusters} / {n_clusters}')

    # Recommendations
    print(f'\n{"="*70}')
    print("RECOMMENDATIONS")
    print("="*70)

    if distances.min() < 30:
        print('⚠️  WARNING: Some points are very close (<30m)')
        print('   → Risk of data leakage with patch size >= 3x3')

    if (min_distances < 50).sum() > len(df) * 0.2:
        print('⚠️  WARNING: >20% points have neighbors within 50m')
        print('   → Use GROUP-AWARE SPLITTING (cluster-based)')

    print(f'\n✅ RECOMMENDED STRATEGY:')
    print(f'   1. Patch size: 3x3 (30m x 30m)')
    print(f'   2. Splitting: Cluster-based (eps=50m)')
    print(f'   3. This ensures NO overlap between train/test patches')

    print(f'\n{"="*70}\n')

    return df, distances, clusters, min_distances


if __name__ == '__main__':
    df, distances, clusters, min_distances = analyze_point_spacing()
    print("✓ Spatial clustering analysis completed!")
