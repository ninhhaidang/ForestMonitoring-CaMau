"""
Feature extraction following GEE approach
Extracts only 12 features from Sentinel-2 data: NDVI, EVI, SAVI, NBR × 3 (before, after, delta)
Matches the Google Earth Engine script workflow
"""
import numpy as np
from tqdm import tqdm


def calculate_spectral_indices(patch, time_period='2024'):
    """
    Calculate spectral indices from S2 raw bands.

    Args:
        patch: Numpy array of shape (18, H, W)
        time_period: '2024' (before) or '2025' (after)

    Returns:
        Dictionary with NDVI, EVI, SAVI, NBR arrays of shape (H, W)
    """
    if time_period == '2024':
        # S2 2024: channels 0-6
        B4_Red = patch[0]    # Red
        B8_NIR = patch[1]    # NIR
        B12_SWIR2 = patch[3]  # SWIR2
        # Note: We don't have Blue band in our data, so we'll approximate EVI
    else:  # 2025
        # S2 2025: channels 9-15
        B4_Red = patch[9]
        B8_NIR = patch[10]
        B12_SWIR2 = patch[12]

    # Add small epsilon to avoid division by zero
    eps = 1e-8

    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = (B8_NIR - B4_Red) / (B8_NIR + B4_Red + eps)

    # EVI = 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
    # Since we don't have Blue band, use simplified EVI without blue correction
    # Simplified EVI = 2.5 * ((NIR - Red) / (NIR + 2.4 * Red + 1))
    evi = 2.5 * ((B8_NIR - B4_Red) / (B8_NIR + 2.4 * B4_Red + 1 + eps))

    # SAVI = ((NIR - Red) / (NIR + Red + 0.5)) * 1.5
    savi = ((B8_NIR - B4_Red) / (B8_NIR + B4_Red + 0.5 + eps)) * 1.5

    # NBR = (NIR - SWIR2) / (NIR + SWIR2)
    nbr = (B8_NIR - B12_SWIR2) / (B8_NIR + B12_SWIR2 + eps)

    return {
        'NDVI': ndvi,
        'EVI': evi,
        'SAVI': savi,
        'NBR': nbr
    }


def extract_patch_features_gee(patch):
    """
    Extract 12 features from a single 18-channel patch following GEE approach.

    **GEE-style approach: 12 features total**
    - 4 spectral indices: NDVI, EVI, SAVI, NBR
    - 3 values per index: before (2024), after (2025), delta (2025-2024)
    - Total: 4 × 3 = 12 features

    Args:
        patch: Numpy array of shape (18, H, W)
               Channel order: [S2_2024 (7), S1_2024 (2), S2_2025 (7), S1_2025 (2)]

    Returns:
        features: 1D numpy array of 12 features
    """
    # Calculate indices for both time periods
    indices_2024 = calculate_spectral_indices(patch, time_period='2024')
    indices_2025 = calculate_spectral_indices(patch, time_period='2025')

    features = []

    # Extract features for each index: before, after, delta
    for index_name in ['NDVI', 'EVI', 'SAVI', 'NBR']:
        before = indices_2024[index_name]
        after = indices_2025[index_name]

        # Extract mean values
        before_val = np.mean(before)
        after_val = np.mean(after)
        delta_val = after_val - before_val

        features.extend([before_val, after_val, delta_val])

    return np.array(features, dtype=np.float32)


def extract_features_from_patches_gee(patches, verbose=True):
    """
    Extract GEE-style features from all patches.

    Args:
        patches: Numpy array of shape (N, 18, H, W)
        verbose: Whether to show progress bar

    Returns:
        features: Numpy array of shape (N, 12)
    """
    num_samples = patches.shape[0]
    features_list = []

    iterator = tqdm(range(num_samples), desc="Extracting GEE-style features") if verbose else range(num_samples)

    for i in iterator:
        patch = patches[i]
        features = extract_patch_features_gee(patch)
        features_list.append(features)

    features_array = np.array(features_list, dtype=np.float32)

    if verbose:
        print(f"\nExtracted GEE-style features shape: {features_array.shape}")
        print(f"  - Samples: {features_array.shape[0]}")
        print(f"  - Features per sample: {features_array.shape[1]}")

    return features_array


def get_feature_names_gee():
    """
    Get descriptive names for all GEE-style features.

    Returns:
        List of feature names (12 names)
    """
    feature_names = []

    # 4 spectral indices
    index_names = ['NDVI', 'EVI', 'SAVI', 'NBR']

    # For each index: before, after, delta
    for index_name in index_names:
        feature_names.extend([
            f"b_{index_name}",  # before (2024)
            f"a_{index_name}",  # after (2025)
            f"d_{index_name}"   # delta (after - before)
        ])

    return feature_names


if __name__ == "__main__":
    # Test feature extraction
    print("Testing GEE-style feature extraction...")

    # Create dummy patch
    dummy_patch = np.random.randn(18, 64, 64).astype(np.float32)

    # Extract features
    features = extract_patch_features_gee(dummy_patch)
    print(f"Extracted {len(features)} features from one patch")

    # Get feature names
    feature_names = get_feature_names_gee()
    print(f"Total feature names: {len(feature_names)}")

    print("\nGEE-style feature names:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1:2d}. {name}")

    # Verify counts match
    assert len(features) == len(feature_names) == 12, "Feature count mismatch!"
    print("\n✓ GEE-style feature extraction test passed!")
    print(f"✓ Total features: {len(features)}")
    print("\nFeature breakdown:")
    print("  - NDVI: 3 features (before, after, delta)")
    print("  - EVI:  3 features (before, after, delta)")
    print("  - SAVI: 3 features (before, after, delta)")
    print("  - NBR:  3 features (before, after, delta)")
