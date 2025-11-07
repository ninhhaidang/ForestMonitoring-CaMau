"""
Feature extraction for traditional ML models (Random Forest)
Simplified to 27 features: before, after, delta for each band
"""
import numpy as np
from tqdm import tqdm


def extract_patch_features(patch):
    """
    Extract handcrafted features from a single 18-channel patch.

    **New simplified approach: 27 features total**
    - 9 bands (7 S2 + 2 S1)
    - 3 values per band: before (2024), after (2025), delta (2025-2024)
    - Total: 9 × 3 = 27 features

    Args:
        patch: Numpy array of shape (18, H, W)
               Channel order: [S2_2024 (7), S1_2024 (2), S2_2025 (7), S1_2025 (2)]

    Returns:
        features: 1D numpy array of 27 features
    """
    features = []

    # Band names for reference
    s2_band_names = ["B4", "B8", "B11", "B12", "NDVI", "NBR", "NDMI"]
    s1_band_names = ["VV", "VH"]

    # ============================================================
    # S2 bands (7 bands × 3 = 21 features)
    # ============================================================
    for i in range(7):
        before = patch[i]        # S2 2024: channels 0-6
        after = patch[9 + i]     # S2 2025: channels 9-15

        # Extract mean value for each time period
        b_val = np.mean(before)
        a_val = np.mean(after)
        d_val = a_val - b_val  # delta = after - before

        features.extend([b_val, a_val, d_val])

    # ============================================================
    # S1 bands (2 bands × 3 = 6 features)
    # ============================================================
    for i in range(2):
        before = patch[7 + i]    # S1 2024: channels 7-8
        after = patch[16 + i]    # S1 2025: channels 16-17

        # Extract mean value for each time period
        b_val = np.mean(before)
        a_val = np.mean(after)
        d_val = a_val - b_val

        features.extend([b_val, a_val, d_val])

    # ============================================================
    # Total: 21 + 6 = 27 features
    # ============================================================

    return np.array(features, dtype=np.float32)


def extract_features_from_patches(patches, verbose=True):
    """
    Extract features from all patches.

    Args:
        patches: Numpy array of shape (N, 18, H, W)
        verbose: Whether to show progress bar

    Returns:
        features: Numpy array of shape (N, 27)
    """
    num_samples = patches.shape[0]
    features_list = []

    iterator = tqdm(range(num_samples), desc="Extracting features") if verbose else range(num_samples)

    for i in iterator:
        patch = patches[i]
        features = extract_patch_features(patch)
        features_list.append(features)

    features_array = np.array(features_list, dtype=np.float32)

    if verbose:
        print(f"\nExtracted features shape: {features_array.shape}")
        print(f"  - Samples: {features_array.shape[0]}")
        print(f"  - Features per sample: {features_array.shape[1]}")

    return features_array


def get_feature_names():
    """
    Get descriptive names for all extracted features.

    Returns:
        List of feature names (27 names)
    """
    feature_names = []

    # S2 band names
    s2_band_names = ["B4", "B8", "B11", "B12", "NDVI", "NBR", "NDMI"]

    # S1 band names
    s1_band_names = ["VV", "VH"]

    # S2 features (7 bands × 3 = 21)
    for band_name in s2_band_names:
        feature_names.extend([
            f"b_{band_name}",  # before (2024)
            f"a_{band_name}",  # after (2025)
            f"d_{band_name}"   # delta (after - before)
        ])

    # S1 features (2 bands × 3 = 6)
    for band_name in s1_band_names:
        feature_names.extend([
            f"b_{band_name}",
            f"a_{band_name}",
            f"d_{band_name}"
        ])

    return feature_names


if __name__ == "__main__":
    # Test feature extraction
    print("Testing feature extraction...")

    # Create dummy patch
    dummy_patch = np.random.randn(18, 64, 64).astype(np.float32)

    # Extract features
    features = extract_patch_features(dummy_patch)
    print(f"Extracted {len(features)} features from one patch")

    # Get feature names
    feature_names = get_feature_names()
    print(f"Total feature names: {len(feature_names)}")

    print("\nFeature names:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1:2d}. {name}")

    # Verify counts match
    assert len(features) == len(feature_names) == 27, "Feature count mismatch!"
    print("\n✓ Feature extraction test passed!")
    print(f"✓ Total features: {len(features)}")
