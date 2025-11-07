"""
Complete GEE-style Workflow - Single Script
============================================
- Uses 27 features (7 S2 bands + 2 S1 bands × 3: before, after, delta)
- Random Forest with 100 trees (matching GEE configuration)
- Morphological operations for smoothing
- Complete visualization and comparison

Usage:
    python gee_workflow.py
"""
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from feature_extraction import extract_features_from_patches, extract_patch_features, get_feature_names


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Paths
    'train_patches': '../data/patches/train_patches.npz',
    'val_patches': '../data/patches/val_patches.npz',
    'full_area_dir': '../data/full_area',
    'model_output': './models/rf_gee_100trees.pkl',
    'results_dir': './results',

    # Model parameters
    'n_estimators': 100,
    'random_state': 42,

    # Inference parameters
    'patch_size': 64,
    'stride': 32,
    'batch_size': 32,

    # Morphological operations
    'apply_morphology': True,
    'morphology_kernel': 3,
    'threshold': 0.5
}


# ============================================================================
# STEP 1: TRAIN RANDOM FOREST
# ============================================================================
def train_model(config):
    """Train Random Forest with 100 trees and 27 features."""
    print("\n" + "=" * 80)
    print("STEP 1: TRAINING RANDOM FOREST")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Trees: {config['n_estimators']}")
    print(f"  - Features: 27 (7 S2 bands + 2 S1 bands × 3)")
    print(f"  - Random state: {config['random_state']}")
    print()

    # Load training data
    print("Loading training data...")
    train_data = np.load(config['train_patches'])
    train_patches = train_data['patches']
    train_labels = train_data['labels']
    print(f"✓ Train patches: {train_patches.shape}")
    print(f"  - No Forest: {np.sum(train_labels == 0)}")
    print(f"  - Deforestation: {np.sum(train_labels == 1)}")
    print()

    # Load validation data
    print("Loading validation data...")
    val_data = np.load(config['val_patches'])
    val_patches = val_data['patches']
    val_labels = val_data['labels']
    print(f"✓ Val patches: {val_patches.shape}")
    print(f"  - No Forest: {np.sum(val_labels == 0)}")
    print(f"  - Deforestation: {np.sum(val_labels == 1)}")
    print()

    # Extract features
    print("Extracting features from training patches...")
    train_features = extract_features_from_patches(train_patches, verbose=True)

    print("\nExtracting features from validation patches...")
    val_features = extract_features_from_patches(val_patches, verbose=True)
    print()

    # Train Random Forest
    print("Training Random Forest...")
    print(f"  - Samples: {train_features.shape[0]:,}")
    print(f"  - Features: {train_features.shape[1]}")
    print(f"  - Trees: {config['n_estimators']}")
    print()

    rf_model = RandomForestClassifier(
        n_estimators=config['n_estimators'],
        random_state=config['random_state'],
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        verbose=1
    )

    rf_model.fit(train_features, train_labels)
    print("\n✓ Training completed!")
    print()

    # Evaluate
    print("Evaluating on validation set...")
    val_preds = rf_model.predict(val_features)
    val_accuracy = accuracy_score(val_labels, val_preds)

    print(f"\n✓ Validation Accuracy: {val_accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(val_labels, val_preds))
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds,
                                target_names=['No Forest', 'Deforestation'],
                                digits=4))

    # Feature importance
    print("Feature Importance (Top 15):")
    feature_names = get_feature_names()
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i in range(min(15, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1:2d}. {feature_names[idx]:12s}: {importances[idx]:.4f}")
    print()

    # Save model
    Path(config['model_output']).parent.mkdir(parents=True, exist_ok=True)
    with open(config['model_output'], 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"✓ Model saved to: {config['model_output']}")
    print()

    return rf_model, val_accuracy


# ============================================================================
# STEP 2: SLIDING WINDOW INFERENCE
# ============================================================================
def sliding_window_inference(s1_2024, s1_2025, s2_2024, s2_2025, forest_mask,
                             model, patch_size=64, stride=32, batch_size=32):
    """Perform sliding window inference on full area."""
    H, W = forest_mask.shape
    probability_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.int32)

    # Generate all window positions
    positions = []
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            positions.append((r, c))

    print(f"  - Total windows: {len(positions):,}")
    print(f"  - Batch size: {batch_size}")

    # Process in batches
    for batch_start in tqdm(range(0, len(positions), batch_size), desc="  Processing batches"):
        batch_end = min(batch_start + batch_size, len(positions))
        batch_positions = positions[batch_start:batch_end]

        patches_batch = []
        valid_positions = []

        for (r, c) in batch_positions:
            forest_patch = forest_mask[r:r+patch_size, c:c+patch_size]

            if np.sum(forest_patch) == 0:
                continue

            # Extract patches from all sources
            s1_2024_patch = s1_2024[:, r:r+patch_size, c:c+patch_size]
            s1_2025_patch = s1_2025[:, r:r+patch_size, c:c+patch_size]
            s2_2024_patch = s2_2024[:, r:r+patch_size, c:c+patch_size]
            s2_2025_patch = s2_2025[:, r:r+patch_size, c:c+patch_size]

            # Combine into 18-channel patch
            patch = np.concatenate([s2_2024_patch, s1_2024_patch,
                                   s2_2025_patch, s1_2025_patch], axis=0)

            patches_batch.append(patch)
            valid_positions.append((r, c))

        if len(patches_batch) == 0:
            continue

        # Extract features and predict
        patches_array = np.array(patches_batch)
        features_batch = []
        for patch in patches_array:
            features = extract_patch_features(patch)
            features_batch.append(features)
        features_array = np.array(features_batch)

        # Predict probabilities
        probs = model.predict_proba(features_array)[:, 1]

        # Update maps
        for prob, (r, c) in zip(probs, valid_positions):
            probability_map[r:r+patch_size, c:c+patch_size] += prob
            count_map[r:r+patch_size, c:c+patch_size] += 1

    # Average probabilities
    count_map[count_map == 0] = 1
    probability_map = probability_map / count_map
    probability_map[forest_mask == 0] = 0

    return probability_map, count_map


# ============================================================================
# STEP 3: MORPHOLOGICAL OPERATIONS
# ============================================================================
def apply_morphological_operations(binary_map, kernel_size=3):
    """Apply morphological operations to smooth the binary map."""
    binary_uint8 = (binary_map * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Closing: fill holes
    closed = cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, kernel)

    # Opening: remove noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    smoothed = (opened > 127).astype(np.uint8)
    return smoothed


# ============================================================================
# STEP 4: INFERENCE AND VISUALIZATION
# ============================================================================
def run_inference(model, config):
    """Run inference and generate visualizations."""
    print("\n" + "=" * 80)
    print("STEP 2: FULL AREA INFERENCE")
    print("=" * 80)
    print()

    # Load full area data
    print("Loading full area data...")
    data_dir = config['full_area_dir']
    s1_2024 = np.load(f"{data_dir}/s1_2024.npy")
    s1_2025 = np.load(f"{data_dir}/s1_2025.npy")
    s2_2024 = np.load(f"{data_dir}/s2_2024.npy")
    s2_2025 = np.load(f"{data_dir}/s2_2025.npy")
    forest_mask = np.load(f"{data_dir}/forest_mask.npy")

    print(f"✓ S1 2024: {s1_2024.shape}")
    print(f"✓ S1 2025: {s1_2025.shape}")
    print(f"✓ S2 2024: {s2_2024.shape}")
    print(f"✓ S2 2025: {s2_2025.shape}")
    print(f"✓ Forest mask: {forest_mask.shape}")
    print(f"  - Forest pixels: {np.sum(forest_mask):,}")
    print()

    # Run sliding window inference
    print("Running sliding window inference...")
    probability_map, count_map = sliding_window_inference(
        s1_2024, s1_2025, s2_2024, s2_2025, forest_mask, model,
        patch_size=config['patch_size'],
        stride=config['stride'],
        batch_size=config['batch_size']
    )
    print(f"\n✓ Probability map generated: {probability_map.shape}")
    print()

    # Generate binary maps
    print("Generating binary maps...")
    binary_map = (probability_map >= config['threshold']).astype(np.uint8)
    deforestation_pct = (np.sum(binary_map) / np.sum(forest_mask)) * 100
    print(f"✓ Binary map (threshold={config['threshold']})")
    print(f"  - Deforestation: {deforestation_pct:.2f}%")
    print()

    # Apply morphological operations
    if config['apply_morphology']:
        print("Applying morphological operations...")
        binary_map_smooth = apply_morphological_operations(
            binary_map,
            kernel_size=config['morphology_kernel']
        )
        deforestation_pct_smooth = (np.sum(binary_map_smooth) / np.sum(forest_mask)) * 100
        print(f"✓ Smoothed binary map")
        print(f"  - Deforestation (smoothed): {deforestation_pct_smooth:.2f}%")
        print(f"  - Change from original: {deforestation_pct_smooth - deforestation_pct:+.2f}%")
        print()
    else:
        binary_map_smooth = None
        deforestation_pct_smooth = None

    # Visualize
    print("Creating visualizations...")
    results_dir = Path(config['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    if config['apply_morphology']:
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))

        # Plot 1: Probability map
        im1 = axes[0, 0].imshow(probability_map, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[0, 0].set_title('Probability Map\n(GEE-style RF: 100 trees, 27 features)',
                            fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # Plot 2: Binary map
        axes[0, 1].imshow(forest_mask, cmap='gray', alpha=0.3)
        axes[0, 1].imshow(binary_map, cmap='Reds', alpha=0.7)
        axes[0, 1].set_title(f'Binary Map (threshold={config["threshold"]})\n'
                           f'Deforestation: {deforestation_pct:.2f}%',
                           fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # Plot 3: Smoothed binary map
        axes[1, 0].imshow(forest_mask, cmap='gray', alpha=0.3)
        axes[1, 0].imshow(binary_map_smooth, cmap='Reds', alpha=0.7)
        axes[1, 0].set_title(f'Smoothed Binary Map (Morphology kernel={config["morphology_kernel"]})\n'
                           f'Deforestation: {deforestation_pct_smooth:.2f}%',
                           fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # Plot 4: Difference map
        diff_map = binary_map_smooth.astype(int) - binary_map.astype(int)
        im4 = axes[1, 1].imshow(diff_map, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, 1].set_title(f'Difference (Smoothed - Original)\n'
                           f'Added: {np.sum(diff_map == 1):,} | Removed: {np.sum(diff_map == -1):,}',
                           fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04,
                    ticks=[-1, 0, 1], label='Change')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Plot 1: Probability map
        im1 = axes[0].imshow(probability_map, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[0].set_title('Probability Map', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot 2: Binary map
        axes[1].imshow(forest_mask, cmap='gray', alpha=0.3)
        axes[1].imshow(binary_map, cmap='Reds', alpha=0.7)
        axes[1].set_title(f'Binary Map\nDeforestation: {deforestation_pct:.2f}%',
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')

    plt.suptitle('GEE-Style Workflow: RF (100 trees, 27 features) + Morphological Smoothing',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    save_path = results_dir / "gee_workflow_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {save_path}")
    plt.close()

    # Save results
    print("\nSaving results...")
    np.save(results_dir / "probability_map.npy", probability_map)
    np.save(results_dir / "binary_map.npy", binary_map)
    if config['apply_morphology']:
        np.save(results_dir / "binary_map_smooth.npy", binary_map_smooth)

    print(f"✓ All results saved to: {results_dir}")
    print()

    return {
        'probability_map': probability_map,
        'binary_map': binary_map,
        'binary_map_smooth': binary_map_smooth,
        'deforestation_pct': deforestation_pct,
        'deforestation_pct_smooth': deforestation_pct_smooth
    }


# ============================================================================
# MAIN WORKFLOW
# ============================================================================
def main():
    """Run complete GEE-style workflow."""
    print("\n" + "=" * 80)
    print("GEE-STYLE WORKFLOW - COMPLETE PIPELINE")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  - Features: 27 (7 S2 bands + 2 S1 bands × 3)")
    print(f"  - Model: Random Forest with {CONFIG['n_estimators']} trees")
    print(f"  - Patch size: {CONFIG['patch_size']}")
    print(f"  - Stride: {CONFIG['stride']}")
    print(f"  - Morphology: {CONFIG['apply_morphology']} (kernel={CONFIG['morphology_kernel']})")
    print(f"  - Threshold: {CONFIG['threshold']}")
    print("=" * 80)

    # Step 1: Train model
    model, val_accuracy = train_model(CONFIG)

    # Step 2: Run inference
    results = run_inference(model, CONFIG)

    # Final summary
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE!")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Model: Random Forest ({CONFIG['n_estimators']} trees, 27 features)")
    print(f"  - Validation accuracy: {val_accuracy:.4f}")
    print(f"  - Deforestation (original): {results['deforestation_pct']:.2f}%")
    if CONFIG['apply_morphology']:
        print(f"  - Deforestation (smoothed): {results['deforestation_pct_smooth']:.2f}%")
    print(f"\nResults saved to: {CONFIG['results_dir']}/")
    print("  - probability_map.npy")
    print("  - binary_map.npy")
    if CONFIG['apply_morphology']:
        print("  - binary_map_smooth.npy")
    print("  - gee_workflow_results.png")
    print("\nModel saved to:", CONFIG['model_output'])
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
