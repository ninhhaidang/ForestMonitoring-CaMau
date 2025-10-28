"""
Run full-image inference for ALL 3 CNN models separately

This script generates separate probability and binary maps for each model:
1. SpatialContextCNN
2. MultiScaleCNN
3. ShallowUNet

Output:
- outputs/spatial_cnn_probability_map.tif
- outputs/spatial_cnn_binary_map.tif
- outputs/multiscale_cnn_probability_map.tif
- outputs/multiscale_cnn_binary_map.tif
- outputs/shallow_unet_probability_map.tif
- outputs/shallow_unet_binary_map.tif
"""
import sys
from pathlib import Path
import numpy as np
import rasterio
import torch
from tqdm.auto import tqdm

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from models import get_model
from preprocessing import normalize_band, handle_nan


def load_full_image_s2_only(s2_2024_path, s2_2025_path):
    """Load and normalize S2-only imagery (14 channels)"""

    print("Loading S2 TIFF files...")

    # Load S2 2024
    with rasterio.open(s2_2024_path) as src:
        s2_2024 = src.read()  # (7, H, W)
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    # Load S2 2025
    with rasterio.open(s2_2025_path) as src:
        s2_2025 = src.read()  # (7, H, W)

    # Stack: (14, H, W)
    all_bands = np.concatenate([s2_2024, s2_2025], axis=0)

    # Transpose to (H, W, 14)
    all_bands = np.transpose(all_bands, (1, 2, 0))

    print(f"Loaded: {all_bands.shape} ({all_bands.dtype})")

    # Normalize
    print("Normalizing bands...")
    for c in tqdm(range(14), desc="Normalize", unit="band"):
        if np.isnan(all_bands[:, :, c]).any():
            all_bands[:, :, c] = handle_nan(all_bands[:, :, c], method='fill')

        # 0-3,7-10: S2 reflectance
        # 4-6,11-13: S2 indices
        if c in [0, 1, 2, 3, 7, 8, 9, 10]:
            all_bands[:, :, c] = normalize_band(all_bands[:, :, c], method='clip', clip_range=(0, 1))
        else:
            all_bands[:, :, c] = (all_bands[:, :, c] + 1) / 2

    return all_bands, transform, crs


def sliding_window_inference(model, image, window_size=128, stride=64, batch_size=32, device='cuda'):
    """Run sliding window inference"""

    h, w, c = image.shape

    # Initialize output
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.int32)

    # Calculate windows
    n_rows = (h - window_size) // stride + 1
    n_cols = (w - window_size) // stride + 1

    print(f"  Image size: {h} x {w}")
    print(f"  Total windows: {n_rows * n_cols:,}")
    print(f"  Batch size: {batch_size}")

    # Extract all windows
    windows = []
    positions = []

    for i in range(n_rows):
        for j in range(n_cols):
            y = i * stride
            x = j * stride

            patch = image[y:y+window_size, x:x+window_size, :]  # (128, 128, 14)
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()

            windows.append(patch_tensor)
            positions.append((y, x))

    # Process in batches
    n_batches = (len(windows) + batch_size - 1) // batch_size

    model.eval()
    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Inference", unit="batch"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(windows))

            batch_patches = torch.stack(windows[start_idx:end_idx]).to(device)
            outputs = model(batch_patches)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()

            for i, (y, x) in enumerate(positions[start_idx:end_idx]):
                prob_map[y:y+window_size, x:x+window_size] += probs[i]
                count_map[y:y+window_size, x:x+window_size] += 1

    # Average overlapping predictions
    prob_map = np.divide(prob_map, count_map, where=count_map > 0)

    return prob_map


def save_maps(prob_map, model_name, transform, crs, outputs_dir, threshold=0.5):
    """Save probability and binary maps"""

    h, w = prob_map.shape

    # Create binary map
    binary_map = (prob_map > threshold).astype(np.uint8)

    # Save probability map
    prob_path = outputs_dir / f'{model_name}_probability_map.tif'
    with rasterio.open(
        prob_path, 'w',
        driver='GTiff',
        height=h, width=w,
        count=1,
        dtype=rasterio.float32,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    # Save binary map
    binary_path = outputs_dir / f'{model_name}_binary_map.tif'
    with rasterio.open(
        binary_path, 'w',
        driver='GTiff',
        height=h, width=w,
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(binary_map, 1)

    # Calculate statistics
    pixel_area_m2 = 10.0 * 10.0
    total_pixels = prob_map.size
    deforestation_pixels = binary_map.sum()
    total_area_km2 = total_pixels * pixel_area_m2 / 1e6
    deforestation_area_km2 = deforestation_pixels * pixel_area_m2 / 1e6
    deforestation_percentage = (deforestation_pixels / total_pixels) * 100

    return {
        'prob_path': prob_path,
        'binary_path': binary_path,
        'total_area_km2': total_area_km2,
        'deforestation_area_km2': deforestation_area_km2,
        'deforestation_percentage': deforestation_percentage,
        'prob_mean': prob_map.mean(),
        'prob_std': prob_map.std()
    }


def main():
    print("=" * 80)
    print("FULL-IMAGE INFERENCE FOR ALL 3 CNN MODELS")
    print("=" * 80)
    print()

    # Paths
    data_dir = project_root / 'data' / 'raw' / 'sentinel2'
    s2_2024_path = data_dir / 'S2_2024_01_30.tif'
    s2_2025_path = data_dir / 'S2_2025_02_28.tif'

    checkpoints_dir = project_root / 'checkpoints'
    outputs_dir = project_root / 'outputs'
    outputs_dir.mkdir(exist_ok=True)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load image
    print("=" * 80)
    print("LOADING IMAGERY")
    print("=" * 80)
    image, transform, crs = load_full_image_s2_only(s2_2024_path, s2_2025_path)
    print()

    # Model configurations
    models_config = [
        {
            'name': 'spatial_cnn',
            'display_name': 'Spatial Context CNN',
            'checkpoint': 'spatial_context_cnn_best.pth',
            'model_type': 'spatial_context_cnn'
        },
        {
            'name': 'multiscale_cnn',
            'display_name': 'Multi-Scale CNN',
            'checkpoint': 'multiscale_cnn_best.pth',
            'model_type': 'multiscale_cnn'
        },
        {
            'name': 'shallow_unet',
            'display_name': 'Shallow U-Net',
            'checkpoint': 'shallow_unet_best.pth',
            'model_type': 'shallow_unet'
        }
    ]

    results = {}

    # Process each model
    for i, config in enumerate(models_config, 1):
        print("=" * 80)
        print(f"MODEL {i}/3: {config['display_name']}")
        print("=" * 80)

        model_path = checkpoints_dir / config['checkpoint']

        if not model_path.exists():
            print(f"Warning: {model_path.name} not found, skipping...")
            print()
            continue

        # Load model
        print(f"Loading model: {model_path.name}")
        model = get_model(config['model_type'], in_channels=14)
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"  Val Accuracy: {checkpoint.get('val_acc', 0)*100:.2f}%")
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        print()

        # Run inference
        print("Running inference...")
        prob_map = sliding_window_inference(
            model=model,
            image=image,
            window_size=128,
            stride=64,
            batch_size=32 if device == 'cuda' else 8,
            device=device
        )
        print()

        # Save results
        print("Saving results...")
        stats = save_maps(prob_map, config['name'], transform, crs, outputs_dir)
        results[config['name']] = stats

        print(f"  Probability map: {stats['prob_path'].name}")
        print(f"  Binary map: {stats['binary_path'].name}")
        print(f"  Deforestation: {stats['deforestation_percentage']:.2f}% ({stats['deforestation_area_km2']:.2f} km²)")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY - ALL MODELS")
    print("=" * 80)
    print()

    for config in models_config:
        name = config['name']
        if name in results:
            stats = results[name]
            print(f"{config['display_name']}:")
            print(f"  Deforestation: {stats['deforestation_percentage']:.2f}%")
            print(f"  Area: {stats['deforestation_area_km2']:.2f} km²")
            print(f"  Mean probability: {stats['prob_mean']:.4f}")
            print()

    print("=" * 80)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Output files in outputs/:")
    for config in models_config:
        name = config['name']
        if name in results:
            print(f"  - {name}_probability_map.tif")
            print(f"  - {name}_binary_map.tif")
    print()


if __name__ == "__main__":
    main()
