"""
Compare and visualize results from all 3 CNN models + Random Forest

Creates comparison visualizations:
1. Side-by-side probability maps (4 models)
2. Side-by-side binary maps (4 models)
3. Deforestation statistics comparison
4. Agreement analysis between models
"""
import sys
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

project_root = Path(__file__).parent


def load_map(filepath):
    """Load GeoTIFF map"""
    with rasterio.open(filepath) as src:
        return src.read(1)


def main():
    print("=" * 80)
    print("COMPARING ALL MODELS")
    print("=" * 80)
    print()

    outputs_dir = project_root / 'outputs'
    figures_dir = project_root / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Model configurations
    models = [
        {
            'name': 'Spatial CNN',
            'prob_file': 'spatial_cnn_probability_map.tif',
            'binary_file': 'spatial_cnn_binary_map.tif',
            'color': 'steelblue'
        },
        {
            'name': 'MultiScale CNN',
            'prob_file': 'multiscale_cnn_probability_map.tif',
            'binary_file': 'multiscale_cnn_binary_map.tif',
            'color': 'coral'
        },
        {
            'name': 'Shallow U-Net',
            'prob_file': 'shallow_unet_probability_map.tif',
            'binary_file': 'shallow_unet_binary_map.tif',
            'color': 'mediumseagreen'
        },
        {
            'name': 'Random Forest',
            'prob_file': 'random_forest_probability_map.tif',
            'binary_file': 'random_forest_binary_map.tif',
            'color': 'mediumpurple'
        }
    ]

    # Load maps
    print("Loading probability maps...")
    prob_maps = {}
    binary_maps = {}

    for model in models:
        prob_path = outputs_dir / model['prob_file']
        binary_path = outputs_dir / model['binary_file']

        if prob_path.exists() and binary_path.exists():
            prob_maps[model['name']] = load_map(prob_path)
            binary_maps[model['name']] = load_map(binary_path)
            print(f"  Loaded: {model['name']}")
        else:
            print(f"  Missing: {model['name']}")

    if len(prob_maps) == 0:
        print("\nError: No maps found. Please run inference first.")
        return

    print(f"\nLoaded {len(prob_maps)} models")
    print()

    # 1. Compare probability maps (2x2 grid)
    print("Creating probability maps comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    for i, model in enumerate(models):
        if model['name'] in prob_maps:
            im = axes[i].imshow(prob_maps[model['name']], cmap='RdYlGn_r', vmin=0, vmax=1)
            axes[i].set_title(f"{model['name']}\nProbability Map", fontsize=14, fontweight='bold')
            axes[i].set_xlabel('X (pixels)', fontsize=11)
            axes[i].set_ylabel('Y (pixels)', fontsize=11)
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        else:
            axes[i].axis('off')
            axes[i].text(0.5, 0.5, f"{model['name']}\nNot Available",
                        ha='center', va='center', fontsize=14, transform=axes[i].transAxes)

    plt.suptitle('Deforestation Probability Maps - All Models Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(figures_dir / 'all_models_probability_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all_models_probability_comparison.png")

    # 2. Compare binary maps (2x2 grid)
    print("Creating binary maps comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    colors = ['#2ecc71', '#e74c3c']  # Green, Red
    cmap = ListedColormap(colors)

    for i, model in enumerate(models):
        if model['name'] in binary_maps:
            im = axes[i].imshow(binary_maps[model['name']], cmap=cmap, vmin=0, vmax=1)
            axes[i].set_title(f"{model['name']}\nBinary Classification", fontsize=14, fontweight='bold')
            axes[i].set_xlabel('X (pixels)', fontsize=11)
            axes[i].set_ylabel('Y (pixels)', fontsize=11)
            cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04, ticks=[0.25, 0.75])
            cbar.ax.set_yticklabels(['No Deforestation', 'Deforestation'], fontsize=10)
        else:
            axes[i].axis('off')
            axes[i].text(0.5, 0.5, f"{model['name']}\nNot Available",
                        ha='center', va='center', fontsize=14, transform=axes[i].transAxes)

    plt.suptitle('Binary Deforestation Maps - All Models Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(figures_dir / 'all_models_binary_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all_models_binary_comparison.png")

    # 3. Statistics comparison
    print("Creating statistics comparison...")
    pixel_area_m2 = 10.0 * 10.0

    stats_data = []
    for model in models:
        if model['name'] in binary_maps:
            binary_map = binary_maps[model['name']]
            prob_map = prob_maps[model['name']]

            total_pixels = binary_map.size
            defor_pixels = binary_map.sum()
            defor_percentage = (defor_pixels / total_pixels) * 100
            defor_area_km2 = defor_pixels * pixel_area_m2 / 1e6

            stats_data.append({
                'Model': model['name'],
                'Deforestation (%)': defor_percentage,
                'Area (km²)': defor_area_km2,
                'Mean Probability': prob_map.mean(),
                'Color': model['color']
            })

    if len(stats_data) > 0:
        # Bar chart comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Deforestation percentage
        model_names = [s['Model'] for s in stats_data]
        defor_pcts = [s['Deforestation (%)'] for s in stats_data]
        colors = [s['Color'] for s in stats_data]

        axes[0].bar(range(len(model_names)), defor_pcts, color=colors, edgecolor='black', alpha=0.8)
        axes[0].set_xticks(range(len(model_names)))
        axes[0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0].set_ylabel('Deforestation (%)', fontsize=12)
        axes[0].set_title('Deforestation Percentage', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(defor_pcts):
            axes[0].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=10, fontweight='bold')

        # Deforestation area
        defor_areas = [s['Area (km²)'] for s in stats_data]
        axes[1].bar(range(len(model_names)), defor_areas, color=colors, edgecolor='black', alpha=0.8)
        axes[1].set_xticks(range(len(model_names)))
        axes[1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1].set_ylabel('Area (km²)', fontsize=12)
        axes[1].set_title('Deforestation Area', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(defor_areas):
            axes[1].text(i, v + 20, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')

        # Mean probability
        mean_probs = [s['Mean Probability'] for s in stats_data]
        axes[2].bar(range(len(model_names)), mean_probs, color=colors, edgecolor='black', alpha=0.8)
        axes[2].set_xticks(range(len(model_names)))
        axes[2].set_xticklabels(model_names, rotation=45, ha='right')
        axes[2].set_ylabel('Mean Probability', fontsize=12)
        axes[2].set_title('Average Deforestation Probability', fontsize=14, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        for i, v in enumerate(mean_probs):
            axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

        plt.suptitle('Deforestation Statistics - All Models Comparison',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(figures_dir / 'all_models_statistics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: all_models_statistics_comparison.png")

    # 4. Model agreement analysis (CNN models only)
    print("Creating model agreement analysis...")
    cnn_models = ['Spatial CNN', 'MultiScale CNN', 'Shallow U-Net']
    cnn_binary_maps = [binary_maps[name] for name in cnn_models if name in binary_maps]

    if len(cnn_binary_maps) >= 2:
        # Stack binary maps
        stacked = np.stack(cnn_binary_maps, axis=0)
        agreement = stacked.sum(axis=0)  # 0 to N models agree

        # Create agreement visualization
        fig, ax = plt.subplots(figsize=(14, 12))
        cmap_agreement = plt.cm.get_cmap('RdYlGn_r', len(cnn_binary_maps) + 1)
        im = ax.imshow(agreement, cmap=cmap_agreement, vmin=0, vmax=len(cnn_binary_maps))
        ax.set_title('CNN Models Agreement on Deforestation Detection',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                           ticks=range(len(cnn_binary_maps) + 1))
        cbar.set_label('Number of Models Predicting Deforestation',
                      fontsize=12, rotation=270, labelpad=25)

        plt.tight_layout()
        plt.savefig(figures_dir / 'cnn_models_agreement.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: cnn_models_agreement.png")

        # Agreement statistics
        agreement_counts = np.bincount(agreement.flatten(), minlength=len(cnn_binary_maps) + 1)
        agreement_pcts = agreement_counts / agreement.size * 100

        print("\nCNN Models Agreement:")
        for i, (count, pct) in enumerate(zip(agreement_counts, agreement_pcts)):
            print(f"  {i} models agree: {count:,} pixels ({pct:.2f}%)")

    # Save summary
    print("\nCreating summary report...")
    summary_path = outputs_dir / 'all_models_comparison_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ALL MODELS COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("DEFORESTATION STATISTICS:\n")
        f.write("-" * 80 + "\n")
        for data in stats_data:
            f.write(f"\n{data['Model']}:\n")
            f.write(f"  Deforestation: {data['Deforestation (%)']:.2f}%\n")
            f.write(f"  Area: {data['Area (km²)']:.2f} km²\n")
            f.write(f"  Mean Probability: {data['Mean Probability']:.4f}\n")

        if len(cnn_binary_maps) >= 2:
            f.write("\n\nCNN MODELS AGREEMENT:\n")
            f.write("-" * 80 + "\n")
            for i, (count, pct) in enumerate(zip(agreement_counts, agreement_pcts)):
                f.write(f"  {i} models agree: {count:,} pixels ({pct:.2f}%)\n")

    print(f"  Saved: {summary_path.name}")

    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETED")
    print("=" * 80)
    print("\nGenerated figures:")
    print("  - all_models_probability_comparison.png")
    print("  - all_models_binary_comparison.png")
    print("  - all_models_statistics_comparison.png")
    print("  - cnn_models_agreement.png")
    print("\nSummary report:")
    print("  - all_models_comparison_summary.txt")
    print()


if __name__ == "__main__":
    main()
