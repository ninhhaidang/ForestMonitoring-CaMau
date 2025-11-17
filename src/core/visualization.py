"""
STEP 9: Visualization
Create visualizations for model evaluation and results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import logging
from pathlib import Path

from config import (
    VIZ_CONFIG, OUTPUT_FILES, PLOTS_DIR,
    LOG_CONFIG
)

# Setup logging
logging.basicConfig(**LOG_CONFIG)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = VIZ_CONFIG['dpi']


class Visualizer:
    """Class to create visualizations"""

    def __init__(self, config: dict = None):
        """
        Initialize Visualizer

        Args:
            config: Visualization configuration (default: from config)
        """
        self.config = config if config is not None else VIZ_CONFIG

    def plot_confusion_matrix(self, cm: np.ndarray, title: str,
                             labels: list = None,
                             ax=None, cmap='Blues'):
        """
        Plot confusion matrix

        Args:
            cm: Confusion matrix
            title: Plot title
            labels: Class labels
            ax: Matplotlib axis (if None, creates new figure)
            cmap: Color map

        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        if labels is None:
            labels = ['No Deforestation', 'Deforestation']

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)

        # Add accuracy
        accuracy = np.trace(cm) / np.sum(cm)
        ax.text(0.5, -0.15, f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)',
               ha='center', va='center', transform=ax.transAxes,
               fontsize=11, fontweight='bold')

        return ax

    def plot_confusion_matrices(self, val_metrics: dict, test_metrics: dict,
                               output_path=None):
        """
        Plot confusion matrices for validation and test sets

        Args:
            val_metrics: Validation metrics dictionary
            test_metrics: Test metrics dictionary
            output_path: Path to save plot
        """
        logger.info("\nPlotting confusion matrices...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Validation confusion matrix
        val_cm = np.array(val_metrics['confusion_matrix'])
        self.plot_confusion_matrix(
            val_cm,
            'Validation Set - Confusion Matrix',
            ax=axes[0]
        )

        # Test confusion matrix
        test_cm = np.array(test_metrics['confusion_matrix'])
        self.plot_confusion_matrix(
            test_cm,
            'Test Set - Confusion Matrix',
            ax=axes[1]
        )

        plt.tight_layout()

        # Save
        if output_path is None:
            output_path = OUTPUT_FILES['confusion_matrices']

        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        logger.info(f"  ✓ Saved to: {output_path}")
        plt.close()

    def plot_roc_curve(self, X_test: np.ndarray, y_test: np.ndarray,
                      model, output_path=None):
        """
        Plot ROC curve

        Args:
            X_test: Test features
            y_test: Test labels
            model: Trained model
            output_path: Path to save plot
        """
        logger.info("\nPlotting ROC curve...")

        from sklearn.metrics import roc_curve, roc_auc_score

        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Random Classifier')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve',
                    fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        if output_path is None:
            output_path = OUTPUT_FILES['roc_curve']

        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        logger.info(f"  ✓ Saved to: {output_path}")
        plt.close()

    def plot_feature_importance(self, feature_importance_df: pd.DataFrame,
                               top_n: int = 20, output_path=None):
        """
        Plot feature importance

        Args:
            feature_importance_df: DataFrame with feature importance
            top_n: Number of top features to plot
            output_path: Path to save plot
        """
        logger.info(f"\nPlotting top {top_n} feature importance...")

        # Get top N features
        top_features = feature_importance_df.head(top_n)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))

        bars = ax.barh(range(len(top_features)),
                      top_features['importance_normalized'],
                      color=colors)

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=10)
        ax.set_xlabel('Importance (%)', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features',
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_features['importance_normalized'])):
            ax.text(value + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{value:.2f}%',
                   va='center', fontsize=9)

        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        # Save
        if output_path is None:
            output_path = OUTPUT_FILES['feature_importance_plot']

        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        logger.info(f"  ✓ Saved to: {output_path}")
        plt.close()

    def plot_multiclass_map(self, multiclass_map: np.ndarray,
                           valid_mask: np.ndarray = None,
                           output_path=None):
        """
        Plot 4-class classification map

        Args:
            multiclass_map: Multiclass classification map (4 classes)
            valid_mask: Valid pixel mask
            output_path: Path to save plot
        """
        logger.info("\nPlotting 4-class multiclass map...")

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Create masked array for visualization
        if valid_mask is not None:
            class_masked = np.ma.masked_where(~valid_mask, multiclass_map)
        else:
            class_masked = np.ma.masked_where(multiclass_map == 255, multiclass_map)

        # Custom colormap for 4 classes
        # 0=Forest Stable (Green), 1=Deforestation (Red), 2=Non-forest (Gray), 3=Reforestation (Blue)
        cmap_multiclass = ListedColormap(['#2ecc71', '#e74c3c', '#95a5a6', '#3498db'])
        im = ax.imshow(class_masked, cmap=cmap_multiclass, vmin=0, vmax=3)

        ax.set_title('4-Class Classification Map\n(Green: Forest Stable | Red: Deforestation | Gray: Non-forest | Blue: Reforestation)',
                     fontsize=12, fontweight='bold')
        ax.axis('off')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                           pad=0.05, fraction=0.046)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(['Forest Stable', 'Deforestation', 'Non-forest', 'Reforestation'])

        plt.tight_layout()

        # Save
        if output_path is None:
            output_path = PLOTS_DIR / 'multiclass_map.png'

        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        logger.info(f"  ✓ Saved to: {output_path}")
        plt.close()

    def plot_classification_maps(self, classification_map: np.ndarray,
                                probability_map: np.ndarray,
                                valid_mask: np.ndarray = None,
                                multiclass_map: np.ndarray = None,
                                output_path=None):
        """
        Plot classification and probability maps (with optional multiclass)

        Args:
            classification_map: Binary classification map
            probability_map: Probability map
            valid_mask: Valid pixel mask
            multiclass_map: Optional 4-class map to display
            output_path: Path to save plot
        """
        logger.info("\nPlotting classification maps...")

        # If multiclass map is provided, create 3-panel figure
        if multiclass_map is not None:
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))

            # Panel 1: Multiclass (4 classes)
            ax1 = axes[0]
            if valid_mask is not None:
                mc_masked = np.ma.masked_where(~valid_mask, multiclass_map)
            else:
                mc_masked = np.ma.masked_where(multiclass_map == 255, multiclass_map)

            cmap_multiclass = ListedColormap(['#2ecc71', '#e74c3c', '#95a5a6', '#3498db'])
            im1 = ax1.imshow(mc_masked, cmap=cmap_multiclass, vmin=0, vmax=3)
            ax1.set_title('4-Class Map\n(Green: Forest Stable | Red: Deforestation | Gray: Non-forest | Blue: Reforestation)',
                         fontsize=11, fontweight='bold')
            ax1.axis('off')

            cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, fraction=0.046)
            cbar1.set_ticks([0, 1, 2, 3])
            cbar1.set_ticklabels(['Forest\nStable', 'Defor.', 'Non-\nforest', 'Refor.'], fontsize=9)

            # Panel 2: Binary classification
            ax2 = axes[1]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            ax2 = axes[0]

        # Classification map (Binary)
        # Create masked array for visualization
        if valid_mask is not None:
            class_masked = np.ma.masked_where(~valid_mask, classification_map)
        else:
            class_masked = np.ma.masked_where(classification_map == -1, classification_map)

        # Custom colormap
        cmap_class = ListedColormap(['#2ecc71', '#e74c3c'])  # Green, Red
        if multiclass_map is not None:
            im2 = ax2.imshow(class_masked, cmap=cmap_class, vmin=0, vmax=1)
        else:
            im1 = ax2.imshow(class_masked, cmap=cmap_class, vmin=0, vmax=1)

        ax2.set_title('Binary Classification Map\n(Green: No Deforestation, Red: Deforestation)',
                     fontsize=12, fontweight='bold')
        ax2.axis('off')

        # Add colorbar
        if multiclass_map is not None:
            cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal',
                                pad=0.05, fraction=0.046)
            cbar2.set_ticks([0, 1])
            cbar2.set_ticklabels(['No Deforestation', 'Deforestation'])
            # Panel 3: Probability map
            ax3 = axes[2]
        else:
            cbar1 = plt.colorbar(im1, ax=ax2, orientation='horizontal',
                                pad=0.05, fraction=0.046)
            cbar1.set_ticks([0, 1])
            cbar1.set_ticklabels(['No Deforestation', 'Deforestation'])
            # Probability map
            ax3 = axes[1]

        # Probability map

        # Create masked array
        if valid_mask is not None:
            prob_masked = np.ma.masked_where(~valid_mask, probability_map)
        else:
            prob_masked = np.ma.masked_where(probability_map == -9999, probability_map)

        # Plot with RdYlGn_r colormap
        if multiclass_map is not None:
            im3 = ax3.imshow(prob_masked, cmap=self.config['cmap_probability'],
                            vmin=0, vmax=1)
        else:
            im2 = ax3.imshow(prob_masked, cmap=self.config['cmap_probability'],
                            vmin=0, vmax=1)

        ax3.set_title('Deforestation Probability Map\n(Green: Low Probability, Red: High Probability)',
                     fontsize=12, fontweight='bold')
        ax3.axis('off')

        # Add colorbar
        if multiclass_map is not None:
            cbar3 = plt.colorbar(im3, ax=ax3, orientation='horizontal',
                                pad=0.05, fraction=0.046)
            cbar3.set_label('Probability of Deforestation', fontsize=10)
        else:
            cbar2 = plt.colorbar(im2, ax=ax3, orientation='horizontal',
                                pad=0.05, fraction=0.046)
            cbar2.set_label('Probability of Deforestation', fontsize=10)

        plt.tight_layout()

        # Save
        if output_path is None:
            output_path = OUTPUT_FILES['classification_maps']

        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        logger.info(f"  ✓ Saved to: {output_path}")
        plt.close()

    def plot_cv_scores(self, cv_scores: dict, output_path=None):
        """
        Plot cross-validation scores

        Args:
            cv_scores: Cross-validation scores dictionary
            output_path: Path to save plot
        """
        logger.info("\nPlotting cross-validation scores...")

        metrics = list(cv_scores.keys())
        means = [cv_scores[m]['mean'] for m in metrics]
        stds = [cv_scores[m]['std'] for m in metrics]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(metrics))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                     color=plt.cm.viridis(np.linspace(0.3, 0.9, len(metrics))),
                     alpha=0.8, edgecolor='black')

        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Cross-Validation Scores (Mean ± Std)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                   f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()

        # Save
        if output_path is None:
            output_path = OUTPUT_FILES['cv_scores']

        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        logger.info(f"  ✓ Saved to: {output_path}")
        plt.close()

    def create_all_visualizations(self, val_metrics: dict, test_metrics: dict,
                                 feature_importance_df: pd.DataFrame,
                                 cv_scores: dict,
                                 classification_map: np.ndarray,
                                 probability_map: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 model,
                                 valid_mask: np.ndarray = None):
        """
        Create all visualizations

        Args:
            val_metrics: Validation metrics
            test_metrics: Test metrics
            feature_importance_df: Feature importance DataFrame
            cv_scores: Cross-validation scores
            classification_map: Classification map
            probability_map: Probability map
            X_test: Test features
            y_test: Test labels
            model: Trained model
            valid_mask: Valid pixel mask
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 9: CREATE VISUALIZATIONS")
        logger.info("="*70)

        # Confusion matrices
        self.plot_confusion_matrices(val_metrics, test_metrics)

        # ROC curve
        self.plot_roc_curve(X_test, y_test, model)

        # Feature importance (show all features)
        self.plot_feature_importance(feature_importance_df, top_n=len(feature_importance_df))

        # Classification maps
        self.plot_classification_maps(classification_map, probability_map, valid_mask)

        # Cross-validation scores
        self.plot_cv_scores(cv_scores)

        logger.info("\n" + "="*70)
        logger.info("✓ ALL VISUALIZATIONS CREATED")
        logger.info("="*70)
        logger.info(f"\nPlots saved to: {PLOTS_DIR}")
        logger.info(f"  - Confusion matrices: {OUTPUT_FILES['confusion_matrices'].name}")
        logger.info(f"  - ROC curve: {OUTPUT_FILES['roc_curve'].name}")
        logger.info(f"  - Feature importance: {OUTPUT_FILES['feature_importance_plot'].name}")
        logger.info(f"  - Classification maps: {OUTPUT_FILES['classification_maps'].name}")
        logger.info(f"  - CV scores: {OUTPUT_FILES['cv_scores'].name}")
        logger.info("="*70)


def main():
    """Main function to test visualization"""
    logger.info("Testing Step 9: Visualization")

    import json
    import rasterio

    # Load evaluation metrics
    metrics_path = OUTPUT_FILES['evaluation_metrics']
    if not metrics_path.exists():
        logger.error("Evaluation metrics not found. Please run step 6 first.")
        return None

    logger.info(f"\nLoading evaluation metrics from: {metrics_path}")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    val_metrics = metrics['validation']
    test_metrics = metrics['test']
    cv_scores = metrics['cross_validation']

    # Load feature importance
    importance_path = OUTPUT_FILES['feature_importance']
    if not importance_path.exists():
        logger.error("Feature importance not found. Please run step 6 first.")
        return None

    logger.info(f"Loading feature importance from: {importance_path}")
    feature_importance_df = pd.read_csv(importance_path)

    # Load classification maps
    class_path = OUTPUT_FILES['classification_raster']
    prob_path = OUTPUT_FILES['probability_raster']

    if not class_path.exists() or not prob_path.exists():
        logger.error("Classification/probability rasters not found. Please run step 7 first.")
        return None

    logger.info(f"Loading rasters...")
    with rasterio.open(class_path) as src:
        classification_map = src.read(1)

    with rasterio.open(prob_path) as src:
        probability_map = src.read(1)

    # Load model
    from step5_train_random_forest import RandomForestTrainer
    trainer = RandomForestTrainer()
    model = trainer.load_model()

    # Load test data (we need to recreate this)
    logger.info("\nRecreating test data...")
    from step1_2_setup_and_load_data import DataLoader
    from core.feature_extraction import FeatureExtraction
    from step4_extract_training_data import TrainingDataExtractor
    from config import FEATURE_NAMES

    loader = DataLoader()
    data = loader.load_all()

    feature_extractor = FeatureExtraction()
    feature_stack, valid_mask = feature_extractor.extract_features(
        data['s2_before'],
        data['s2_after'],
        data['s1_before'],
        data['s1_after']
    )

    extractor = TrainingDataExtractor()
    training_df = extractor.extract_pixel_values(
        feature_stack,
        data['ground_truth'],
        data['metadata']['s2_before']['transform']
    )

    X = training_df[FEATURE_NAMES].values
    y = training_df['label'].values
    _, _, X_test, _, _, y_test = extractor.split_data(X, y)

    # Create visualizations
    visualizer = Visualizer()
    visualizer.create_all_visualizations(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        feature_importance_df=feature_importance_df,
        cv_scores=cv_scores,
        classification_map=classification_map,
        probability_map=probability_map,
        X_test=X_test,
        y_test=y_test,
        model=model,
        valid_mask=valid_mask
    )

    return visualizer


if __name__ == "__main__":
    visualizer = main()
