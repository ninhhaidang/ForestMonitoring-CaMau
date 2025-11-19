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
    VIZ_CONFIG, PLOTS_DIR,
    OUTPUT_FILES, LOG_CONFIG
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

    def plot_5fold_results(self, fold_results: list, output_path=None):
        """
        Plot 5-fold cross-validation results for CNN

        Args:
            fold_results: List of dictionaries with fold results
                Each dict should contain: {
                    'fold': int,
                    'train_loss': float,
                    'val_loss': float,
                    'train_acc': float,
                    'val_acc': float,
                    'test_acc': float,
                    'test_metrics': dict (optional)
                }
            output_path: Path to save plot
        """
        logger.info("\nPlotting 5-fold cross-validation results...")

        n_folds = len(fold_results)

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Training & Validation Accuracy per Fold
        ax1 = fig.add_subplot(gs[0, :2])
        folds = [r['fold'] for r in fold_results]
        train_accs = [r['train_acc'] for r in fold_results]
        val_accs = [r['val_acc'] for r in fold_results]
        test_accs = [r['test_acc'] for r in fold_results]

        x = np.arange(n_folds)
        width = 0.25

        bars1 = ax1.bar(x - width, train_accs, width, label='Train',
                       color='#3498db', alpha=0.8)
        bars2 = ax1.bar(x, val_accs, width, label='Validation',
                       color='#2ecc71', alpha=0.8)
        bars3 = ax1.bar(x + width, test_accs, width, label='Test',
                       color='#e74c3c', alpha=0.8)

        ax1.set_xlabel('Fold', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title(f'{n_folds}-Fold Cross-Validation: Accuracy per Fold',
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
        ax1.legend(fontsize=11)
        ax1.set_ylim([0, 1.05])
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)

        # 2. Loss per Fold
        ax2 = fig.add_subplot(gs[0, 2])
        train_losses = [r['train_loss'] for r in fold_results]
        val_losses = [r['val_loss'] for r in fold_results]

        bars1 = ax2.bar(x - width/2, train_losses, width, label='Train',
                       color='#3498db', alpha=0.8)
        bars2 = ax2.bar(x + width/2, val_losses, width, label='Validation',
                       color='#2ecc71', alpha=0.8)

        ax2.set_xlabel('Fold', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax2.set_title('Loss per Fold', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'F{i+1}' for i in range(n_folds)])
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)

        # 3. Statistics Summary (Box plots)
        ax3 = fig.add_subplot(gs[1, :])

        data_to_plot = [train_accs, val_accs, test_accs]
        labels = ['Train Accuracy', 'Val Accuracy', 'Test Accuracy']

        bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(color='green', linewidth=2, linestyle='--'))

        ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax3.set_title('Accuracy Distribution Across Folds',
                     fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim([0, 1.05])

        # Add statistics text
        train_mean = np.mean(train_accs)
        train_std = np.std(train_accs)
        val_mean = np.mean(val_accs)
        val_std = np.std(val_accs)
        test_mean = np.mean(test_accs)
        test_std = np.std(test_accs)

        stats_text = (
            f'Train: {train_mean:.4f} ± {train_std:.4f}\n'
            f'Val: {val_mean:.4f} ± {val_std:.4f}\n'
            f'Test: {test_mean:.4f} ± {test_std:.4f}'
        )
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 4. Metrics per Fold (if available)
        if 'test_metrics' in fold_results[0]:
            ax4 = fig.add_subplot(gs[2, 0])

            precision_scores = [r['test_metrics']['precision'] for r in fold_results]
            recall_scores = [r['test_metrics']['recall'] for r in fold_results]
            f1_scores = [r['test_metrics']['f1'] for r in fold_results]

            width = 0.25
            bars1 = ax4.bar(x - width, precision_scores, width,
                           label='Precision', color='#9b59b6', alpha=0.8)
            bars2 = ax4.bar(x, recall_scores, width,
                           label='Recall', color='#e67e22', alpha=0.8)
            bars3 = ax4.bar(x + width, f1_scores, width,
                           label='F1-Score', color='#1abc9c', alpha=0.8)

            ax4.set_xlabel('Fold', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
            ax4.set_title('Classification Metrics per Fold',
                         fontsize=12, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels([f'F{i+1}' for i in range(n_folds)])
            ax4.legend(fontsize=10)
            ax4.set_ylim([0, 1.05])
            ax4.grid(axis='y', alpha=0.3)

        # 5. Final Summary Table
        ax5 = fig.add_subplot(gs[2, 1:])
        ax5.axis('off')

        # Create summary table
        summary_data = []
        for i, result in enumerate(fold_results):
            row = [
                f"Fold {i+1}",
                f"{result['train_acc']:.4f}",
                f"{result['val_acc']:.4f}",
                f"{result['test_acc']:.4f}",
            ]
            if 'test_metrics' in result:
                row.extend([
                    f"{result['test_metrics']['precision']:.4f}",
                    f"{result['test_metrics']['recall']:.4f}",
                    f"{result['test_metrics']['f1']:.4f}"
                ])
            summary_data.append(row)

        # Add mean row
        mean_row = [
            "Mean ± Std",
            f"{train_mean:.4f} ± {train_std:.4f}",
            f"{val_mean:.4f} ± {val_std:.4f}",
            f"{test_mean:.4f} ± {test_std:.4f}"
        ]
        if 'test_metrics' in fold_results[0]:
            mean_row.extend([
                f"{np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}",
                f"{np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}",
                f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}"
            ])
        summary_data.append(mean_row)

        columns = ['Fold', 'Train Acc', 'Val Acc', 'Test Acc']
        if 'test_metrics' in fold_results[0]:
            columns.extend(['Precision', 'Recall', 'F1-Score'])

        table = ax5.table(cellText=summary_data, colLabels=columns,
                         loc='center', cellLoc='center',
                         colWidths=[0.15] * len(columns))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style mean row
        for i in range(len(columns)):
            table[(len(summary_data), i)].set_facecolor('#f39c12')
            table[(len(summary_data), i)].set_text_props(weight='bold')

        ax5.set_title('Summary Table', fontsize=12, fontweight='bold', pad=20)

        plt.suptitle(f'{n_folds}-Fold Cross-Validation Results',
                    fontsize=16, fontweight='bold', y=0.995)

        # Save
        if output_path is None:
            output_path = PLOTS_DIR / 'cnn_5fold_results.png'

        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        logger.info(f"  ✓ Saved to: {output_path}")
        plt.close()

        return output_path

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
        logger.info("="*70)
