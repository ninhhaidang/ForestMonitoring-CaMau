"""
Evaluation and comparison script for Phase 1 models (Random Forest vs Simple CNN)
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from .config import *


def load_results(model_type):
    """
    Load results from JSON file.

    Args:
        model_type: 'rf' or 'cnn'

    Returns:
        results: Dictionary containing metrics and predictions
    """
    if model_type == 'rf':
        results_path = RESULTS_DIR / "random_forest_results.json"
    elif model_type == 'cnn':
        results_path = RESULTS_DIR / "simple_cnn_results.json"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, 'r') as f:
        results = json.load(f)

    return results


def compare_models(save_report=True):
    """
    Compare Random Forest and Simple CNN models.

    Args:
        save_report: Whether to save comparison report

    Returns:
        comparison: Dictionary with comparison results
    """
    print("=" * 70)
    print("PHASE 1 MODEL COMPARISON: Random Forest vs Simple CNN")
    print("=" * 70)

    # Load results
    print("\nLoading results...")
    try:
        rf_results = load_results('rf')
        print("✓ Random Forest results loaded")
    except FileNotFoundError as e:
        print(f"✗ Random Forest results not found: {e}")
        rf_results = None

    try:
        cnn_results = load_results('cnn')
        print("✓ Simple CNN results loaded")
    except FileNotFoundError as e:
        print(f"✗ Simple CNN results not found: {e}")
        cnn_results = None

    if rf_results is None and cnn_results is None:
        print("\nError: No model results found. Please train models first.")
        return None

    # ============================================================
    # 1. Create comparison table
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST SET PERFORMANCE COMPARISON")
    print("=" * 70)

    comparison = {}

    # Extract test metrics
    if rf_results:
        rf_test = rf_results['metrics']['test']
        comparison['rf'] = {
            'accuracy': rf_test['accuracy'],
            'precision': rf_test['precision'],
            'recall': rf_test['recall'],
            'f1': rf_test['f1'],
            'auc': rf_test['auc'],
            'training_time': rf_results['training_time'],
            'model_size': 'N/A (sklearn)',
            'inference_speed': 'Fast (CPU)'
        }

    if cnn_results:
        cnn_test = cnn_results['metrics']['test']
        comparison['cnn'] = {
            'accuracy': cnn_test['accuracy'],
            'precision': cnn_test['precision'],
            'recall': cnn_test['recall'],
            'f1': cnn_test['f1'],
            'auc': cnn_test['auc'],
            'training_time': cnn_results['training_time'],
            'model_size': f"{cnn_results['model_params']['total'] / 1e6:.2f}M params",
            'inference_speed': 'Fast (GPU/CPU)'
        }

    # Print comparison table
    print(f"\n{'Metric':<20} {'Random Forest':<20} {'Simple CNN':<20} {'Winner':<15}")
    print("-" * 75)

    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    for metric in metrics_to_compare:
        rf_val = comparison.get('rf', {}).get(metric, 0)
        cnn_val = comparison.get('cnn', {}).get(metric, 0)

        if rf_results and cnn_results:
            winner = 'Random Forest' if rf_val > cnn_val else 'Simple CNN' if cnn_val > rf_val else 'Tie'
            print(f"{metric.capitalize():<20} {rf_val:<20.4f} {cnn_val:<20.4f} {winner:<15}")
        elif rf_results:
            print(f"{metric.capitalize():<20} {rf_val:<20.4f} {'N/A':<20} {'RF (only)':<15}")
        elif cnn_results:
            print(f"{metric.capitalize():<20} {'N/A':<20} {cnn_val:<20.4f} {'CNN (only)':<15}")

    # Training time
    print("-" * 75)
    if rf_results and cnn_results:
        rf_time = rf_results['training_time']
        cnn_time = cnn_results['training_time']
        time_winner = 'Random Forest' if rf_time < cnn_time else 'Simple CNN'
        print(f"{'Training Time':<20} {rf_time:<20.2f}s {cnn_time:<20.2f}s {time_winner:<15}")
    elif rf_results:
        print(f"{'Training Time':<20} {rf_results['training_time']:<20.2f}s {'N/A':<20} {'RF (only)':<15}")
    elif cnn_results:
        print(f"{'Training Time':<20} {'N/A':<20} {cnn_results['training_time']:<20.2f}s {'CNN (only)':<15}")

    # ============================================================
    # 2. Determine winner
    # ============================================================
    print("\n" + "=" * 70)
    print("OVERALL WINNER DETERMINATION")
    print("=" * 70)

    if rf_results and cnn_results:
        # Compare F1 scores (primary metric for binary classification)
        rf_f1 = comparison['rf']['f1']
        cnn_f1 = comparison['cnn']['f1']
        f1_diff = abs(rf_f1 - cnn_f1)

        print(f"\nPrimary Metric: F1-Score")
        print(f"  - Random Forest: {rf_f1:.4f}")
        print(f"  - Simple CNN:    {cnn_f1:.4f}")
        print(f"  - Difference:    {f1_diff:.4f}")

        if f1_diff < 0.02:  # Less than 2% difference
            print("\n✓ RESULT: Models perform SIMILARLY (difference < 2%)")
            print("  → Recommendation: Use Random Forest (faster, simpler)")
            winner = "Tie (RF recommended)"
        elif rf_f1 > cnn_f1:
            print(f"\n✓ RESULT: Random Forest WINS by {(rf_f1 - cnn_f1) * 100:.2f}%")
            print("  → Recommendation: Use Random Forest")
            winner = "Random Forest"
        else:
            print(f"\n✓ RESULT: Simple CNN WINS by {(cnn_f1 - rf_f1) * 100:.2f}%")
            print("  → Recommendation: Proceed to Phase 2 with more advanced CNNs")
            winner = "Simple CNN"

        comparison['winner'] = winner
        comparison['f1_difference'] = f1_diff

    # ============================================================
    # 3. Create visualizations
    # ============================================================
    print("\n" + "=" * 70)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 10))

    # 1. Metrics comparison bar chart
    ax1 = plt.subplot(2, 3, 1)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    x = np.arange(len(metrics_names))
    width = 0.35

    if rf_results and cnn_results:
        rf_values = [comparison['rf'][m.lower()] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']]
        cnn_values = [comparison['cnn'][m.lower()] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']]

        plt.bar(x - width/2, rf_values, width, label='Random Forest', alpha=0.8, color='#2ecc71')
        plt.bar(x + width/2, cnn_values, width, label='Simple CNN', alpha=0.8, color='#3498db')

        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Test Set Metrics Comparison')
        plt.xticks(x, metrics_names)
        plt.ylim([0, 1.05])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

    # 2. Confusion matrices side-by-side
    if rf_results:
        ax2 = plt.subplot(2, 3, 2)
        cm_rf = np.array(rf_results['metrics']['test']['confusion_matrix'])
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title('Random Forest - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

    if cnn_results:
        ax3 = plt.subplot(2, 3, 3)
        cm_cnn = np.array(cnn_results['metrics']['test']['confusion_matrix'])
        sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title('Simple CNN - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

    # 3. ROC curves comparison
    ax4 = plt.subplot(2, 3, 4)
    from sklearn.metrics import roc_curve

    if rf_results:
        rf_pred = rf_results['predictions']['test']
        fpr_rf, tpr_rf, _ = roc_curve(rf_pred['y_true'], rf_pred['y_proba'])
        plt.plot(fpr_rf, tpr_rf, label=f'RF (AUC={comparison["rf"]["auc"]:.3f})', linewidth=2, color='#2ecc71')

    if cnn_results:
        cnn_pred = cnn_results['predictions']['test']
        fpr_cnn, tpr_cnn, _ = roc_curve(cnn_pred['y_true'], cnn_pred['y_proba'])
        plt.plot(fpr_cnn, tpr_cnn, label=f'CNN (AUC={comparison["cnn"]["auc"]:.3f})', linewidth=2, color='#3498db')

    plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(alpha=0.3)

    # 4. Training time comparison
    ax5 = plt.subplot(2, 3, 5)
    models = []
    times = []

    if rf_results:
        models.append('Random\nForest')
        times.append(rf_results['training_time'] / 60)  # Convert to minutes

    if cnn_results:
        models.append('Simple\nCNN')
        times.append(cnn_results['training_time'] / 60)

    colors = ['#2ecc71' if 'Forest' in m else '#3498db' for m in models]
    plt.bar(models, times, color=colors, alpha=0.8)
    plt.ylabel('Training Time (minutes)')
    plt.title('Training Time Comparison')
    plt.grid(axis='y', alpha=0.3)

    # 5. F1-Score breakdown
    ax6 = plt.subplot(2, 3, 6)
    splits = ['Train', 'Val', 'Test']

    if rf_results and cnn_results:
        rf_f1_scores = [
            rf_results['metrics']['train']['f1'],
            rf_results['metrics']['val']['f1'],
            rf_results['metrics']['test']['f1']
        ]
        cnn_f1_scores = [
            cnn_results['metrics']['train']['f1'],
            cnn_results['metrics']['val']['f1'],
            cnn_results['metrics']['test']['f1']
        ]

        x = np.arange(len(splits))
        width = 0.35
        plt.bar(x - width/2, rf_f1_scores, width, label='Random Forest', alpha=0.8, color='#2ecc71')
        plt.bar(x + width/2, cnn_f1_scores, width, label='Simple CNN', alpha=0.8, color='#3498db')

        plt.xlabel('Data Split')
        plt.ylabel('F1-Score')
        plt.title('F1-Score Across Data Splits')
        plt.xticks(x, splits)
        plt.ylim([0, 1.05])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "phase1_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison visualization saved to: {fig_path}")
    plt.close()

    # ============================================================
    # 4. Save comparison report
    # ============================================================
    if save_report:
        report_path = RESULTS_DIR / "phase1_comparison.json"
        comparison['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(report_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"✓ Comparison report saved to: {report_path}")

        # Also save markdown report
        md_report_path = RESULTS_DIR / "phase1_comparison.md"
        with open(md_report_path, 'w') as f:
            f.write("# Phase 1 Model Comparison Report\n\n")
            f.write(f"**Generated:** {comparison['timestamp']}\n\n")
            f.write("## Test Set Performance\n\n")
            f.write("| Metric | Random Forest | Simple CNN | Winner |\n")
            f.write("|--------|---------------|------------|--------|\n")

            for metric in metrics_to_compare:
                rf_val = comparison.get('rf', {}).get(metric, 'N/A')
                cnn_val = comparison.get('cnn', {}).get(metric, 'N/A')

                rf_str = f"{rf_val:.4f}" if isinstance(rf_val, float) else rf_val
                cnn_str = f"{cnn_val:.4f}" if isinstance(cnn_val, float) else cnn_val

                if isinstance(rf_val, float) and isinstance(cnn_val, float):
                    winner = 'RF' if rf_val > cnn_val else 'CNN' if cnn_val > rf_val else 'Tie'
                else:
                    winner = '-'

                f.write(f"| {metric.capitalize()} | {rf_str} | {cnn_str} | {winner} |\n")

            if rf_results and cnn_results:
                f.write(f"\n## Overall Winner\n\n")
                f.write(f"**{comparison['winner']}**\n\n")
                f.write(f"F1-Score difference: {comparison['f1_difference']:.4f}\n\n")

        print(f"✓ Markdown report saved to: {md_report_path}")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED!")
    print("=" * 70)

    return comparison


if __name__ == "__main__":
    comparison = compare_models()
