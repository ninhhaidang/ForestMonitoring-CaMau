"""
Evaluation Utilities
Các hàm tính metrics và visualize kết quả
"""

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true, y_pred):
    """Calculate all metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary')
    }
    
    # IoU (Intersection over Union)
    cm = confusion_matrix(y_true, y_pred)
    intersection = np.diag(cm)
    ground_truth = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    union = ground_truth + predicted - intersection
    iou = intersection / union
    metrics['iou'] = iou[1]  # IoU for change class
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Change', 'Change'],
                yticklabels=['No Change', 'Change'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
