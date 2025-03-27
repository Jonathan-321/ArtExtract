"""
Classification metrics module for the ArtExtract project.
This module implements various metrics for evaluating art style classification models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """
    Evaluator for classification models.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize the evaluator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        logger.info(f"Initialized ClassificationEvaluator with {self.num_classes} classes")
    
    def compute_metrics(self, 
                       y_true: np.ndarray, 
                       y_pred: np.ndarray, 
                       y_score: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_score: Predicted probabilities (optional, for ROC and PR curves)
            
        Returns:
            Dictionary with metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle multi-class metrics
        if self.num_classes > 2:
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            # Class-specific metrics
            class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
            class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
            class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            # Create class-specific metrics dictionary
            class_metrics = {}
            for i, class_name in enumerate(self.class_names):
                if i < len(class_precision):  # Ensure index is valid
                    class_metrics[class_name] = {
                        'precision': float(class_precision[i]),
                        'recall': float(class_recall[i]),
                        'f1': float(class_f1[i])
                    }
        else:
            # Binary classification
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            # No need for class-specific metrics in binary case
            class_metrics = {
                self.class_names[1]: {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1)
                }
            }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Prepare result
        result = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist(),
            'class_metrics': class_metrics
        }
        
        # Add ROC and PR curve metrics if probabilities are provided
        if y_score is not None:
            if self.num_classes == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
                roc_auc = auc(fpr, tpr)
                
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score[:, 1])
                pr_auc = average_precision_score(y_true, y_score[:, 1])
                
                result['roc_auc'] = float(roc_auc)
                result['pr_auc'] = float(pr_auc)
                result['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
                result['pr_curve'] = {
                    'precision': precision_curve.tolist(),
                    'recall': recall_curve.tolist()
                }
            else:
                # Multi-class: one-vs-rest ROC
                roc_auc = {}
                for i in range(self.num_classes):
                    if i < len(self.class_names):  # Ensure index is valid
                        class_name = self.class_names[i]
                        y_true_binary = (y_true == i).astype(int)
                        if i < y_score.shape[1]:  # Ensure index is valid
                            fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i])
                            roc_auc[class_name] = float(auc(fpr, tpr))
                
                result['roc_auc'] = roc_auc
        
        return result
    
    def plot_confusion_matrix(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             normalize: bool = False,
                             figsize: Tuple[int, int] = (10, 8),
                             save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size
            save_path: Optional path to save the plot
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Create plot
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, 
                      y_true: np.ndarray, 
                      y_score: np.ndarray,
                      figsize: Tuple[int, int] = (10, 8),
                      save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: Ground truth labels
            y_score: Predicted probabilities
            figsize: Figure size
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=figsize)
        
        if self.num_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            # Multi-class: one-vs-rest ROC
            for i in range(self.num_classes):
                if i < len(self.class_names) and i < y_score.shape[1]:  # Ensure indices are valid
                    class_name = self.class_names[i]
                    y_true_binary = (y_true == i).astype(int)
                    fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        # Add diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"ROC curve plot saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, 
                                  y_true: np.ndarray, 
                                  y_score: np.ndarray,
                                  figsize: Tuple[int, int] = (10, 8),
                                  save_path: Optional[str] = None) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: Ground truth labels
            y_score: Predicted probabilities
            figsize: Figure size
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=figsize)
        
        if self.num_classes == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_score[:, 1])
            pr_auc = average_precision_score(y_true, y_score[:, 1])
            
            plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {pr_auc:.2f})')
        else:
            # Multi-class: one-vs-rest PR
            for i in range(self.num_classes):
                if i < len(self.class_names) and i < y_score.shape[1]:  # Ensure indices are valid
                    class_name = self.class_names[i]
                    y_true_binary = (y_true == i).astype(int)
                    precision, recall, _ = precision_recall_curve(y_true_binary, y_score[:, i])
                    pr_auc = average_precision_score(y_true_binary, y_score[:, i])
                    
                    plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {pr_auc:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Precision-recall curve plot saved to {save_path}")
        
        plt.show()
    
    def plot_class_distribution(self, 
                              y_true: np.ndarray, 
                              figsize: Tuple[int, int] = (12, 6),
                              save_path: Optional[str] = None) -> None:
        """
        Plot class distribution.
        
        Args:
            y_true: Ground truth labels
            figsize: Figure size
            save_path: Optional path to save the plot
        """
        # Count samples per class
        class_counts = np.bincount(y_true, minlength=self.num_classes)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Class': self.class_names[:len(class_counts)],
            'Count': class_counts
        })
        
        # Sort by count
        df = df.sort_values('Count', ascending=False)
        
        # Create plot
        plt.figure(figsize=figsize)
        ax = sns.barplot(x='Class', y='Count', data=df)
        
        # Add count labels
        for i, p in enumerate(ax.patches):
            ax.annotate(
                f'{p.get_height():,}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='bottom',
                fontsize=10,
                rotation=0
            )
        
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Class distribution plot saved to {save_path}")
        
        plt.show()
    
    def generate_classification_report(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     output_path: Optional[str] = None) -> str:
        """
        Generate a classification report.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            output_path: Optional path to save the report
            
        Returns:
            Classification report as string
        """
        # Generate report
        report = classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0)
        
        # Save to file if requested
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Classification report saved to {output_path}")
        
        return report
    
    def save_metrics_to_json(self, metrics: Dict[str, Any], output_path: str) -> None:
        """
        Save metrics to a JSON file.
        
        Args:
            metrics: Dictionary with metrics
            output_path: Path to save the metrics
        """
        import json
        
        # Convert numpy types to Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_metrics = convert_to_serializable(metrics)
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        
        logger.info(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Classification Metrics module")
    print("Use this module to evaluate art style classification models.")
