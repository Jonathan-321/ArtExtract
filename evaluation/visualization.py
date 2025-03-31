"""
Visualization utilities for model performance.
This module provides functions to visualize training progress, confusion matrices,
and other performance metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None) -> None:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    if 'train_acc' in history:
        plt.plot(history['train_acc'], label='Training Accuracy')
    if 'val_acc' in history:
        plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         class_names: List[str],
                         normalize: bool = True,
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        save_path: Optional path to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def plot_feature_space(features: np.ndarray, 
                      labels: np.ndarray, 
                      class_names: List[str],
                      method: str = 'tsne',
                      save_path: Optional[str] = None) -> None:
    """
    Visualize feature space using dimensionality reduction.
    
    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Label array of shape (n_samples,)
        class_names: List of class names
        method: Dimensionality reduction method ('tsne' or 'pca')
        save_path: Optional path to save the plot
    """
    # Reduce dimensionality
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(features)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(features)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot each class
    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(
            reduced_features[mask, 0],
            reduced_features[mask, 1],
            alpha=0.7,
            label=class_name
        )
    
    plt.title(f"Feature Space Visualization ({method.upper()})")
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Feature space visualization saved to {save_path}")
    
    plt.show()


def plot_class_distribution(labels: np.ndarray, 
                           class_names: List[str],
                           save_path: Optional[str] = None) -> None:
    """
    Plot class distribution.
    
    Args:
        labels: Label array of shape (n_samples,)
        class_names: List of class names
        save_path: Optional path to save the plot
    """
    # Count samples per class
    class_counts = np.bincount(labels)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Class': class_names[:len(class_counts)],
        'Count': class_counts
    })
    
    # Sort by count
    df = df.sort_values('Count', ascending=False)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='Count', data=df)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Class distribution plot saved to {save_path}")
    
    plt.show()


def plot_similarity_heatmap(similarity_matrix: np.ndarray,
                           labels: Optional[List[str]] = None,
                           save_path: Optional[str] = None) -> None:
    """
    Plot similarity heatmap.
    
    Args:
        similarity_matrix: Similarity matrix of shape (n_samples, n_samples)
        labels: Optional list of labels for the samples
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=False, cmap='viridis',
                xticklabels=labels, yticklabels=labels)
    plt.title('Similarity Heatmap')
    plt.xlabel('Samples')
    plt.ylabel('Samples')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Similarity heatmap saved to {save_path}")
    
    plt.show()


def visualize_model_predictions(model: torch.nn.Module,
                               images: torch.Tensor,
                               labels: torch.Tensor,
                               class_names: List[str],
                               device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                               num_samples: int = 5,
                               save_path: Optional[str] = None) -> None:
    """
    Visualize model predictions on sample images.
    
    Args:
        model: PyTorch model
        images: Batch of images
        labels: True labels
        class_names: List of class names
        device: Device to run the model on
        num_samples: Number of samples to visualize
        save_path: Optional path to save the plot
    """
    # Set model to evaluation mode
    model.eval()
    
    # Move model to device
    model = model.to(device)
    
    # Get predictions
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Convert to numpy
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    
    # Limit number of samples
    num_samples = min(num_samples, len(images))
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    # Plot each sample
    for i in range(num_samples):
        # Get image
        img = images[i].transpose(1, 2, 0)
        
        # Denormalize if needed
        img = np.clip(img, 0, 1)
        
        # Get true and predicted labels
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        
        # Plot image
        axes[i].imshow(img)
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Model predictions visualization saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Visualization module")
    print("Use this module to visualize model performance metrics.")
