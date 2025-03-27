"""
Utility functions for models in the ArtExtract project.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_epoch(model: nn.Module, 
               dataloader: DataLoader, 
               criterion: nn.Module, 
               optimizer: torch.optim.Optimizer, 
               device: torch.device) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run the model on
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / total,
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc
    }


def validate(model: nn.Module, 
            dataloader: DataLoader, 
            criterion: nn.Module, 
            device: torch.device) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run the model on
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Store targets and predictions for metrics
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'targets': np.array(all_targets),
        'predictions': np.array(all_predictions)
    }


def save_model(model: nn.Module, 
              optimizer: torch.optim.Optimizer, 
              epoch: int, 
              metrics: Dict[str, float], 
              save_path: str) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary with metrics
        save_path: Path to save the checkpoint
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")


def load_model(model: nn.Module, 
              optimizer: Optional[torch.optim.Optimizer], 
              load_path: str, 
              device: torch.device) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int, Dict[str, float]]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (can be None if not needed)
        load_path: Path to load the checkpoint from
        device: Device to load the model to
        
    Returns:
        Tuple of (model, optimizer, epoch, metrics)
    """
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    
    logger.info(f"Model loaded from {load_path} (epoch {epoch})")
    
    return model, optimizer, epoch, metrics


def plot_confusion_matrix(targets: np.ndarray, 
                         predictions: np.ndarray, 
                         class_names: List[str], 
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the plot
    """
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def get_classification_report(targets: np.ndarray, 
                             predictions: np.ndarray, 
                             class_names: List[str]) -> str:
    """
    Get classification report.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted labels
        class_names: List of class names
        
    Returns:
        Classification report as string
    """
    return classification_report(targets, predictions, target_names=class_names)


def extract_features(model: nn.Module, 
                    dataloader: DataLoader, 
                    device: torch.device, 
                    layer_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from a specific layer of the model.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to run the model on
        layer_name: Name of the layer to extract features from (if None, use the output)
        
    Returns:
        Tuple of (features, labels)
    """
    model.eval()
    features = []
    labels = []
    
    # Register hook to get intermediate layer outputs if layer_name is provided
    if layer_name is not None:
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # Find the layer and register the hook
        for name, module in model.named_modules():
            if name == layer_name:
                module.register_forward_hook(get_activation(layer_name))
                break
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Extracting features"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get features
            if layer_name is not None:
                batch_features = activation[layer_name].cpu().numpy()
            else:
                batch_features = outputs.cpu().numpy()
            
            # Reshape if needed
            if len(batch_features.shape) > 2:
                batch_features = batch_features.reshape(batch_features.shape[0], -1)
            
            features.append(batch_features)
            labels.append(targets.cpu().numpy())
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    return features, labels


if __name__ == "__main__":
    # Example usage
    print("Model Utilities module")
    print("Use this module for training, evaluation, and visualization of models in the ArtExtract project.")
