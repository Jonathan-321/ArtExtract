"""
Training script for the CNN-RNN model with hyperparameter tuning.
This script implements a more comprehensive training pipeline with hyperparameter tuning
for the art style classification model.
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocessing.data_preprocessing import ArtDatasetPreprocessor, create_data_loaders
from models.style_classification.cnn_rnn_model import create_model
from evaluation.classification_metrics import ClassificationEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_wikiart_dataset(data_dir: str) -> pd.DataFrame:
    """
    Load WikiArt dataset metadata into a DataFrame.
    
    Args:
        data_dir: Directory containing the WikiArt dataset
        
    Returns:
        DataFrame with image paths and labels
    """
    # Create a list to store image information
    data = []
    
    # Get all style directories
    style_dir = Path(data_dir) / 'style'
    if not style_dir.exists():
        logger.error(f"Style directory {style_dir} does not exist")
        return pd.DataFrame()
    
    styles = [d.name for d in style_dir.iterdir() if d.is_dir()]
    
    # Create style to index mapping
    style_to_idx = {style: idx for idx, style in enumerate(sorted(styles))}
    
    # Iterate through each style directory
    for style in styles:
        style_path = style_dir / style
        style_idx = style_to_idx[style]
        
        # Get all image files in this style directory
        image_files = list(style_path.glob('*.jpg')) + list(style_path.glob('*.png'))
        
        # Add each image to the data list
        for img_path in image_files:
            data.append({
                'image_path': str(img_path),
                'style': style,
                'label': style_idx
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    logger.info(f"Loaded {len(df)} images from WikiArt dataset")
    logger.info(f"Styles: {sorted(styles)}")
    
    return df

def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: str) -> Tuple[float, float]:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        total_loss += loss.item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def validate(model: nn.Module,
            dataloader: DataLoader,
            criterion: nn.Module,
            device: str) -> Tuple[float, float, List[int], List[int]]:
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_loss += loss.item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, all_preds, all_labels

def train_model(model: nn.Module,
               dataloaders: Dict[str, DataLoader],
               criterion: nn.Module,
               optimizer: optim.Optimizer,
               scheduler: Any,
               device: str,
               num_epochs: int,
               patience: int,
               output_dir: str) -> Dict[str, Any]:
    """
    Train and validate the model.
    
    Args:
        model: Model to train
        dataloaders: Dictionary of dataloaders for train, val, test
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Number of epochs to train for
        patience: Patience for early stopping
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with training results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize variables
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _ = validate(
            model, dataloaders['val'], criterion, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save model checkpoint
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            logger.info("Early stopping triggered")
            break
    
    # Load best model for evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    
    # Evaluate on test set
    test_loss, test_acc, test_preds, test_labels = validate(
        model, dataloaders['test'], criterion, device
    )
    
    # Create evaluator
    evaluator = ClassificationEvaluator(
        num_classes=len(np.unique(test_labels)),
        class_names=[str(i) for i in range(len(np.unique(test_labels)))]
    )
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(test_labels, test_preds)
    
    # Create confusion matrix
    evaluator.plot_confusion_matrix(
        test_labels, test_preds, 
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Log results
    logger.info("\nTest Results:")
    logger.info(f"Loss: {test_loss:.4f}")
    logger.info(f"Accuracy: {test_acc:.2f}%")
    logger.info(f"F1 Score: {metrics['f1_score_weighted']:.4f}")
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_f1_score': metrics['f1_score_weighted'],
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }, f, indent=4)
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_f1_score': metrics['f1_score_weighted'],
        'best_val_loss': best_val_loss,
        'epochs_trained': len(train_losses)
    }

def run_hyperparameter_tuning(data_dir: str, output_dir: str, param_grid: Dict[str, List[Any]]) -> None:
    """
    Run hyperparameter tuning for the CNN-RNN model.
    
    Args:
        data_dir: Directory containing the dataset
        output_dir: Directory to save outputs
        param_grid: Dictionary of hyperparameters to tune
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    df = load_wikiart_dataset(data_dir)
    if len(df) == 0:
        logger.error("Failed to load dataset")
        return
    
    # Get number of classes
    num_classes = len(df['style'].unique())
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Generate parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    logger.info(f"Running {len(param_combinations)} hyperparameter combinations")
    
    # Store results
    results = []
    
    # Run hyperparameter tuning
    for i, params in enumerate(param_combinations):
        logger.info(f"\nRunning combination {i+1}/{len(param_combinations)}")
        logger.info(f"Parameters: {params}")
        
        # Create experiment directory
        experiment_dir = os.path.join(output_dir, f"experiment_{i+1}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save parameters
        with open(os.path.join(experiment_dir, 'params.json'), 'w') as f:
            json.dump(params, f, indent=4)
        
        # Create preprocessor
        preprocessor = ArtDatasetPreprocessor(
            img_size=(224, 224),
            use_augmentation=params['use_augmentation'],
            normalize=True
        )
        
        # Create data loaders
        dataloaders = create_data_loaders(
            df,
            preprocessor,
            batch_size=params['batch_size'],
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            num_workers=4,
            target_column='style'
        )
        
        # Create model
        if params['model_type'] == 'cnn_rnn':
            model = create_model(
                num_classes=num_classes,
                cnn_backbone=params['cnn_backbone'],
                rnn_type=params['rnn_type'],
                rnn_hidden_size=params['rnn_hidden_size'],
                rnn_num_layers=params['rnn_num_layers'],
                dropout=params['dropout']
            )
        else:
            # Use a simple CNN model
            if params['cnn_backbone'] == 'resnet18':
                model = torchvision.models.resnet18(pretrained=True)
            elif params['cnn_backbone'] == 'resnet50':
                model = torchvision.models.resnet50(pretrained=True)
            else:
                raise ValueError(f"Unknown CNN backbone: {params['cnn_backbone']}")
            
            # Replace the last fully connected layer
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(params['dropout']),
                nn.Linear(num_features, num_classes)
            )
        
        model = model.to(device)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if params['optimizer'] == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        elif params['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=params['learning_rate'],
                momentum=0.9,
                weight_decay=params['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {params['optimizer']}")
        
        # Learning rate scheduler
        if params['scheduler'] == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=3,
                verbose=True
            )
        elif params['scheduler'] == 'cosine_annealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=params['num_epochs'],
                eta_min=1e-6
            )
        else:
            scheduler = None
        
        # Train model
        start_time = time.time()
        experiment_results = train_model(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=params['num_epochs'],
            patience=params['patience'],
            output_dir=experiment_dir
        )
        end_time = time.time()
        
        # Add experiment info
        experiment_results.update({
            'experiment_id': i+1,
            'params': params,
            'training_time': end_time - start_time
        })
        
        # Add to results
        results.append(experiment_results)
        
        # Save results
        with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    # Find best model
    best_result = max(results, key=lambda x: x['test_accuracy'])
    logger.info("\nBest Model:")
    logger.info(f"Experiment ID: {best_result['experiment_id']}")
    logger.info(f"Parameters: {best_result['params']}")
    logger.info(f"Test Accuracy: {best_result['test_accuracy']:.2f}%")
    logger.info(f"Test F1 Score: {best_result['test_f1_score']:.4f}")
    
    # Create symlink to best model
    best_model_path = os.path.join(output_dir, f"experiment_{best_result['experiment_id']}", 'best_model.pth')
    best_model_symlink = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(best_model_symlink):
        os.remove(best_model_symlink)
    os.symlink(best_model_path, best_model_symlink)
    
    # Plot results
    plot_hyperparameter_results(results, output_dir)

def plot_hyperparameter_results(results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Plot hyperparameter tuning results.
    
    Args:
        results: List of experiment results
        output_dir: Directory to save plots
    """
    # Extract data
    experiment_ids = [r['experiment_id'] for r in results]
    test_accuracies = [r['test_accuracy'] for r in results]
    test_f1_scores = [r['test_f1_score'] for r in results]
    val_losses = [r['best_val_loss'] for r in results]
    
    # Plot test accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(experiment_ids, test_accuracies)
    plt.xlabel('Experiment ID')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy by Experiment')
    plt.xticks(experiment_ids)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'test_accuracy_comparison.png'))
    
    # Plot test F1 score
    plt.figure(figsize=(10, 6))
    plt.bar(experiment_ids, test_f1_scores)
    plt.xlabel('Experiment ID')
    plt.ylabel('Test F1 Score')
    plt.title('Test F1 Score by Experiment')
    plt.xticks(experiment_ids)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'test_f1_score_comparison.png'))
    
    # Plot validation loss
    plt.figure(figsize=(10, 6))
    plt.bar(experiment_ids, val_losses)
    plt.xlabel('Experiment ID')
    plt.ylabel('Best Validation Loss')
    plt.title('Best Validation Loss by Experiment')
    plt.xticks(experiment_ids)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'val_loss_comparison.png'))

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train CNN-RNN model with hyperparameter tuning')
    parser.add_argument('--data_dir', type=str, default='data/wikiart', help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/hyperparameter_tuning', help='Directory to save outputs')
    parser.add_argument('--quick_run', action='store_true', help='Run a quick test with minimal hyperparameters')
    args = parser.parse_args()
    
    # Define hyperparameter grid
    if args.quick_run:
        # Quick run with minimal hyperparameters
        param_grid = {
            'model_type': ['cnn_only'],
            'cnn_backbone': ['resnet18'],
            'rnn_type': ['lstm'],
            'rnn_hidden_size': [128],
            'rnn_num_layers': [1],
            'batch_size': [16],
            'learning_rate': [0.001],
            'weight_decay': [0.01],
            'dropout': [0.2],
            'optimizer': ['adam'],
            'scheduler': ['reduce_on_plateau'],
            'num_epochs': [10],
            'patience': [3],
            'use_augmentation': [True]
        }
    else:
        # Full hyperparameter grid
        param_grid = {
            'model_type': ['cnn_only', 'cnn_rnn'],
            'cnn_backbone': ['resnet18', 'resnet50'],
            'rnn_type': ['lstm', 'gru'],
            'rnn_hidden_size': [128, 256],
            'rnn_num_layers': [1, 2],
            'batch_size': [16, 32],
            'learning_rate': [0.001, 0.0001],
            'weight_decay': [0.01, 0.001],
            'dropout': [0.2, 0.5],
            'optimizer': ['adam', 'sgd'],
            'scheduler': ['reduce_on_plateau', 'cosine_annealing'],
            'num_epochs': [50],
            'patience': [10],
            'use_augmentation': [True]
        }
    
    # Run hyperparameter tuning
    run_hyperparameter_tuning(args.data_dir, args.output_dir, param_grid)

if __name__ == "__main__":
    main()
