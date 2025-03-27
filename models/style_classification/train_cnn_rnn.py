"""
Training script for CNN-RNN model for art style/artist/genre classification.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path to import from project
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocessing.art_dataset_loader import WikiArtDataset
from data.preprocessing.data_preprocessing import ArtDatasetPreprocessor, create_data_loaders
from models.style_classification.cnn_rnn_model import CNNRNNModel
from evaluation.classification_metrics import (
    calculate_metrics, plot_confusion_matrix, plot_roc_curve, plot_training_history
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CNN-RNN model for art classification')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing the WikiArt dataset')
    parser.add_argument('--category', type=str, default='style', choices=['style', 'artist', 'genre'],
                        help='Classification category (style, artist, or genre)')
    
    # Model parameters
    parser.add_argument('--cnn_backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b3'],
                        help='CNN backbone architecture')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden size of the RNN')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Number of epochs to wait for improvement before stopping')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save model checkpoints and results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment (default: auto-generated)')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / total,
            'acc': 100 * correct / total
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / total,
                'acc': 100 * correct / total
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    if args.experiment_name is None:
        args.experiment_name = f"{args.category}_{args.cnn_backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load dataset
    logger.info(f"Loading WikiArt dataset from {args.data_dir}")
    dataset = WikiArtDataset(args.data_dir)
    df = dataset.create_dataframe(category=args.category)
    
    # Create data preprocessor and loaders
    logger.info("Creating data loaders")
    preprocessor = ArtDatasetPreprocessor(img_size=(224, 224), use_augmentation=True)
    dataloaders = create_data_loaders(
        dataframe=df,
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Get number of classes
    num_classes = len(df['label'].unique())
    logger.info(f"Number of classes: {num_classes}")
    
    # Create model
    logger.info(f"Creating CNN-RNN model with {args.cnn_backbone} backbone")
    model = CNNRNNModel(
        num_classes=num_classes,
        cnn_backbone=args.cnn_backbone,
        hidden_size=args.hidden_size,
        dropout=args.dropout
    )
    model = model.to(args.device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    
    best_val_acc = 0.0
    early_stopping_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, dataloaders['train'], criterion, optimizer, args.device
        )
        
        # Validate
        val_loss, val_acc, val_labels, val_preds, val_probs = validate(
            model, dataloaders['val'], criterion, args.device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint if best model
        if val_acc > best_val_acc:
            logger.info(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%")
            best_val_acc = val_acc
            early_stopping_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'args': vars(args)
            }, output_dir / 'best_model.pth')
            
            # Calculate and save validation metrics
            metrics = calculate_metrics(val_labels, val_preds, val_probs, list(range(num_classes)))
            with open(output_dir / 'best_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Plot confusion matrix
            plot_confusion_matrix(
                val_labels, val_preds, 
                class_names=[dataset.idx_to_style[i] if args.category == 'style' else 
                             dataset.idx_to_artist[i] if args.category == 'artist' else
                             dataset.idx_to_genre[i] for i in range(num_classes)],
                output_path=output_dir / 'confusion_matrix.png'
            )
            
            # Plot ROC curve
            plot_roc_curve(
                val_labels, val_probs, 
                class_names=[dataset.idx_to_style[i] if args.category == 'style' else 
                             dataset.idx_to_artist[i] if args.category == 'artist' else
                             dataset.idx_to_genre[i] for i in range(num_classes)],
                output_path=output_dir / 'roc_curve.png'
            )
        else:
            early_stopping_counter += 1
            logger.info(f"Validation accuracy did not improve. Best: {best_val_acc:.2f}%")
            logger.info(f"Early stopping counter: {early_stopping_counter}/{args.early_stopping}")
        
        # Early stopping
        if early_stopping_counter >= args.early_stopping:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training history
    plot_training_history(history, output_path=output_dir / 'training_history.png')
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'args': vars(args)
    }, output_dir / 'final_model.pth')
    
    # Final evaluation on test set
    logger.info("Evaluating on test set")
    test_loss, test_acc, test_labels, test_preds, test_probs = validate(
        model, dataloaders['test'], criterion, args.device
    )
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Calculate and save test metrics
    metrics = calculate_metrics(test_labels, test_preds, test_probs, list(range(num_classes)))
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot test confusion matrix
    plot_confusion_matrix(
        test_labels, test_preds, 
        class_names=[dataset.idx_to_style[i] if args.category == 'style' else 
                     dataset.idx_to_artist[i] if args.category == 'artist' else
                     dataset.idx_to_genre[i] for i in range(num_classes)],
        output_path=output_dir / 'test_confusion_matrix.png'
    )
    
    logger.info(f"Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
