#!/usr/bin/env python3
"""
Training script for CNN-RNN art classifier.
This script trains a CNN-RNN model for classifying art attributes like style, artist, and genre.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from models.classification.cnn_rnn_classifier import CNNRNNClassifier, ClassificationTrainer
from data.classification.wikiart_dataset import create_wikiart_dataloaders

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CNN-RNN art classifier')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to WikiArt dataset')
    parser.add_argument('--attributes', type=str, nargs='+', default=['style', 'artist', 'genre'],
                        help='Art attributes to classify')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='CNN backbone architecture')
    parser.add_argument('--rnn_hidden_size', type=int, default=512,
                        help='Hidden size of RNN layers')
    parser.add_argument('--rnn_num_layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='How often to save checkpoints (epochs)')
    parser.add_argument('--save_dir', type=str, default='./model_checkpoints/classification',
                        help='Directory to save checkpoints')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--limit_samples', type=int, default=None,
                        help='Limit number of samples (for testing)')
    
    return parser.parse_args()


def plot_training_curves(metrics, save_path):
    """
    Plot training and validation curves.
    
    Args:
        metrics: Dictionary of training metrics
        save_path: Path to save the plot
    """
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy for each attribute
    plt.subplot(2, 1, 2)
    for attr in metrics['train_acc'][0].keys():
        train_acc = [acc[attr] for acc in metrics['train_acc']]
        val_acc = [acc[attr] for acc in metrics['val_acc']]
        
        plt.plot(epochs, train_acc, label=f'Train {attr.capitalize()} Acc')
        plt.plot(epochs, val_acc, label=f'Val {attr.capitalize()} Acc')
    
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Training curves saved to {save_path}")


def main():
    """Main function to train the CNN-RNN classifier."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    limit_samples_dict = {'train': args.limit_samples, 'val': args.limit_samples // 5 if args.limit_samples else None}
    train_loader, val_loader, test_loader, class_weights = create_wikiart_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        attributes=args.attributes,
        limit_samples=limit_samples_dict
    )
    
    # Get number of classes for each attribute
    num_classes = {}
    for attr in args.attributes:
        # Get the number of classes from the class_weights tensor
        num_classes[attr] = len(class_weights[attr])
    
    logger.info(f"Number of classes: {num_classes}")
    
    # Create model
    logger.info(f"Creating CNN-RNN classifier with {args.backbone} backbone...")
    model = CNNRNNClassifier(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=args.pretrained,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create trainer
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_dir=save_dir,
        class_weights=class_weights
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        trainer.current_epoch = start_epoch
    
    # Train model
    logger.info(f"Training for {args.num_epochs} epochs...")
    trainer.train(num_epochs=args.num_epochs, save_freq=args.save_freq)
    
    # Plot training curves
    metrics_path = save_dir / 'training_metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        plot_path = save_dir / 'training_curves.png'
        plot_training_curves(metrics, plot_path)
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
