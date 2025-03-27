"""
Test training script for the CNN-RNN model using a small test dataset.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.models

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
    from data.preprocessing.art_dataset_loader import WikiArtDataset
    
    # Initialize dataset
    dataset = WikiArtDataset(data_dir)
    
    # Create DataFrame with style labels
    df = dataset.create_dataframe()
    
    logger.info(f"Loaded {len(df)} images from WikiArt dataset")
    logger.info(f"Styles: {df['style'].unique().tolist()}")
    
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

def main():
    # Configuration
    data_dir = 'data/wikiart'  # WikiArt dataset directory
    batch_size = 32  # Larger batch size for more data
    num_epochs = 100  # More epochs for larger dataset
    learning_rate = 0.001  # Higher learning rate
    weight_decay = 0.01  # Moderate L2 regularization
    patience = 15  # More patience for convergence
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # Load test dataset
    df = load_test_dataset(test_data_dir)
    
    # Create preprocessor with augmentation
    preprocessor = ArtDatasetPreprocessor(
        img_size=(224, 224),
        use_augmentation=True,  # Enable augmentation
        normalize=True  # Use normalization
    )
    
    # Create data loaders
    dataloaders = create_data_loaders(
        df,
        preprocessor,
        batch_size=batch_size,
        train_ratio=0.5,  # Equal split for better validation
        val_ratio=0.25,
        test_ratio=0.25,
        num_workers=0,  # Use 0 for small dataset
        target_column='style'
    )
    
    # Create model with dropout
    num_classes = len(df['style'].unique())
    # Load pre-trained ResNet18
    model = torchvision.models.resnet18(pretrained=True)
    
    # Freeze all layers except the last few
    for param in list(model.parameters())[:-4]:  # Keep last residual block trainable
        param.requires_grad = False
    
    # Replace the last fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, num_classes)
    )
    model = model.to(device)
    
    # Loss and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        verbose=True
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 5  # Early stopping patience
    patience_counter = 0
    
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, dataloaders['val'], criterion, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'outputs/test_best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('outputs/test_training_history.png')
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('outputs/test_best_model.pth'))
    
    # Final evaluation
    test_loss, test_acc, test_preds, test_labels = validate(model, dataloaders['test'], criterion, device)
    
    # Initialize evaluator
    evaluator = ClassificationEvaluator(list(df['style'].unique()))
    
    # Compute metrics
    metrics = evaluator.compute_metrics(np.array(test_labels), np.array(test_preds))
    
    logger.info("\nTest Results:")
    logger.info(f"Loss: {test_loss:.4f}")
    logger.info(f"Accuracy: {test_acc:.2f}%")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        np.array(test_labels),
        np.array(test_preds),
        save_path='outputs/test_confusion_matrix.png'
    )

if __name__ == "__main__":
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    main()
