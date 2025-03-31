"""
Training script for the CNN-RNN model using the refined WikiArt dataset.
This script is optimized for the ArtGAN refined WikiArt dataset structure.
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
from torch.utils.data import DataLoader, Dataset
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocessing.data_preprocessing import ArtDatasetPreprocessor
from models.style_classification.cnn_rnn_model import create_model
from evaluation.classification_metrics import ClassificationEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WikiArtRefinedDataset(Dataset):
    """Dataset class for the refined WikiArt dataset."""
    
    def __init__(self, csv_file: str, wikiart_dir: str, transform=None):
        """
        Args:
            csv_file: Path to the CSV file with image paths and labels
            wikiart_dir: Directory containing the WikiArt images
            transform: Optional transform to be applied on a sample
        """
        # Check if csv_file exists, if not try to find it in subdirectories
        csv_path = Path(csv_file)
        if not csv_path.exists():
            # Try to find in wikiart subdirectory
            alt_csv_path = Path(wikiart_dir) / 'wikiart' / csv_path.name
            if alt_csv_path.exists():
                csv_file = str(alt_csv_path)
                logger.info(f"Using CSV file from alternate location: {csv_file}")
            else:
                raise FileNotFoundError(f"CSV file not found at {csv_path} or {alt_csv_path}")
        
        self.data = pd.read_csv(csv_file, header=None)
        self.wikiart_dir = Path(wikiart_dir)
        self.transform = transform
        
        # Load class mapping
        self.class_file = csv_file.replace('_train.csv', '_class.txt').replace('_val.csv', '_class.txt')
        
        # Check if class file exists, if not try to find it in subdirectories
        class_path = Path(self.class_file)
        if not class_path.exists():
            # Try to find in wikiart subdirectory
            alt_class_path = Path(wikiart_dir) / 'wikiart' / class_path.name
            if alt_class_path.exists():
                self.class_file = str(alt_class_path)
                logger.info(f"Using class file from alternate location: {self.class_file}")
        self.class_to_idx = {}
        if os.path.exists(self.class_file):
            with open(self.class_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        idx = int(parts[0])
                        class_name = ' '.join(parts[1:])
                        self.class_to_idx[class_name] = idx
        
        logger.info(f"Loaded {len(self.data)} samples from {csv_file}")
        
    def __len__(self):
        return len(self.data)
    
    # Class variables to track progress and cache paths
    _processed_count = 0
    _total_count = 0
    _path_cache = {}
    _last_log_time = 0
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get the relative path from CSV
        rel_path = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])
        
        # Use cached path if available
        if rel_path in WikiArtRefinedDataset._path_cache:
            img_path = WikiArtRefinedDataset._path_cache[rel_path]
        else:
            # Extract style category and filename from the relative path
            path_parts = rel_path.split('/')
            if len(path_parts) >= 3 and path_parts[0] == 'train':
                # CSV format is 'train/Style_Category/filename.jpg'
                style_category = path_parts[1]  # The style category (e.g., 'Impressionism')
                filename = path_parts[2]        # The filename (e.g., 'monet_water-lilies.jpg')
            elif len(path_parts) >= 2:
                style_category = path_parts[-2]  # The style category
                filename = path_parts[-1]        # The filename
            else:
                style_category = ""
                filename = path_parts[-1]
            
            # Based on the actual file structure we found, the most likely path is:
            # data/wikiart_full/wikiart/Style_Category/filename.jpg
            img_path = self.wikiart_dir / "wikiart" / style_category / filename
            
            # Cache the path for future use
            WikiArtRefinedDataset._path_cache[rel_path] = img_path
        
        # Update progress counter
        WikiArtRefinedDataset._processed_count += 1
        if WikiArtRefinedDataset._total_count == 0:
            WikiArtRefinedDataset._total_count = len(self.data)
        
        # Log progress every 1000 items or every 10 seconds
        current_time = time.time()
        if (WikiArtRefinedDataset._processed_count % 1000 == 0 or 
                current_time - WikiArtRefinedDataset._last_log_time > 10):
            logger.info(f"Loading images: {WikiArtRefinedDataset._processed_count}/{WikiArtRefinedDataset._total_count} "
                       f"({WikiArtRefinedDataset._processed_count/WikiArtRefinedDataset._total_count*100:.1f}%)")
            WikiArtRefinedDataset._last_log_time = current_time
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            # If the primary path fails, try these fallback paths
            fallback_paths = [
                self.wikiart_dir / rel_path,                              # Original path as in CSV
                self.wikiart_dir / style_category / filename,             # No wikiart subdirectory
                Path(self.wikiart_dir.parent) / "wikiart" / style_category / filename  # Try parent dir
            ]
            
            for path in fallback_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    
                    if self.transform:
                        image = self.transform(image)
                    
                    # Update the cache with the correct path
                    WikiArtRefinedDataset._path_cache[rel_path] = path
                    return image, label
                except Exception:
                    continue
            
            # If all paths fail, log the error and return a placeholder
            logger.warning(f"Error loading image {rel_path}: Could not find at {img_path} or fallback locations")
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, label

def debug_dataset(dataset, num_samples=5):
    """
    Debug function to check if images can be loaded from the dataset.
    
    Args:
        dataset: Dataset to debug
        num_samples: Number of samples to check
    """
    logger.info(f"Debugging dataset with {len(dataset)} samples")
    success = 0
    for i in range(min(num_samples, len(dataset))):
        try:
            # Get sample without using the dataset's __getitem__ to avoid placeholder images
            rel_path = dataset.data.iloc[i, 0]
            label = int(dataset.data.iloc[i, 1])
            
            # Parse the path
            path_parts = rel_path.split('/')
            if len(path_parts) >= 3 and path_parts[0] == 'train':
                style_category = path_parts[1]
                filename = path_parts[2]
            elif len(path_parts) >= 2:
                style_category = path_parts[-2]
                filename = path_parts[-1]
            else:
                style_category = ""
                filename = path_parts[-1]
            
            # Try the primary path
            img_path = dataset.wikiart_dir / "wikiart" / style_category / filename
            logger.info(f"Sample {i}: Trying to load {img_path}")
            
            # Check if file exists
            if os.path.exists(img_path):
                logger.info(f"  ✓ File exists at {img_path}")
                success += 1
            else:
                logger.info(f"  ✗ File does not exist at {img_path}")
                
                # Try fallback paths
                fallback_paths = [
                    dataset.wikiart_dir / rel_path,
                    dataset.wikiart_dir / style_category / filename,
                    Path(dataset.wikiart_dir.parent) / "wikiart" / style_category / filename
                ]
                
                for path in fallback_paths:
                    if os.path.exists(path):
                        logger.info(f"  ✓ File exists at fallback path: {path}")
                        success += 1
                        break
                else:
                    logger.info(f"  ✗ File not found in any location for {rel_path}")
        except Exception as e:
            logger.error(f"Error debugging sample {i}: {str(e)}")
    
    logger.info(f"Successfully found {success}/{num_samples} images")
    return success > 0

class PrefetchLoader(DataLoader):
    """
    DataLoader with prefetching capability for faster data loading.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
    def __iter__(self):
        iterator = super().__iter__()
        self.preload = None
        self.preload_stream = self.stream
        
        # Start prefetching the first batch
        try:
            self.preload = next(iterator)
        except StopIteration:
            return
            
        for batch in iterator:
            if self.preload_stream is not None:
                torch.cuda.current_stream().wait_stream(self.preload_stream)
            
            # Get the preloaded batch
            curr_batch = self.preload
            
            # Start prefetching the next batch
            self.preload = batch
            
            yield curr_batch
            
        # Return the last preloaded batch
        if self.preload is not None:
            yield self.preload


def create_dataloaders(args):
    """
    Create dataloaders for training, validation, and testing.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of dataloaders
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    # Check for dataset structure
    wikiart_dir = os.path.join(args.data_dir, "wikiart")
    if os.path.exists(wikiart_dir) and os.path.isdir(wikiart_dir):
        logger.info(f"Found wikiart subdirectory at {wikiart_dir}")
        # Images are in the wikiart subdirectory
        csv_dir = args.data_dir
        if os.path.exists(os.path.join(wikiart_dir, f"{args.task}_train.csv")):
            logger.info(f"Using CSV files from wikiart subdirectory")
            csv_dir = wikiart_dir
    else:
        # No wikiart subdirectory, assume flat structure
        wikiart_dir = args.data_dir
        csv_dir = args.data_dir
    
    logger.info(f"Using CSV files from {csv_dir}")
    logger.info(f"Using images from {wikiart_dir}")
    
    train_dataset = WikiArtRefinedDataset(
        csv_file=os.path.join(csv_dir, f"{args.task}_train.csv"),
        wikiart_dir=args.data_dir,
        transform=train_transform
    )
    
    val_dataset = WikiArtRefinedDataset(
        csv_file=os.path.join(csv_dir, f"{args.task}_val.csv"),
        wikiart_dir=args.data_dir,
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create a small test set from validation set
    test_size = min(int(len(val_dataset) * 0.3), 1000)  # 30% of val set or 1000 samples, whichever is smaller
    test_indices = torch.randperm(len(val_dataset))[:test_size]
    test_dataset = torch.utils.data.Subset(val_dataset, test_indices)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders with {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test samples")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

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
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}: Loss: {loss.item():.4f}")
    
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
               args) -> Dict[str, Any]:
    """
    Train and validate the model.
    
    Args:
        model: Model to train
        dataloaders: Dictionary of dataloaders for train, val, test
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        args: Command line arguments
        
    Returns:
        Dictionary with training results
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        
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
        
        epoch_time = time.time() - epoch_start_time
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} (Time: {epoch_time:.2f}s):")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            
            logger.info(f"  Saved new best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
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
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = validate(
        model, dataloaders['test'], criterion, device
    )
    
    # Get class names if available
    class_file = os.path.join(args.data_dir, f"{args.task}_class.txt")
    class_names = []
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            class_dict = {}
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    idx = int(parts[0])
                    class_name = ' '.join(parts[1:])
                    class_dict[idx] = class_name
            
            # Sort by index
            class_names = [class_dict.get(i, str(i)) for i in range(max(class_dict.keys()) + 1)]
    else:
        class_names = [str(i) for i in range(len(np.unique(test_labels)))]
    
    # Create evaluator
    evaluator = ClassificationEvaluator(
        num_classes=len(np.unique(test_labels)),
        class_names=class_names
    )
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(test_labels, test_preds)
    
    # Create confusion matrix
    evaluator.plot_confusion_matrix(
        test_labels, test_preds, 
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Log results
    logger.info("\nTest Results:")
    logger.info(f"Loss: {test_loss:.4f}")
    logger.info(f"Accuracy: {test_acc:.2f}%")
    logger.info(f"F1 Score: {metrics['f1_score_weighted']:.4f}")
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_f1_score': metrics['f1_score_weighted'],
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'class_names': class_names,
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }, f, indent=4)
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_f1_score': metrics['f1_score_weighted'],
        'best_val_loss': best_val_loss,
        'epochs_trained': len(train_losses)
    }

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train CNN-RNN model on refined WikiArt dataset')
    parser.add_argument('--data_dir', type=str, default='data/wikiart_refined', help='Directory containing the refined WikiArt dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/wikiart_refined', help='Directory to save outputs')
    parser.add_argument('--task', type=str, default='style', choices=['style', 'artist', 'genre'], help='Classification task')
    parser.add_argument('--model_type', type=str, default='cnn_rnn', choices=['cnn_only', 'cnn_rnn'], help='Model type')
    parser.add_argument('--cnn_backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50'], help='CNN backbone')
    parser.add_argument('--rnn_type', type=str, default='lstm', choices=['lstm', 'gru'], help='RNN type')
    parser.add_argument('--rnn_hidden_size', type=int, default=256, help='RNN hidden size')
    parser.add_argument('--rnn_num_layers', type=int, default=2, help='RNN number of layers')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(args)
    
    # Debug the dataset to check if images can be loaded
    logger.info("Debugging training dataset...")
    train_debug_success = debug_dataset(dataloaders['train'].dataset, num_samples=10)
    logger.info("Debugging validation dataset...")
    val_debug_success = debug_dataset(dataloaders['val'].dataset, num_samples=5)
    
    if not train_debug_success or not val_debug_success:
        logger.error("Failed to load images from the dataset. Please check the paths and file structure.")
        return
    
    # Get number of classes from class file
    class_file = os.path.join(args.data_dir, f"{args.task}_class.txt")
    num_classes = 0
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            lines = f.readlines()
            num_classes = max([int(line.strip().split()[0]) for line in lines]) + 1
    else:
        # Get number of classes from dataset
        train_labels = []
        for _, labels in dataloaders['train']:
            train_labels.extend(labels.numpy())
        num_classes = len(np.unique(train_labels))
    
    logger.info(f"Number of classes: {num_classes}")
    
    # Create model
    if args.model_type == 'cnn_rnn':
        model = create_model(
            num_classes=num_classes,
            cnn_backbone=args.cnn_backbone,
            rnn_type=args.rnn_type,
            rnn_hidden_size=args.rnn_hidden_size,
            rnn_num_layers=args.rnn_num_layers,
            dropout=args.dropout
        )
    else:
        # Use a simple CNN model
        if args.cnn_backbone == 'resnet18':
            model = torchvision.models.resnet18(pretrained=True)
        elif args.cnn_backbone == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unknown CNN backbone: {args.cnn_backbone}")
        
        # Replace the last fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(num_features, num_classes)
        )
    
    model = model.to(device)
    
    # Print model summary
    logger.info(f"Model architecture: {args.model_type}")
    logger.info(f"CNN backbone: {args.cnn_backbone}")
    if args.model_type == 'cnn_rnn':
        logger.info(f"RNN type: {args.rnn_type}")
        logger.info(f"RNN hidden size: {args.rnn_hidden_size}")
        logger.info(f"RNN num layers: {args.rnn_num_layers}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            logger.warning(f"No checkpoint found at {args.resume}")
    
    # Train model
    train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args
    )
    
    logger.info("Training complete")

if __name__ == "__main__":
    main()
