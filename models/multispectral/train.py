"""
Training script for the MultispectralModel.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import json
from tqdm import tqdm

from model import MultispectralModel
from data.preprocessing.multispectral_dataset import create_dataloaders

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for MultispectralModel."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        save_dir: str,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """Initialize the trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.scheduler = scheduler
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        with tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}') as pbar:
            for batch in pbar:
                rgb_imgs = batch[0]
                ms_masks = batch[1]
                # Create binary labels (0 for real, 1 for fake)
                targets = torch.tensor([
                    1 if 'fake' in path.lower() else 0
                    for path in batch[2]  # Assuming this is where filenames are stored
                ], device=self.device)
                # Move data to device
                rgb_imgs = rgb_imgs.to(self.device)
                ms_masks = ms_masks.to(self.device)
                
                # Forward pass
                outputs = self.model(rgb_imgs, ms_masks)
                loss = self.criterion(outputs['logits'], targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
                
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }
        
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                rgb_imgs = batch[0]
                ms_masks = batch[1]
                # Create binary labels (0 for real, 1 for fake)
                targets = torch.tensor([
                    1 if 'fake' in path.lower() else 0
                    for path in batch[2]  # Assuming this is where filenames are stored
                ], device=self.device)
                # Move data to device
                rgb_imgs = rgb_imgs.to(self.device)
                ms_masks = ms_masks.to(self.device)
                
                # Forward pass
                outputs = self.model(rgb_imgs, ms_masks)
                loss = self.criterion(outputs['logits'], targets)
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': 100. * correct / total
        }
        
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save a checkpoint of the model."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
            
    def train(
        self,
        num_epochs: int,
        patience: int = 10,
        save_freq: int = 5
    ):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            patience: Early stopping patience
            save_freq: How often to save checkpoints
        """
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Train Accuracy: {train_metrics['accuracy']:.2f}%")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Accuracy: {val_metrics['accuracy']:.2f}%")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['loss'])
            
            # Save metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
                
            if (epoch + 1) % save_freq == 0 or is_best:
                self.save_checkpoint({**train_metrics, **val_metrics}, is_best)
                
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
                
        # Save final metrics
        metrics_path = self.save_dir / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }, f)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir='data/multispectral',
        batch_size=32,
        num_workers=4
    )
    
    # Create model for binary classification (real vs fake)
    model = MultispectralModel(
        num_classes=2,  # Binary classification: real vs fake
        rgb_backbone='resnet18',
        ms_backbone='resnet18',
        fusion_method='attention'
    ).to(device)
    
    logger.info(f"Created model with architecture:\n{model}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir='models/multispectral/checkpoints',
        scheduler=scheduler
    )
    
    # Train
    trainer.train(
        num_epochs=100,
        patience=15,
        save_freq=5
    )

if __name__ == '__main__':
    main()
