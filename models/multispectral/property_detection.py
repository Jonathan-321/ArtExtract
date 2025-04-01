"""
Painting property detection module for the ArtExtract project.
This module extends the MultispectralModel to detect specific painting properties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .model import MultispectralModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define painting properties
PROPERTY_CATEGORIES = {
    'pigment_type': ['lead-based', 'organic', 'modern-synthetic', 'earth-pigment', 'other'],
    'damage_type': ['craquelure', 'water-damage', 'fading', 'discoloration', 'none'],
    'restoration': ['recent', 'historical', 'none'],
    'hidden_content': ['underdrawing', 'previous-painting', 'modification', 'none']
}

class PaintingPropertyDetector(nn.Module):
    """
    Extended MultispectralModel for detecting specific painting properties.
    """
    
    def __init__(
        self,
        rgb_backbone: str = 'resnet18',
        ms_backbone: str = 'resnet18',
        pretrained: bool = True,
        fusion_method: str = 'attention'
    ):
        """
        Initialize the PaintingPropertyDetector.
        
        Args:
            rgb_backbone: Backbone model for RGB stream
            ms_backbone: Backbone model for multispectral stream
            pretrained: Whether to use pretrained weights
            fusion_method: How to fuse features ('concat', 'sum', 'attention')
        """
        super().__init__()
        
        # Create base multispectral model (without classification head)
        self.base_model = MultispectralModel(
            num_classes=2,  # Placeholder, we'll replace the classifier
            rgb_backbone=rgb_backbone,
            ms_backbone=ms_backbone,
            pretrained=pretrained,
            fusion_method=fusion_method
        )
        
        # Get feature dimension
        if fusion_method == 'concat':
            feature_dim = self.base_model.rgb_features + self.base_model.ms_features
        else:
            feature_dim = self.base_model.rgb_features
            
        # Remove the original classifier
        delattr(self.base_model, 'classifier')
        
        # Create separate classifiers for each property
        self.property_classifiers = nn.ModuleDict()
        
        for property_name, categories in PROPERTY_CATEGORIES.items():
            num_categories = len(categories)
            self.property_classifiers[property_name] = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_categories)
            )
            
        # Create a feature attention module
        self.feature_attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        rgb_imgs: torch.Tensor,
        ms_masks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            rgb_imgs: RGB images tensor (B, 3, H, W)
            ms_masks: Multispectral masks tensor (B, 8, H, W)
            
        Returns:
            dict containing:
                - property logits for each property category
                - features
        """
        # Process RGB stream
        rgb_features = self.base_model.rgb_encoder(rgb_imgs)
        rgb_features = rgb_features.view(rgb_features.size(0), -1)
        
        # Process MS stream
        ms_features = self.base_model.ms_encoder(ms_masks)
        ms_features = ms_features.view(ms_features.size(0), -1)
        
        # Fuse features
        fused_features = self.base_model.fuse_features(rgb_features, ms_features)
        
        # Apply feature attention
        attention_weights = self.feature_attention(fused_features)
        attended_features = fused_features * attention_weights
        
        # Apply classifiers
        results = {
            'features': attended_features,
            'rgb_features': rgb_features,
            'ms_features': ms_features
        }
        
        # Add property logits
        for property_name, classifier in self.property_classifiers.items():
            results[f'{property_name}_logits'] = classifier(attended_features)
            
        return results
    
    def predict_properties(
        self,
        rgb_imgs: torch.Tensor,
        ms_masks: torch.Tensor
    ) -> Dict[str, List[str]]:
        """
        Predict painting properties.
        
        Args:
            rgb_imgs: RGB images tensor (B, 3, H, W)
            ms_masks: Multispectral masks tensor (B, 8, H, W)
            
        Returns:
            Dictionary mapping property names to predicted categories
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(rgb_imgs, ms_masks)
            
            predictions = {}
            for property_name in PROPERTY_CATEGORIES:
                logits = outputs[f'{property_name}_logits']
                pred_indices = torch.argmax(logits, dim=1)
                
                # Convert indices to category names
                categories = PROPERTY_CATEGORIES[property_name]
                pred_categories = [categories[idx.item()] for idx in pred_indices]
                
                predictions[property_name] = pred_categories
                
            return predictions


class PropertyDetectionTrainer:
    """Trainer class for PaintingPropertyDetector."""
    
    def __init__(
        self,
        model: PaintingPropertyDetector,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        save_dir: str,
        class_weights: Optional[Dict[str, torch.Tensor]] = None
    ):
        """Initialize the trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.class_weights = class_weights or {}
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_metrics = []
        self.val_metrics = []
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        property_correct = {prop: 0 for prop in PROPERTY_CATEGORIES}
        property_total = {prop: 0 for prop in PROPERTY_CATEGORIES}
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Unpack batch
            rgb_imgs = batch[0].to(self.device)
            ms_masks = batch[1].to(self.device)
            
            # Get property labels (in a real scenario, you would have these in your dataset)
            # For demonstration, we're creating dummy labels
            property_labels = self._get_dummy_labels(batch_size=rgb_imgs.size(0))
            
            # Forward pass
            outputs = self.model(rgb_imgs, ms_masks)
            
            # Compute loss for each property
            loss = 0
            for property_name in PROPERTY_CATEGORIES:
                logits = outputs[f'{property_name}_logits']
                labels = property_labels[property_name].to(self.device)
                
                # Apply class weights if available
                if property_name in self.class_weights:
                    criterion = nn.CrossEntropyLoss(weight=self.class_weights[property_name])
                else:
                    criterion = self.criterion
                    
                property_loss = criterion(logits, labels)
                loss += property_loss
                
                # Calculate accuracy
                _, predicted = logits.max(1)
                property_total[property_name] += labels.size(0)
                property_correct[property_name] += predicted.eq(labels).sum().item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                logger.info(f"Train Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.4f}")
        
        # Calculate average loss and accuracies
        avg_loss = total_loss / len(self.train_loader)
        accuracies = {
            f"{prop}_accuracy": 100. * property_correct[prop] / property_total[prop]
            for prop in PROPERTY_CATEGORIES
        }
        
        return {"loss": avg_loss, **accuracies}
        
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        property_correct = {prop: 0 for prop in PROPERTY_CATEGORIES}
        property_total = {prop: 0 for prop in PROPERTY_CATEGORIES}
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Unpack batch
                rgb_imgs = batch[0].to(self.device)
                ms_masks = batch[1].to(self.device)
                
                # Get property labels (dummy for demonstration)
                property_labels = self._get_dummy_labels(batch_size=rgb_imgs.size(0))
                
                # Forward pass
                outputs = self.model(rgb_imgs, ms_masks)
                
                # Compute loss for each property
                loss = 0
                for property_name in PROPERTY_CATEGORIES:
                    logits = outputs[f'{property_name}_logits']
                    labels = property_labels[property_name].to(self.device)
                    
                    property_loss = self.criterion(logits, labels)
                    loss += property_loss
                    
                    # Calculate accuracy
                    _, predicted = logits.max(1)
                    property_total[property_name] += labels.size(0)
                    property_correct[property_name] += predicted.eq(labels).sum().item()
                
                # Update metrics
                total_loss += loss.item()
        
        # Calculate average loss and accuracies
        avg_loss = total_loss / len(self.val_loader)
        accuracies = {
            f"{prop}_accuracy": 100. * property_correct[prop] / property_total[prop]
            for prop in PROPERTY_CATEGORIES
        }
        
        return {"loss": avg_loss, **accuracies}
    
    def _get_dummy_labels(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Create dummy labels for demonstration purposes.
        In a real scenario, these would come from your dataset.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Dictionary mapping property names to label tensors
        """
        labels = {}
        for property_name, categories in PROPERTY_CATEGORIES.items():
            num_categories = len(categories)
            # Random labels for demonstration
            labels[property_name] = torch.randint(0, num_categories, (batch_size,))
        return labels
        
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
        torch.save(checkpoint, self.save_dir / 'latest_property_detector_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_property_detector_model.pth')
            
    def train(self, num_epochs: int, save_freq: int = 5):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            save_freq: How often to save checkpoints
        """
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            self.train_metrics.append(train_metrics)
            
            # Log training metrics
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            for prop in PROPERTY_CATEGORIES:
                logger.info(f"Train {prop} Accuracy: {train_metrics[f'{prop}_accuracy']:.2f}%")
            
            # Validate
            val_metrics = self.validate()
            self.val_metrics.append(val_metrics)
            
            # Log validation metrics
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            for prop in PROPERTY_CATEGORIES:
                logger.info(f"Val {prop} Accuracy: {val_metrics[f'{prop}_accuracy']:.2f}%")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                
            if (epoch + 1) % save_freq == 0 or is_best:
                self.save_checkpoint({**train_metrics, **val_metrics}, is_best)
                
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training and validation metrics.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.train_metrics or not self.val_metrics:
            logger.warning("No training history to plot")
            return
            
        # Convert metrics to DataFrame
        train_df = pd.DataFrame(self.train_metrics)
        val_df = pd.DataFrame(self.val_metrics)
        
        # Create figure
        fig, axes = plt.subplots(len(PROPERTY_CATEGORIES) + 1, 1, figsize=(10, 4 * (len(PROPERTY_CATEGORIES) + 1)))
        
        # Plot loss
        axes[0].plot(train_df['loss'], label='Train Loss')
        axes[0].plot(val_df['loss'], label='Val Loss')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracies
        for i, prop in enumerate(PROPERTY_CATEGORIES):
            axes[i+1].plot(train_df[f'{prop}_accuracy'], label=f'Train {prop} Accuracy')
            axes[i+1].plot(val_df[f'{prop}_accuracy'], label=f'Val {prop} Accuracy')
            axes[i+1].set_title(f'{prop.replace("_", " ").title()} Accuracy')
            axes[i+1].set_xlabel('Epoch')
            axes[i+1].set_ylabel('Accuracy (%)')
            axes[i+1].legend()
            axes[i+1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
            
        return fig
    
    def visualize_property_predictions(
        self,
        rgb_img: torch.Tensor,
        ms_masks: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """
        Visualize property predictions for a single image.
        
        Args:
            rgb_img: RGB image tensor of shape (3, H, W)
            ms_masks: Multispectral masks tensor of shape (8, H, W)
            save_path: Optional path to save the visualization
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Add batch dimension
        rgb_img_batch = rgb_img.unsqueeze(0).to(self.device)
        ms_masks_batch = ms_masks.unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model.predict_properties(rgb_img_batch, ms_masks_batch)
        
        # Convert RGB image to numpy for visualization
        rgb_img_np = rgb_img.cpu().numpy().transpose(1, 2, 0)
        rgb_img_np = np.clip(rgb_img_np, 0, 1)  # Ensure values are in [0, 1]
        
        # Create visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot RGB image
        ax[0].imshow(rgb_img_np)
        ax[0].set_title('Original RGB Image')
        ax[0].axis('off')
        
        # Plot predictions as text
        ax[1].axis('off')
        ax[1].text(0.1, 0.9, 'Detected Properties:', fontsize=14, fontweight='bold')
        
        y_pos = 0.8
        for prop, value in predictions.items():
            ax[1].text(0.1, y_pos, f"{prop.replace('_', ' ').title()}: {value[0]}", fontsize=12)
            y_pos -= 0.1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Property prediction visualization saved to {save_path}")
            
        return fig


def main():
    """Main function to demonstrate property detection."""
    import argparse
    from torch.utils.data import DataLoader
    from data.preprocessing.multispectral_dataset import MultispectralDataset
    
    parser = argparse.ArgumentParser(description='Train painting property detector')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    train_dataset = MultispectralDataset(args.data_dir, split='train')
    val_dataset = MultispectralDataset(args.data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = PaintingPropertyDetector(
        rgb_backbone='resnet18',
        ms_backbone='resnet18',
        pretrained=True,
        fusion_method='attention'
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create trainer
    trainer = PropertyDetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir
    )
    
    # Train model
    trainer.train(num_epochs=args.epochs)
    
    # Plot training history
    trainer.plot_training_history(save_path=Path(args.save_dir) / 'property_detection_history.png')
    
    # Visualize example
    sample = next(iter(val_loader))
    rgb_img = sample[0][0]  # First image in batch
    ms_masks = sample[1][0]  # First mask in batch
    
    trainer.visualize_property_predictions(
        rgb_img=rgb_img,
        ms_masks=ms_masks,
        save_path=Path(args.save_dir) / 'property_detection_example.png'
    )
    
    logger.info("Training and visualization complete!")


if __name__ == '__main__':
    main()
