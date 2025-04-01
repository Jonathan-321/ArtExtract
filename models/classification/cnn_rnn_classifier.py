"""
CNN-RNN Classifier for art classification tasks.
This module implements a convolutional-recurrent architecture for classifying
Style, Artist, Genre, and other attributes in artwork.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CNNRNNClassifier(nn.Module):
    """
    CNN-RNN architecture for art classification.
    Uses a CNN backbone for feature extraction followed by recurrent layers
    to capture sequential relationships in the features.
    """
    
    def __init__(
        self,
        num_classes: Dict[str, int],
        backbone: str = 'resnet50',
        pretrained: bool = True,
        rnn_hidden_size: int = 512,
        rnn_num_layers: int = 2,
        dropout: float = 0.5
    ):
        """
        Initialize the CNN-RNN classifier.
        
        Args:
            num_classes: Dictionary mapping attribute names to number of classes
                         e.g., {'style': 27, 'artist': 23, 'genre': 10}
            backbone: CNN backbone architecture ('resnet18', 'resnet50', 'vgg16', etc.)
            pretrained: Whether to use pretrained weights for the backbone
            rnn_hidden_size: Hidden size of the RNN layers
            rnn_num_layers: Number of RNN layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.attribute_names = list(num_classes.keys())
        
        # Initialize CNN backbone
        self.backbone_name = backbone
        if backbone.startswith('resnet'):
            if backbone == 'resnet18':
                self.cnn = models.resnet18(pretrained=pretrained)
                feature_dim = 512
            elif backbone == 'resnet34':
                self.cnn = models.resnet34(pretrained=pretrained)
                feature_dim = 512
            elif backbone == 'resnet50':
                self.cnn = models.resnet50(pretrained=pretrained)
                feature_dim = 2048
            else:
                raise ValueError(f"Unsupported ResNet variant: {backbone}")
                
            # Remove the final fully connected layer
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
            
        elif backbone.startswith('vgg'):
            if backbone == 'vgg16':
                self.cnn = models.vgg16(pretrained=pretrained).features
                feature_dim = 512
            else:
                raise ValueError(f"Unsupported VGG variant: {backbone}")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feature_dim, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # RNN for sequential feature processing
        self.rnn = nn.GRU(
            input_size=feature_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_num_layers > 1 else 0
        )
        
        # Final classifiers (one for each attribute)
        self.classifiers = nn.ModuleDict()
        rnn_output_dim = rnn_hidden_size * 2  # bidirectional
        
        for attr_name, num_cls in num_classes.items():
            self.classifiers[attr_name] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(rnn_output_dim, rnn_output_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(rnn_output_dim // 2, num_cls)
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Dictionary mapping attribute names to logits
        """
        batch_size = x.size(0)
        
        # Extract CNN features
        features = self.cnn(x)  # (batch_size, channels, h, w)
        
        # Apply spatial attention
        attention_weights = self.spatial_attention(features)
        attended_features = features * attention_weights
        
        # Reshape for RNN: (batch_size, seq_len, features)
        # Treat spatial dimensions as sequence length
        c, h, w = attended_features.size(1), attended_features.size(2), attended_features.size(3)
        attended_features = attended_features.view(batch_size, c, h * w)
        attended_features = attended_features.permute(0, 2, 1)  # (batch_size, h*w, c)
        
        # Apply RNN
        rnn_out, _ = self.rnn(attended_features)
        
        # Use the last output of the RNN
        rnn_features = rnn_out[:, -1, :]
        
        # Apply classifiers
        results = {}
        for attr_name in self.attribute_names:
            results[attr_name] = self.classifiers[attr_name](rnn_features)
            
        return results
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions for the input.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Dictionary mapping attribute names to class predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = {}
            
            for attr_name, attr_logits in logits.items():
                predictions[attr_name] = torch.argmax(attr_logits, dim=1)
                
            return predictions
    
    def get_outlier_scores(
        self, 
        x: torch.Tensor, 
        method: str = 'softmax_uncertainty'
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate outlier scores for the input.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            method: Method to calculate outlier scores
                    - 'softmax_uncertainty': 1 - max(softmax)
                    - 'entropy': Entropy of softmax distribution
                    
        Returns:
            Dictionary mapping attribute names to outlier scores
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            outlier_scores = {}
            
            for attr_name, attr_logits in logits.items():
                if method == 'softmax_uncertainty':
                    # 1 - max probability
                    probs = F.softmax(attr_logits, dim=1)
                    outlier_scores[attr_name] = 1 - torch.max(probs, dim=1)[0]
                    
                elif method == 'entropy':
                    # Entropy of the probability distribution
                    probs = F.softmax(attr_logits, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    # Normalize by maximum entropy (log of number of classes)
                    max_entropy = torch.log(torch.tensor(self.num_classes[attr_name], 
                                                        dtype=torch.float32))
                    outlier_scores[attr_name] = entropy / max_entropy
                    
                else:
                    raise ValueError(f"Unsupported outlier score method: {method}")
                    
            return outlier_scores


class ClassificationTrainer:
    """Trainer class for CNN-RNN art classifier."""
    
    def __init__(
        self,
        model: CNNRNNClassifier,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        save_dir: str,
        class_weights: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: CNN-RNN classifier model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            device: Device to use
            save_dir: Directory to save checkpoints
            class_weights: Optional weights for each class to handle imbalanced data
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.class_weights = class_weights
        
        # Loss functions
        self.criterion = {}
        for attr_name, num_classes in model.num_classes.items():
            if class_weights and attr_name in class_weights:
                weights = class_weights[attr_name].to(device)
                self.criterion[attr_name] = nn.CrossEntropyLoss(weight=weights)
            else:
                self.criterion[attr_name] = nn.CrossEntropyLoss()
                
        # Metrics tracking
        self.best_val_acc = {attr: 0.0 for attr in model.attribute_names}
        self.current_epoch = 0
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = {attr: 0 for attr in self.model.attribute_names}
        total = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data = data.to(self.device)
            # Move each target tensor to device
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            # Calculate loss for each attribute
            loss = 0.0
            for attr_name in self.model.attribute_names:
                attr_loss = self.criterion[attr_name](outputs[attr_name], targets[attr_name])
                loss += attr_loss
                
                # Calculate accuracy
                _, predicted = outputs[attr_name].max(1)
                correct[attr_name] += predicted.eq(targets[attr_name]).sum().item()
                
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total += data.size(0)
            
            if batch_idx % 10 == 0:
                logger.info(f'Train Epoch: {self.current_epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                           f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / len(self.train_loader)
        accuracy = {attr: 100. * correct[attr] / total for attr in self.model.attribute_names}
        
        logger.info(f'Train Epoch: {self.current_epoch}\tAverage Loss: {avg_loss:.4f}')
        for attr, acc in accuracy.items():
            logger.info(f'Train Accuracy ({attr}): {acc:.2f}%')
            
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        correct = {attr: 0 for attr in self.model.attribute_names}
        total = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data = data.to(self.device)
                # Move each target tensor to device
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                outputs = self.model(data)
                
                # Calculate loss for each attribute
                batch_loss = 0.0
                for attr_name in self.model.attribute_names:
                    attr_loss = self.criterion[attr_name](outputs[attr_name], targets[attr_name])
                    batch_loss += attr_loss
                    
                    # Calculate accuracy
                    _, predicted = outputs[attr_name].max(1)
                    correct[attr_name] += predicted.eq(targets[attr_name]).sum().item()
                
                val_loss += batch_loss.item()
                total += data.size(0)
        
        # Calculate average loss and accuracy
        avg_loss = val_loss / len(self.val_loader)
        accuracy = {attr: 100. * correct[attr] / total for attr in self.model.attribute_names}
        
        logger.info(f'Validation Epoch: {self.current_epoch}\tAverage Loss: {avg_loss:.4f}')
        for attr, acc in accuracy.items():
            logger.info(f'Validation Accuracy ({attr}): {acc:.2f}%')
            
        return avg_loss, accuracy
    
    def train(self, num_epochs, save_freq=5):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            save_freq: How often to save checkpoints
        """
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(
                    metrics={
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_acc': train_acc,
                        'val_acc': val_acc
                    }
                )
            
            # Save best model for each attribute
            for attr in self.model.attribute_names:
                if val_acc[attr] > self.best_val_acc[attr]:
                    self.best_val_acc[attr] = val_acc[attr]
                    self.save_checkpoint(
                        metrics={
                            'epoch': epoch,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'train_acc': train_acc,
                            'val_acc': val_acc
                        },
                        is_best=True,
                        attr_name=attr
                    )
    
    def save_checkpoint(self, metrics, is_best=False, attr_name=None):
        """
        Save a checkpoint of the model.
        
        Args:
            metrics: Dictionary of metrics to save
            is_best: Whether this is the best model so far
            attr_name: If is_best is True, the attribute for which this is the best model
        """
        import os
        from pathlib import Path
        
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best and attr_name:
            checkpoint_path = save_dir / f'best_{attr_name}_model.pth'
        else:
            checkpoint_path = save_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'metrics': metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f'Checkpoint saved to {checkpoint_path}')
