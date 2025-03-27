"""
CNN-RNN Model for art style classification.
This module implements a convolutional-recurrent neural network for classifying art styles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CNNRNNModel(nn.Module):
    """
    CNN-RNN model for art style classification.
    
    This model combines a CNN backbone for feature extraction with RNN layers
    to capture sequential information in the visual features.
    """
    
    def __init__(self,
                num_classes: int,
                cnn_backbone: str = 'resnet50',
                pretrained: bool = True,
                rnn_type: str = 'lstm',
                rnn_hidden_size: int = 512,
                rnn_num_layers: int = 2,
                dropout: float = 0.5,
                bidirectional: bool = True):
        """
        Initialize the CNN-RNN model.
        
        Args:
            num_classes: Number of output classes
            cnn_backbone: CNN backbone architecture ('resnet50', 'efficientnet_b0', etc.)
            pretrained: Whether to use pretrained weights for the CNN backbone
            rnn_type: RNN type ('lstm' or 'gru')
            rnn_hidden_size: Hidden size of the RNN
            rnn_num_layers: Number of RNN layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional RNN
        """
        super(CNNRNNModel, self).__init__()
        
        self.num_classes = num_classes
        self.cnn_backbone = cnn_backbone
        self.rnn_type = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Initialize CNN backbone
        self._init_cnn_backbone(pretrained)
        
        # Initialize RNN
        self._init_rnn()
        
        # Initialize classifier
        rnn_output_size = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_output_size, num_classes)
        )
        
        logger.info(f"Initialized CNN-RNN model with {cnn_backbone} backbone and {rnn_type} RNN")
    
    def _init_cnn_backbone(self, pretrained: bool) -> None:
        """
        Initialize CNN backbone.
        
        Args:
            pretrained: Whether to use pretrained weights
        """
        if self.cnn_backbone == 'resnet50':
            self.cnn = models.resnet50(pretrained=pretrained)
            self.feature_dim = self.cnn.fc.in_features
            # Remove the final fully connected layer
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        
        elif self.cnn_backbone == 'resnet18':
            self.cnn = models.resnet18(pretrained=pretrained)
            self.feature_dim = self.cnn.fc.in_features
            # Remove the final fully connected layer
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        
        elif self.cnn_backbone == 'efficientnet_b0':
            self.cnn = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = self.cnn.classifier[1].in_features
            # Remove the classifier
            self.cnn = self.cnn.features
        
        elif self.cnn_backbone == 'mobilenet_v2':
            self.cnn = models.mobilenet_v2(pretrained=pretrained)
            self.feature_dim = self.cnn.classifier[1].in_features
            # Remove the classifier
            self.cnn = self.cnn.features
        
        else:
            raise ValueError(f"Unsupported CNN backbone: {self.cnn_backbone}")
    
    def _init_rnn(self) -> None:
        """Initialize RNN layers."""
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.feature_dim,
                hidden_size=self.rnn_hidden_size,
                num_layers=self.rnn_num_layers,
                batch_first=True,
                dropout=self.dropout if self.rnn_num_layers > 1 else 0,
                bidirectional=self.bidirectional
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=self.feature_dim,
                hidden_size=self.rnn_hidden_size,
                num_layers=self.rnn_num_layers,
                batch_first=True,
                dropout=self.dropout if self.rnn_num_layers > 1 else 0,
                bidirectional=self.bidirectional
            )
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # CNN feature extraction
        features = self.cnn(x)  # (batch_size, channels, height, width)
        
        # Reshape for RNN
        batch_size, channels, height, width = features.size()
        features = features.permute(0, 2, 3, 1)  # (batch_size, height, width, channels)
        features = features.reshape(batch_size, height * width, channels)  # (batch_size, seq_len, features)
        
        # RNN processing
        rnn_out, _ = self.rnn(features)  # (batch_size, seq_len, hidden_size*2 if bidirectional)
        
        # Use the last output of the RNN
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size*2 if bidirectional)
        
        # Classification
        output = self.classifier(last_output)  # (batch_size, num_classes)
        
        return output


class CNNAttentionRNNModel(nn.Module):
    """
    CNN-Attention-RNN model for art style classification.
    
    This model extends the CNN-RNN model with an attention mechanism
    to focus on important parts of the feature maps.
    """
    
    def __init__(self,
                num_classes: int,
                cnn_backbone: str = 'resnet50',
                pretrained: bool = True,
                rnn_type: str = 'lstm',
                rnn_hidden_size: int = 512,
                rnn_num_layers: int = 2,
                dropout: float = 0.5,
                bidirectional: bool = True):
        """
        Initialize the CNN-Attention-RNN model.
        
        Args:
            num_classes: Number of output classes
            cnn_backbone: CNN backbone architecture ('resnet50', 'efficientnet_b0', etc.)
            pretrained: Whether to use pretrained weights for the CNN backbone
            rnn_type: RNN type ('lstm' or 'gru')
            rnn_hidden_size: Hidden size of the RNN
            rnn_num_layers: Number of RNN layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional RNN
        """
        super(CNNAttentionRNNModel, self).__init__()
        
        self.num_classes = num_classes
        self.cnn_backbone = cnn_backbone
        self.rnn_type = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Initialize CNN backbone
        self._init_cnn_backbone(pretrained)
        
        # Initialize attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize RNN
        self._init_rnn()
        
        # Initialize classifier
        rnn_output_size = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_output_size, num_classes)
        )
        
        logger.info(f"Initialized CNN-Attention-RNN model with {cnn_backbone} backbone and {rnn_type} RNN")
    
    def _init_cnn_backbone(self, pretrained: bool) -> None:
        """
        Initialize CNN backbone.
        
        Args:
            pretrained: Whether to use pretrained weights
        """
        if self.cnn_backbone == 'resnet50':
            self.cnn = models.resnet50(pretrained=pretrained)
            self.feature_dim = self.cnn.fc.in_features
            # Remove the final fully connected layer
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        
        elif self.cnn_backbone == 'resnet18':
            self.cnn = models.resnet18(pretrained=pretrained)
            self.feature_dim = self.cnn.fc.in_features
            # Remove the final fully connected layer
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        
        elif self.cnn_backbone == 'efficientnet_b0':
            self.cnn = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = self.cnn.classifier[1].in_features
            # Remove the classifier
            self.cnn = self.cnn.features
        
        elif self.cnn_backbone == 'mobilenet_v2':
            self.cnn = models.mobilenet_v2(pretrained=pretrained)
            self.feature_dim = self.cnn.classifier[1].in_features
            # Remove the classifier
            self.cnn = self.cnn.features
        
        else:
            raise ValueError(f"Unsupported CNN backbone: {self.cnn_backbone}")
    
    def _init_rnn(self) -> None:
        """Initialize RNN layers."""
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.feature_dim,
                hidden_size=self.rnn_hidden_size,
                num_layers=self.rnn_num_layers,
                batch_first=True,
                dropout=self.dropout if self.rnn_num_layers > 1 else 0,
                bidirectional=self.bidirectional
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=self.feature_dim,
                hidden_size=self.rnn_hidden_size,
                num_layers=self.rnn_num_layers,
                batch_first=True,
                dropout=self.dropout if self.rnn_num_layers > 1 else 0,
                bidirectional=self.bidirectional
            )
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # CNN feature extraction
        features = self.cnn(x)  # (batch_size, channels, height, width)
        
        # Reshape for attention and RNN
        batch_size, channels, height, width = features.size()
        features_reshaped = features.permute(0, 2, 3, 1)  # (batch_size, height, width, channels)
        features_reshaped = features_reshaped.reshape(batch_size, height * width, channels)  # (batch_size, seq_len, features)
        
        # Attention mechanism
        attention_scores = self.attention(features_reshaped)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Apply attention weights
        attended_features = features_reshaped * attention_weights  # (batch_size, seq_len, features)
        
        # RNN processing
        rnn_out, _ = self.rnn(attended_features)  # (batch_size, seq_len, hidden_size*2 if bidirectional)
        
        # Use the last output of the RNN
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size*2 if bidirectional)
        
        # Classification
        output = self.classifier(last_output)  # (batch_size, num_classes)
        
        return output


def create_model(model_type: str,
                num_classes: int,
                cnn_backbone: str = 'resnet50',
                pretrained: bool = True,
                rnn_type: str = 'lstm',
                rnn_hidden_size: int = 512,
                rnn_num_layers: int = 2,
                dropout: float = 0.5,
                bidirectional: bool = True) -> nn.Module:
    """
    Create a model instance.
    
    Args:
        model_type: Type of model ('cnn_rnn' or 'cnn_attention_rnn')
        num_classes: Number of output classes
        cnn_backbone: CNN backbone architecture
        pretrained: Whether to use pretrained weights
        rnn_type: RNN type
        rnn_hidden_size: Hidden size of the RNN
        rnn_num_layers: Number of RNN layers
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional RNN
        
    Returns:
        Model instance
    """
    if model_type == 'cnn_rnn':
        return CNNRNNModel(
            num_classes=num_classes,
            cnn_backbone=cnn_backbone,
            pretrained=pretrained,
            rnn_type=rnn_type,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
    elif model_type == 'cnn_attention_rnn':
        return CNNAttentionRNNModel(
            num_classes=num_classes,
            cnn_backbone=cnn_backbone,
            pretrained=pretrained,
            rnn_type=rnn_type,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    print("CNN-RNN Model module")
    print("Use this module to create CNN-RNN models for art style classification.")
