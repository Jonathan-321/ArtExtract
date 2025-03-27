"""
Model architecture for processing RGB images and multispectral masks.
Uses a dual-stream architecture with ResNet backbones.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Dict

class MultispectralModel(nn.Module):
    """Dual-stream model for processing RGB images and multispectral masks."""
    
    def __init__(
        self,
        num_classes: int,
        rgb_backbone: str = 'resnet18',
        ms_backbone: str = 'resnet18',
        pretrained: bool = True,
        fusion_method: str = 'concat'
    ):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of output classes
            rgb_backbone: Backbone model for RGB stream ('resnet18', 'resnet34', etc.)
            ms_backbone: Backbone model for multispectral stream
            pretrained: Whether to use pretrained weights for RGB stream
            fusion_method: How to fuse RGB and MS features ('concat', 'sum', 'attention')
        """
        super().__init__()
        self.fusion_method = fusion_method
        
        # RGB stream
        self.rgb_encoder = self._create_encoder(
            backbone_name=rgb_backbone,
            in_channels=3,
            pretrained=pretrained
        )
        
        # Multispectral stream
        self.ms_encoder = self._create_encoder(
            backbone_name=ms_backbone,
            in_channels=8,  # 8 spectral bands
            pretrained=False  # No pretrained weights for MS
        )
        
        # Get feature dimensions
        self.rgb_features = self._get_feature_dim(rgb_backbone)
        self.ms_features = self._get_feature_dim(ms_backbone)
        
        # Feature fusion
        if fusion_method == 'concat':
            fusion_dim = self.rgb_features + self.ms_features
        elif fusion_method in ['sum', 'attention']:
            # Project MS features to same dim as RGB if needed
            if self.ms_features != self.rgb_features:
                self.ms_proj = nn.Linear(self.ms_features, self.rgb_features)
            fusion_dim = self.rgb_features
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
            
        # Attention mechanism for feature fusion
        if fusion_method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(self.rgb_features * 2, self.rgb_features),
                nn.ReLU(),
                nn.Linear(self.rgb_features, 2),  # 2 attention weights
                nn.Softmax(dim=1)
            )
            
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def _create_encoder(
        self,
        backbone_name: str,
        in_channels: int,
        pretrained: bool
    ) -> nn.Module:
        """Create encoder network from specified backbone."""
        # Get the model class
        if not hasattr(models, backbone_name):
            raise ValueError(f"Unknown backbone: {backbone_name}")
        model_class = getattr(models, backbone_name)
        
        # Create model
        if pretrained:
            model = model_class(weights='DEFAULT')
        else:
            model = model_class(weights=None)
            
        # Modify first conv layer if needed
        if in_channels != 3:
            old_conv = model.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            model.conv1 = new_conv
            
        # Remove classification head
        return nn.Sequential(*list(model.children())[:-1])
    
    def _get_feature_dim(self, backbone_name: str) -> int:
        """Get feature dimension for backbone."""
        feature_dims = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048
        }
        return feature_dims.get(backbone_name, 512)
    
    def fuse_features(
        self,
        rgb_feat: torch.Tensor,
        ms_feat: torch.Tensor
    ) -> torch.Tensor:
        """Fuse RGB and MS features using specified method."""
        if self.fusion_method == 'concat':
            return torch.cat([rgb_feat, ms_feat], dim=1)
            
        if hasattr(self, 'ms_proj'):
            ms_feat = self.ms_proj(ms_feat)
            
        if self.fusion_method == 'sum':
            return rgb_feat + ms_feat
            
        if self.fusion_method == 'attention':
            # Compute attention weights
            combined = torch.cat([rgb_feat, ms_feat], dim=1)
            weights = self.attention(combined)
            # Apply attention
            return weights[:, 0:1] * rgb_feat + weights[:, 1:2] * ms_feat
            
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
                - logits: Classification logits
                - rgb_features: Features from RGB stream
                - ms_features: Features from MS stream
                - fused_features: Fused features
        """
        # Process RGB stream
        rgb_features = self.rgb_encoder(rgb_imgs)
        rgb_features = rgb_features.view(rgb_features.size(0), -1)
        
        # Process MS stream
        ms_features = self.ms_encoder(ms_masks)
        ms_features = ms_features.view(ms_features.size(0), -1)
        
        # Fuse features
        fused_features = self.fuse_features(rgb_features, ms_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'rgb_features': rgb_features,
            'ms_features': ms_features,
            'fused_features': fused_features
        }
        
    def get_features(
        self,
        rgb_imgs: torch.Tensor,
        ms_masks: torch.Tensor
    ) -> torch.Tensor:
        """Extract fused features for a batch of images."""
        with torch.no_grad():
            outputs = self.forward(rgb_imgs, ms_masks)
            return outputs['fused_features']
