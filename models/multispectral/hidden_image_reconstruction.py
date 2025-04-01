"""
Hidden image reconstruction module for the ArtExtract project.
This module implements methods to reconstruct hidden images from multispectral data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HiddenImageReconstructor(nn.Module):
    """
    Neural network for reconstructing hidden images from multispectral data.
    Uses a U-Net style architecture to generate hidden image content.
    """
    
    def __init__(
        self,
        in_channels: int = 8,  # 8 spectral bands
        out_channels: int = 3,  # RGB output
        features: List[int] = [64, 128, 256, 512]
    ):
        """
        Initialize the HiddenImageReconstructor.
        
        Args:
            in_channels: Number of input channels (spectral bands)
            out_channels: Number of output channels (typically 3 for RGB)
            features: List of feature dimensions for each level of the U-Net
        """
        super().__init__()
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        # First encoder block (no downsampling)
        self.encoder_blocks.append(self._create_conv_block(in_channels, features[0]))
        
        # Remaining encoder blocks (with downsampling)
        for i in range(len(features) - 1):
            self.encoder_blocks.append(
                self._create_conv_block(features[i], features[i + 1])
            )
            
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        
        # Create upsampling and decoder blocks
        for i in range(len(features) - 1, 0, -1):
            self.upsampling_layers.append(
                nn.ConvTranspose2d(
                    features[i], features[i - 1], 
                    kernel_size=2, stride=2
                )
            )
            self.decoder_blocks.append(
                self._create_conv_block(features[i], features[i - 1])
            )
            
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _create_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block with two conv layers and batch normalization."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where C is the number of spectral bands
            
        Returns:
            Reconstructed hidden image tensor of shape (B, 3, H, W)
        """
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if i < len(self.encoder_blocks) - 1:
                skip_connections.append(x)
                x = F.max_pool2d(x, kernel_size=2, stride=2)
                
        # Decoder path with skip connections
        for i in range(len(self.decoder_blocks)):
            x = self.upsampling_layers[i](x)
            skip = skip_connections.pop()
            
            # Handle potential size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[i](x)
            
        # Final layer
        x = self.final_conv(x)
        
        # Apply sigmoid to get values in [0, 1] range
        return torch.sigmoid(x)


class HiddenImageTrainer:
    """Trainer class for HiddenImageReconstructor."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        save_dir: str
    ):
        """Initialize the trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.perceptual_loss = nn.L1Loss()  # Simple perceptual loss
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            # Unpack batch
            rgb_imgs = batch[0].to(self.device)
            ms_masks = batch[1].to(self.device)
            
            # Forward pass
            reconstructed = self.model(ms_masks)
            
            # Compute loss
            # For simplicity, we use the RGB image as the target
            # In a real scenario, you would use ground truth hidden images
            loss = self.reconstruction_loss(reconstructed, rgb_imgs)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
        return {'loss': total_loss / len(self.train_loader)}
        
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Unpack batch
                rgb_imgs = batch[0].to(self.device)
                ms_masks = batch[1].to(self.device)
                
                # Forward pass
                reconstructed = self.model(ms_masks)
                
                # Compute loss
                loss = self.reconstruction_loss(reconstructed, rgb_imgs)
                
                # Update metrics
                total_loss += loss.item()
                
        return {'loss': total_loss / len(self.val_loader)}
        
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
        torch.save(checkpoint, self.save_dir / 'latest_reconstructor_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_reconstructor_model.pth')
            
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
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                
            if (epoch + 1) % save_freq == 0 or is_best:
                self.save_checkpoint({**train_metrics, **val_metrics}, is_best)
                
    def reconstruct_hidden_image(self, ms_masks: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct hidden image from multispectral masks.
        
        Args:
            ms_masks: Multispectral masks tensor of shape (B, C, H, W)
            
        Returns:
            Reconstructed hidden image tensor of shape (B, 3, H, W)
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(ms_masks.to(self.device))
            
    def visualize_reconstruction(
        self,
        rgb_img: torch.Tensor,
        ms_masks: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """
        Visualize original RGB image, multispectral data, and reconstructed hidden image.
        
        Args:
            rgb_img: RGB image tensor of shape (3, H, W)
            ms_masks: Multispectral masks tensor of shape (8, H, W)
            save_path: Optional path to save the visualization
        """
        # Ensure inputs are on CPU and convert to numpy
        rgb_img = rgb_img.cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize RGB image if needed
        rgb_img = np.clip(rgb_img, 0, 1)
        
        # Reconstruct hidden image
        ms_masks_batch = ms_masks.unsqueeze(0)  # Add batch dimension
        reconstructed = self.reconstruct_hidden_image(ms_masks_batch)
        reconstructed = reconstructed.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot RGB image
        axes[0].imshow(rgb_img)
        axes[0].set_title('Original RGB Image')
        axes[0].axis('off')
        
        # Plot one channel of multispectral data (e.g., infrared)
        ms_channel = ms_masks[0].cpu().numpy()
        axes[1].imshow(ms_channel, cmap='inferno')
        axes[1].set_title('Multispectral Data (IR Channel)')
        axes[1].axis('off')
        
        # Plot reconstructed hidden image
        axes[2].imshow(reconstructed)
        axes[2].set_title('Reconstructed Hidden Image')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
            
        return fig


def main():
    """Main function to demonstrate hidden image reconstruction."""
    import argparse
    from torch.utils.data import DataLoader
    from data.preprocessing.multispectral_dataset import MultispectralDataset
    
    parser = argparse.ArgumentParser(description='Train hidden image reconstructor')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
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
    model = HiddenImageReconstructor().to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create trainer
    trainer = HiddenImageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir
    )
    
    # Train model
    trainer.train(num_epochs=args.epochs)
    
    # Visualize example
    sample = next(iter(val_loader))
    rgb_img = sample[0][0]  # First image in batch
    ms_masks = sample[1][0]  # First mask in batch
    
    trainer.visualize_reconstruction(
        rgb_img=rgb_img,
        ms_masks=ms_masks,
        save_path=Path(args.save_dir) / 'reconstruction_example.png'
    )
    
    logger.info("Training and visualization complete!")


if __name__ == '__main__':
    main()
