"""
Training script for ArtExtract multispectral models.
This script trains both the property detection and hidden image reconstruction models.
"""

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import models
from models.multispectral.property_detection import PaintingPropertyDetector, PropertyDetectionTrainer
from models.multispectral.hidden_image_reconstruction import HiddenImageReconstructor, HiddenImageTrainer
from data.preprocessing.multispectral_dataset import MultispectralDataset

def train_property_detector(args):
    """Train the painting property detector model."""
    logger.info("=== Training Painting Property Detector ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info(f"Loading data from {args.data_dir}")
    train_dataset = MultispectralDataset(args.data_dir, split='train')
    val_dataset = MultispectralDataset(args.data_dir, split='val')
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    logger.info("Creating PaintingPropertyDetector model")
    model = PaintingPropertyDetector(
        rgb_backbone='resnet18',
        ms_backbone='resnet18',
        pretrained=True,
        fusion_method='attention'
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create save directory
    property_save_dir = Path(args.save_dir) / 'property_detector'
    property_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = PropertyDetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_dir=str(property_save_dir)
    )
    
    # Train model
    logger.info(f"Training for {args.property_epochs} epochs")
    trainer.train(num_epochs=args.property_epochs)
    
    # Plot training history
    trainer.plot_training_history(save_path=property_save_dir / 'property_detection_history.png')
    
    # Visualize example
    sample = next(iter(val_loader))
    rgb_img = sample[0][0]  # First image in batch
    ms_masks = sample[1][0]  # First mask in batch
    
    trainer.visualize_property_predictions(
        rgb_img=rgb_img,
        ms_masks=ms_masks,
        save_path=property_save_dir / 'property_detection_example.png'
    )
    
    logger.info("Property detector training complete!")
    return model

def train_hidden_image_reconstructor(args):
    """Train the hidden image reconstructor model."""
    logger.info("=== Training Hidden Image Reconstructor ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info(f"Loading data from {args.data_dir}")
    train_dataset = MultispectralDataset(args.data_dir, split='train')
    val_dataset = MultispectralDataset(args.data_dir, split='val')
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    logger.info("Creating HiddenImageReconstructor model")
    model = HiddenImageReconstructor().to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create save directory
    reconstruction_save_dir = Path(args.save_dir) / 'hidden_reconstructor'
    reconstruction_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = HiddenImageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_dir=str(reconstruction_save_dir)
    )
    
    # Train model
    logger.info(f"Training for {args.reconstruction_epochs} epochs")
    trainer.train(num_epochs=args.reconstruction_epochs)
    
    # Visualize example
    sample = next(iter(val_loader))
    rgb_img = sample[0][0]  # First image in batch
    ms_masks = sample[1][0]  # First mask in batch
    
    trainer.visualize_reconstruction(
        rgb_img=rgb_img,
        ms_masks=ms_masks,
        save_path=reconstruction_save_dir / 'reconstruction_example.png'
    )
    
    logger.info("Hidden image reconstructor training complete!")
    return model

def main():
    """Main function to train both models."""
    parser = argparse.ArgumentParser(description='Train ArtExtract multispectral models')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--property_epochs', type=int, default=30, help='Number of epochs for property detector')
    parser.add_argument('--reconstruction_epochs', type=int, default=50, help='Number of epochs for hidden reconstructor')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--train_property', action='store_true', help='Train property detector')
    parser.add_argument('--train_reconstruction', action='store_true', help='Train hidden image reconstructor')
    args = parser.parse_args()
    
    # If no specific model is selected, train both
    if not args.train_property and not args.train_reconstruction:
        args.train_property = True
        args.train_reconstruction = True
    
    # Train models
    if args.train_property:
        property_model = train_property_detector(args)
    
    if args.train_reconstruction:
        reconstruction_model = train_hidden_image_reconstructor(args)
    
    logger.info("All training complete!")

if __name__ == '__main__':
    main()
