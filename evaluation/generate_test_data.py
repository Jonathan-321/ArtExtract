"""
Script to generate mock data and model checkpoints for testing.
This script creates sample data and model checkpoints for testing the visualization scripts.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
import shutil
import random
from PIL import Image, ImageDraw
import logging

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from models.style_classification.cnn_rnn_model import CNNRNNModel
from models.similarity_detection.similarity_model import PaintingSimilaritySystem
from models.multispectral.hidden_image_reconstruction import HiddenImageReconstructor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_random_image(size=(224, 224), mode='RGB'):
    """Generate a random image for testing."""
    if mode == 'RGB':
        # Create a random colored image
        img = Image.new(mode, size, color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ))
        
        # Add some random shapes
        draw = ImageDraw.Draw(img)
        
        # Add random rectangles
        for _ in range(random.randint(1, 5)):
            x1 = random.randint(0, size[0])
            y1 = random.randint(0, size[1])
            x2 = random.randint(0, size[0])
            y2 = random.randint(0, size[1])
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            draw.rectangle([x1, y1, x2, y2], fill=color)
        
        # Add random ellipses
        for _ in range(random.randint(1, 3)):
            x1 = random.randint(0, size[0])
            y1 = random.randint(0, size[1])
            x2 = random.randint(0, size[0])
            y2 = random.randint(0, size[1])
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            draw.ellipse([x1, y1, x2, y2], fill=color)
            
        return img
    else:
        # For multispectral data, create a random numpy array
        return np.random.rand(*size, 8)  # 8 spectral bands


def generate_classification_data(output_dir, num_samples=15):
    """Generate classification model and test data."""
    logger.info("Generating classification model and test data...")
    
    # Create directories
    model_dir = Path(output_dir) / 'models' / 'classification'
    data_dir = Path(output_dir) / 'data' / 'classification'
    
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple classification model
    num_classes = 10
    model = CNNRNNModel(
        num_classes=num_classes,
        cnn_backbone='resnet18',  # Use a smaller model for faster generation
        hidden_size=128,
        dropout=0.5
    )
    
    # Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'args': {
            'num_classes': num_classes,
            'cnn_backbone': 'resnet18',
            'hidden_size': 128,
            'dropout': 0.5
        }
    }
    
    torch.save(checkpoint, model_dir / 'classification_model.pth')
    logger.info(f"Classification model saved to {model_dir / 'classification_model.pth'}")
    
    # Generate test images
    for i in range(num_samples):
        img = generate_random_image()
        img.save(data_dir / f"test_image_{i+1}.jpg")
    
    logger.info(f"Generated {num_samples} test images in {data_dir}")
    
    return str(model_dir / 'classification_model.pth'), str(data_dir)


def generate_similarity_data(output_dir, num_samples=15):
    """Generate similarity model and test data."""
    logger.info("Generating similarity model and test data...")
    
    # Create directories
    model_dir = Path(output_dir) / 'models' / 'similarity'
    data_dir = Path(output_dir) / 'data' / 'similarity'
    
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple similarity model
    similarity_system = PaintingSimilaritySystem(
        feature_extractor_type='resnet18',  # Use a smaller model for faster generation
        feature_dim=512,
        similarity_metric='cosine'
    )
    
    # Save model checkpoint
    checkpoint = {
        'model_state_dict': similarity_system.feature_extractor.state_dict(),
        'feature_extractor_type': 'resnet18',
        'feature_dim': 512,
        'similarity_metric': 'cosine'
    }
    
    torch.save(checkpoint, model_dir / 'similarity_model.pth')
    logger.info(f"Similarity model saved to {model_dir / 'similarity_model.pth'}")
    
    # Generate test images (create some similar images)
    base_images = []
    for i in range(5):  # Create 5 base images
        img = generate_random_image()
        img.save(data_dir / f"base_image_{i+1}.jpg")
        base_images.append(img)
    
    # Create variations of base images
    for i, base_img in enumerate(base_images):
        # Create 2 variations of each base image
        for j in range(2):
            # Copy the base image
            variation = base_img.copy()
            
            # Add some random modifications
            draw = ImageDraw.Draw(variation)
            
            # Add a few random shapes
            for _ in range(random.randint(1, 3)):
                x1 = random.randint(0, 224)
                y1 = random.randint(0, 224)
                x2 = random.randint(0, 224)
                y2 = random.randint(0, 224)
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
                draw.rectangle([x1, y1, x2, y2], fill=color)
            
            variation.save(data_dir / f"variation_{i+1}_{j+1}.jpg")
    
    logger.info(f"Generated {len(base_images)} base images and {len(base_images)*2} variations in {data_dir}")
    
    return str(model_dir / 'similarity_model.pth'), str(data_dir)


def generate_multispectral_data(output_dir, num_samples=5):
    """Generate multispectral model and test data."""
    logger.info("Generating multispectral model and test data...")
    
    # Create directories
    model_dir = Path(output_dir) / 'models' / 'multispectral'
    data_dir = Path(output_dir) / 'data' / 'multispectral'
    
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple multispectral model
    model = HiddenImageReconstructor(
        in_channels=8,
        out_channels=3,
        features=[32, 64, 128, 256]  # Smaller features for faster generation
    )
    
    # Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'in_channels': 8,
        'out_channels': 3
    }
    
    torch.save(checkpoint, model_dir / 'multispectral_model.pth')
    logger.info(f"Multispectral model saved to {model_dir / 'multispectral_model.pth'}")
    
    # Generate test multispectral data
    for i in range(num_samples):
        # Generate random multispectral data (8 channels)
        ms_data = np.random.rand(8, 224, 224).astype(np.float32)
        
        # Save as numpy array
        np.save(data_dir / f"ms_data_{i+1}.npy", ms_data)
    
    logger.info(f"Generated {num_samples} multispectral data samples in {data_dir}")
    
    return str(model_dir / 'multispectral_model.pth'), str(data_dir)


def main():
    """Main function."""
    # Create output directory
    output_dir = Path('./test_data')
    output_dir.mkdir(exist_ok=True)
    
    # Generate test data and models
    classification_model, classification_data = generate_classification_data(output_dir)
    similarity_model, similarity_data = generate_similarity_data(output_dir)
    multispectral_model, multispectral_data = generate_multispectral_data(output_dir)
    
    # Save configuration for test_and_visualize.py
    config = {
        'classification_model_path': classification_model,
        'classification_data_dir': classification_data,
        'similarity_model_path': similarity_model,
        'similarity_data_dir': similarity_data,
        'multispectral_model_path': multispectral_model,
        'multispectral_data_dir': multispectral_data,
        'output_dir': str(output_dir / 'visualization_results')
    }
    
    with open(output_dir / 'test_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Test configuration saved to {output_dir / 'test_config.json'}")
    
    print("\n" + "="*80)
    print("TEST DATA GENERATION COMPLETE")
    print("="*80)
    print("The following test data and models have been generated:")
    print(f"Classification model: {classification_model}")
    print(f"Classification data: {classification_data}")
    print(f"Similarity model: {similarity_model}")
    print(f"Similarity data: {similarity_data}")
    print(f"Multispectral model: {multispectral_model}")
    print(f"Multispectral data: {multispectral_data}")
    print("\nTo run the test and visualization script, use:")
    print(f"python evaluation/test_and_visualize.py --classification_model_path {classification_model} \\")
    print(f"                                      --classification_data_dir {classification_data} \\")
    print(f"                                      --similarity_model_path {similarity_model} \\")
    print(f"                                      --similarity_data_dir {similarity_data} \\")
    print(f"                                      --multispectral_model_path {multispectral_model} \\")
    print(f"                                      --multispectral_data_dir {multispectral_data}")
    print("="*80)


if __name__ == "__main__":
    main()
