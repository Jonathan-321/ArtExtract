#!/usr/bin/env python3
"""
Mock enhanced visualization script for ArtExtract.
This script creates mock visualizations to demonstrate the capabilities of the enhanced visualization tools
without requiring trained models.
"""

import argparse
import torch
import numpy as np
import logging
from pathlib import Path
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.preprocessing.multispectral_dataset import MultispectralDataset
from evaluation.enhanced_visualization import EnhancedVisualization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define band wavelengths (nm) for visualization
BAND_WAVELENGTHS = [400, 450, 500, 550, 600, 650, 700, 800]

def generate_mock_properties():
    """Generate mock painting properties for visualization."""
    # Define possible values for each property
    pigment_types = ['lead-based', 'oil-based', 'acrylic', 'tempera', 'other']
    damage_types = ['cracking', 'water-damage', 'fading', 'none']
    restoration_types = ['recent', 'historical', 'none']
    hidden_content_types = ['underdrawing', 'pentimento', 'earlier-painting', 'none']
    
    # Randomly select one value for each property
    properties = {
        'pigment_type': random.choice(pigment_types),
        'damage_type': random.choice(damage_types),
        'restoration': random.choice(restoration_types),
        'hidden_content': random.choice(hidden_content_types)
    }
    
    # Generate random confidence scores
    confidence_scores = {
        'pigment_type': random.uniform(0.7, 0.95),
        'damage_type': random.uniform(0.7, 0.95),
        'restoration': random.uniform(0.7, 0.95),
        'hidden_content': random.uniform(0.7, 0.95)
    }
    
    return properties, confidence_scores

def generate_mock_hidden_image(ms_masks):
    """Generate a mock hidden image reconstruction based on multispectral data."""
    # Use PCA to create a mock reconstruction from the multispectral data
    # This simulates how a model might extract features from multispectral data
    
    # Convert tensor to numpy if needed
    if isinstance(ms_masks, torch.Tensor):
        ms_np = ms_masks.cpu().numpy()
    else:
        ms_np = ms_masks
    
    # Reshape for PCA
    h, w = ms_np.shape[1], ms_np.shape[2]
    ms_reshaped = ms_np.reshape(ms_np.shape[0], -1).T  # (H*W, bands)
    
    # Apply PCA to get 3 components (RGB)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    rgb_pca = pca.fit_transform(ms_reshaped)
    
    # Reshape back to image
    rgb_pca = rgb_pca.reshape(h, w, 3)
    
    # Normalize to [0, 1]
    rgb_pca = (rgb_pca - rgb_pca.min()) / (rgb_pca.max() - rgb_pca.min() + 1e-8)
    
    # Add some noise to simulate reconstruction artifacts
    noise = np.random.normal(0, 0.05, rgb_pca.shape)
    rgb_pca = np.clip(rgb_pca + noise, 0, 1)
    
    # Convert to tensor
    if isinstance(ms_masks, torch.Tensor):
        # Convert to tensor with same device as input
        rgb_pca = torch.tensor(rgb_pca, dtype=ms_masks.dtype, device=ms_masks.device)
        # Convert from HWC to CHW
        rgb_pca = rgb_pca.permute(2, 0, 1)
    
    return rgb_pca

def visualize_sample(rgb_img, ms_masks, output_dir, sample_idx):
    """Create mock enhanced visualizations for a sample."""
    # Generate mock properties and hidden image
    properties, confidence_scores = generate_mock_properties()
    reconstructed_img = generate_mock_hidden_image(ms_masks)
    
    logger.info(f"Mock properties: {properties}")
    
    # Create enhanced visualizer
    visualizer = EnhancedVisualization(save_dir=output_dir)
    visualizer.band_wavelengths = BAND_WAVELENGTHS
    
    # Create enhanced hidden content visualization
    logger.info("Creating enhanced hidden content visualization...")
    visualizer.visualize_hidden_content_enhanced(
        rgb_img=rgb_img,
        ms_masks=ms_masks,
        reconstructed_img=reconstructed_img,
        title="Enhanced Hidden Content Detection",
        save_path=output_dir / f'enhanced_hidden_content_{sample_idx}.png'
    )
    
    # Create enhanced property visualization
    logger.info("Creating enhanced property visualization...")
    visualizer.visualize_property_detection_enhanced(
        rgb_img=rgb_img,
        properties=properties,
        confidence_scores=confidence_scores,
        title="Enhanced Painting Property Detection",
        save_path=output_dir / f'enhanced_properties_{sample_idx}.png'
    )
    
    # Extract hidden features
    logger.info("Extracting hidden features...")
    visualizer.extract_hidden_features(
        rgb_img=rgb_img,
        ms_masks=ms_masks,
        threshold=0.6,  # Adjust threshold as needed
        title="Hidden Feature Extraction",
        save_path=output_dir / f'hidden_features_{sample_idx}.png'
    )
    
    return properties, reconstructed_img

def main():
    """Main function to create mock enhanced visualizations."""
    parser = argparse.ArgumentParser(description='Create mock enhanced visualizations')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./mock_visualizations', help='Path to output directory')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    data_dir = Path(args.data_dir)
    dataset = MultispectralDataset(
        data_dir=data_dir,
        split='val',  # Use validation set for visualization
        transform=None  # Will use default transforms
    )
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    
    # Select samples to visualize
    num_samples = min(args.num_samples, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    # Visualize samples
    for i, idx in enumerate(tqdm(sample_indices, desc="Visualizing samples")):
        # Get sample - dataset returns (rgb_img, ms_masks) tuple
        rgb_img, ms_masks = dataset[idx]
        
        # Log sample info
        logger.info(f"Processing sample {i+1}/{num_samples}")
        logger.info(f"Sample: {dataset.rgb_files[idx].name}")
        logger.info(f"RGB image shape: {rgb_img.shape}")
        logger.info(f"MS masks shape: {ms_masks.shape}")
        
        # Visualize sample
        visualize_sample(
            rgb_img=rgb_img,
            ms_masks=ms_masks,
            output_dir=output_dir,
            sample_idx=i
        )
    
    logger.info(f"Mock enhanced visualization complete! Results saved to {output_dir}")

if __name__ == '__main__':
    main()
