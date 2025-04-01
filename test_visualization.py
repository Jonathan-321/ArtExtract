"""
Test script for visualization tools in ArtExtract.
This script demonstrates how to use the visualization tools with existing data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.insert(0, '.')

# Import the visualization tools
from evaluation.visualization_tools_part2 import AdvancedVisualization
from data.preprocessing.multispectral_dataset import MultispectralDataset

# Define wavelengths for multispectral bands (in nm)
BAND_WAVELENGTHS = [400, 450, 500, 550, 600, 650, 700, 750]

def main():
    # Create output directory
    output_dir = Path('./visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load a sample from the multispectral dataset
    data_dir = Path('./data/multispectral')
    dataset = MultispectralDataset(data_dir, split='val')
    
    if len(dataset) == 0:
        print(f"Error: No samples found in dataset")
        return
        
    # Get the first sample
    sample_idx = 0
    rgb_img, ms_masks, filename = dataset[sample_idx]
    
    print(f"Loaded sample: {filename}")
    print(f"RGB image shape: {rgb_img.shape}")
    print(f"MS masks shape: {ms_masks.shape}")
    
    # Create visualizer with band wavelengths
    visualizer = AdvancedVisualization(save_dir=output_dir)
    # Add band wavelengths attribute
    visualizer.band_wavelengths = BAND_WAVELENGTHS
    
    # Create dummy reconstructed image and properties for demonstration
    reconstructed_img = torch.zeros_like(rgb_img)  # Dummy
    properties = {
        'pigment_type': 'lead-based',
        'damage_type': 'craquelure',
        'restoration': 'historical',
        'hidden_content': 'underdrawing'
    }
    
    # Test the visualize_hidden_content method (we know this exists)
    print("Creating hidden content visualization...")
    visualizer.visualize_hidden_content(
        rgb_img=rgb_img,
        ms_masks=ms_masks,
        reconstructed_img=reconstructed_img,
        title="Hidden Content Detection",
        save_path=output_dir / f'hidden_content_{sample_idx}.png'
    )
    
    # Create a simple property visualization using matplotlib directly
    print("Creating simple property visualization...")
    # Convert tensor to numpy if needed
    if isinstance(rgb_img, torch.Tensor):
        rgb_img_np = rgb_img.cpu().numpy().transpose(1, 2, 0)
        rgb_img_np = np.clip(rgb_img_np, 0, 1)
    else:
        rgb_img_np = rgb_img
        
    # Create a simple figure with the image and properties
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(rgb_img_np)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Display properties
    ax2.axis('off')
    ax2.text(0.05, 0.95, 'Detected Properties:', fontsize=14, fontweight='bold', 
             va='top')
    
    y_pos = 0.85
    for prop, value in properties.items():
        prop_name = prop.replace('_', ' ').title()
        ax2.text(0.05, y_pos, f"{prop_name}: {value}", fontsize=12, va='top')
        y_pos -= 0.1
        
    plt.tight_layout()
    plt.savefig(output_dir / f'simple_properties_{sample_idx}.png')
    
    print("Visualization complete!")
    print(f"Output saved to: {output_dir.absolute()}")

if __name__ == '__main__':
    main()
