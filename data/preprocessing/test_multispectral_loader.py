"""
Test script for the MultispectralDataset loader.
"""

import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from multispectral_dataset import MultispectralDataset, create_dataloaders

def denormalize(tensor):
    """Denormalize an RGB image tensor."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def visualize_sample(rgb_image, ms_masks, save_path=None):
    """
    Visualize an RGB image and its MS masks.
    
    Args:
        rgb_image (torch.Tensor): RGB image tensor (3, H, W)
        ms_masks (torch.Tensor): MS masks tensor (8, H, W)
        save_path (str): Optional path to save the visualization
    """
    # Create a figure with 3 rows: 1 for RGB, 2 for MS masks (4 each)
    fig = plt.figure(figsize=(20, 15))
    
    # Plot RGB image
    ax = plt.subplot(3, 1, 1)
    rgb_image = denormalize(rgb_image)
    plt.imshow(rgb_image.permute(1, 2, 0).clip(0, 1))
    ax.set_title('RGB Image')
    ax.axis('off')
    
    # Plot MS masks in a grid
    for i in range(8):
        row = 2 if i < 4 else 3
        col = i % 4 + 1
        ax = plt.subplot(3, 4, (row - 1) * 4 + col)
        plt.imshow(ms_masks[i].squeeze(), cmap='viridis')
        ax.set_title(f'MS Mask {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Create output directory for visualizations
    os.makedirs('data/test_output', exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir='data/multispectral',
        batch_size=4,
        num_workers=0  # Use 0 for debugging
    )
    
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    
    # Get a batch from training set
    rgb_images, ms_masks = next(iter(train_loader))
    
    print("\nShapes:")
    print(f"RGB images batch shape: {rgb_images.shape}")
    print(f"MS masks batch shape: {ms_masks.shape}")
    
    # Visualize each sample in the batch
    for i in range(rgb_images.shape[0]):
        save_path = f'data/test_output/sample_{i}.png'
        visualize_sample(
            rgb_images[i],
            ms_masks[i],
            save_path=save_path
        )
        print(f"Saved visualization to {save_path}")

if __name__ == '__main__':
    main()
