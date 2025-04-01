"""
Additional visualization tools for the ArtExtract project.
This module extends the visualization capabilities for multispectral data and model predictions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
from PIL import Image
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedVisualization:
    """
    Advanced visualization tools for multispectral data and model predictions.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Optional directory to save visualizations
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_hidden_content(
        self,
        rgb_img: torch.Tensor,
        ms_masks: torch.Tensor,
        reconstructed_img: torch.Tensor,
        title: str = 'Hidden Content Detection',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5)
    ):
        """
        Visualize original RGB image, key multispectral bands, and reconstructed hidden image.
        
        Args:
            rgb_img: RGB image tensor of shape (3, H, W)
            ms_masks: Multispectral masks tensor of shape (8, H, W)
            reconstructed_img: Reconstructed hidden image tensor of shape (3, H, W)
            title: Plot title
            save_path: Optional path to save the visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Convert tensors to numpy
        if isinstance(rgb_img, torch.Tensor):
            rgb_img = rgb_img.cpu().numpy().transpose(1, 2, 0)
            rgb_img = np.clip(rgb_img, 0, 1)
            
        if isinstance(ms_masks, torch.Tensor):
            ms_masks = ms_masks.cpu().numpy()
            
        if isinstance(reconstructed_img, torch.Tensor):
            reconstructed_img = reconstructed_img.cpu().numpy().transpose(1, 2, 0)
            reconstructed_img = np.clip(reconstructed_img, 0, 1)
            
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot RGB image
        axes[0].imshow(rgb_img)
        axes[0].set_title('Original RGB Image')
        axes[0].axis('off')
        
        # Plot IR band (often reveals hidden content)
        axes[1].imshow(ms_masks[7], cmap='inferno')
        axes[1].set_title('IR Band (800nm)')
        axes[1].axis('off')
        
        # Plot reconstructed hidden image
        axes[2].imshow(reconstructed_img)
        axes[2].set_title('Reconstructed Hidden Image')
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Hidden content visualization saved to {save_path}")
            
        return fig
    
    def visualize_property_detection(
        self,
        rgb_img: torch.Tensor,
        properties: Dict[str, str],
        confidence_scores: Optional[Dict[str, float]] = None,
        title: str = 'Painting Property Detection',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Visualize detected painting properties.
        
        Args:
            rgb_img: RGB image tensor of shape (3, H, W)
            properties: Dictionary mapping property names to detected values
            confidence_scores: Optional dictionary mapping property names to confidence scores
            title: Plot title
            save_path: Optional path to save the visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Convert tensor to numpy
        if isinstance(rgb_img, torch.Tensor):
            rgb_img = rgb_img.cpu().numpy().transpose(1, 2, 0)
            rgb_img = np.clip(rgb_img, 0, 1)
            
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 1]})
        
        # Plot RGB image
        ax1.imshow(rgb_img)
        ax1.set_title('Painting')
        ax1.axis('off')
        
        # Plot properties as text
        ax2.axis('off')
        ax2.text(0.05, 0.95, 'Detected Properties:', fontsize=14, fontweight='bold', 
                 transform=ax2.transAxes, va='top')
        
        y_pos = 0.85
        for i, (prop, value) in enumerate(properties.items()):
            prop_name = prop.replace('_', ' ').title()
            
            # Add confidence score if available
            if confidence_scores and prop in confidence_scores:
                conf = confidence_scores[prop]
                text = f"{prop_name}: {value} (Confidence: {conf:.2f})"
            else:
                text = f"{prop_name}: {value}"
                
            ax2.text(0.05, y_pos, text, fontsize=12, transform=ax2.transAxes)
            y_pos -= 0.1
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Property detection visualization saved to {save_path}")
            
        return fig
    
    def visualize_feature_embedding(
        self,
        features: torch.Tensor,
        labels: List[str],
        method: str = 'tsne',
        title: str = 'Feature Embedding',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Visualize feature embeddings using dimensionality reduction.
        
        Args:
            features: Feature tensor of shape (N, D) where N is the number of samples
                     and D is the feature dimension
            labels: List of labels for each sample
            method: Dimensionality reduction method ('pca' or 'tsne')
            title: Plot title
            save_path: Optional path to save the visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Convert tensor to numpy
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
            
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")
            
        # Reduce dimensionality
        reduced_features = reducer.fit_transform(features)
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'x': reduced_features[:, 0],
            'y': reduced_features[:, 1],
            'label': labels
        })
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot embeddings
        sns.scatterplot(
            data=df,
            x='x',
            y='y',
            hue='label',
            palette='viridis',
            alpha=0.8,
            s=100,
            ax=ax
        )
        
        # Add labels
        for i, row in df.iterrows():
            ax.text(row['x'] + 0.02, row['y'] + 0.02, str(i), fontsize=9)
            
        ax.set_title(f'{title} ({method.upper()})', fontsize=16)
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature embedding visualization saved to {save_path}")
            
        return fig
    
    def create_composite_visualization(
        self,
        rgb_img: torch.Tensor,
        ms_masks: torch.Tensor,
        reconstructed_img: torch.Tensor,
        properties: Dict[str, str],
        title: str = 'Comprehensive Painting Analysis',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 12)
    ):
        """
        Create a comprehensive visualization combining multiple analysis results.
        
        Args:
            rgb_img: RGB image tensor of shape (3, H, W)
            ms_masks: Multispectral masks tensor of shape (8, H, W)
            reconstructed_img: Reconstructed hidden image tensor of shape (3, H, W)
            properties: Dictionary mapping property names to detected values
            title: Plot title
            save_path: Optional path to save the visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Convert tensors to numpy
        if isinstance(rgb_img, torch.Tensor):
            rgb_img = rgb_img.cpu().numpy().transpose(1, 2, 0)
            rgb_img = np.clip(rgb_img, 0, 1)
            
        if isinstance(ms_masks, torch.Tensor):
            ms_masks = ms_masks.cpu().numpy()
            
        if isinstance(reconstructed_img, torch.Tensor):
            reconstructed_img = reconstructed_img.cpu().numpy().transpose(1, 2, 0)
            reconstructed_img = np.clip(reconstructed_img, 0, 1)
            
        # Create figure with complex layout
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 3)
        
        # Original RGB image
        ax_rgb = plt.subplot(gs[0, 0])
        ax_rgb.imshow(rgb_img)
        ax_rgb.set_title('Original RGB Image')
        ax_rgb.axis('off')
        
        # Key spectral bands
        for i, band_idx in enumerate([0, 4, 7]):  # Blue, Red, IR
            ax = plt.subplot(gs[0, i+1] if i < 2 else gs[1, 0])
            im = ax.imshow(ms_masks[band_idx], cmap='inferno')
            ax.set_title(f'Band {band_idx} ({self.band_wavelengths[band_idx]}nm)')
            ax.axis('off')
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            
        # Band difference (IR - Blue)
        ax_diff = plt.subplot(gs[1, 1])
        diff = ms_masks[7] - ms_masks[0]  # IR - Blue
        vmin, vmax = np.percentile(diff, [2, 98])
        im = ax_diff.imshow(diff, cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax_diff.set_title('IR - Blue (Hidden Content)')
        ax_diff.axis('off')
        
        # Add colorbar
        divider = make_axes_locatable(ax_diff)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # Reconstructed hidden image
        ax_recon = plt.subplot(gs[1, 2])
        ax_recon.imshow(reconstructed_img)
        ax_recon.set_title('Reconstructed Hidden Image')
        ax_recon.axis('off')
        
        # Properties
        ax_props = plt.subplot(gs[2, :])
        ax_props.axis('off')
        
        # Create a table for properties
        prop_data = []
        headers = ['Property', 'Value']
        
        for prop, value in properties.items():
            prop_name = prop.replace('_', ' ').title()
            prop_data.append([prop_name, value])
            
        table = ax_props.table(
            cellText=prop_data,
            colLabels=headers,
            loc='center',
            cellLoc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        # Set title for properties section
        ax_props.set_title('Detected Properties', pad=20)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Composite visualization saved to {save_path}")
            
        return fig


def main():
    """Main function to demonstrate visualization tools."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demonstrate visualization tools')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./visualizations', help='Output directory')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to visualize')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load a sample
    from data.preprocessing.multispectral_dataset import MultispectralDataset
    dataset = MultispectralDataset(args.data_dir, split='val')
    
    if args.sample_idx >= len(dataset):
        logger.error(f"Sample index {args.sample_idx} out of range (dataset has {len(dataset)} samples)")
        return
        
    rgb_img, ms_masks, filename = dataset[args.sample_idx]
    
    # Create visualizer
    visualizer = AdvancedVisualization(save_dir=output_dir)
    
    # Create dummy reconstructed image and properties for demonstration
    reconstructed_img = torch.zeros_like(rgb_img)  # Dummy
    properties = {
        'pigment_type': 'lead-based',
        'damage_type': 'craquelure',
        'restoration': 'historical',
        'hidden_content': 'underdrawing'
    }
    
    # Create visualizations
    visualizer.create_composite_visualization(
        rgb_img=rgb_img,
        ms_masks=ms_masks,
        reconstructed_img=reconstructed_img,
        properties=properties,
        save_path=output_dir / f'composite_{args.sample_idx}.png'
    )
    
    logger.info("Visualization complete!")


if __name__ == '__main__':
    main()
