"""
Visualization tools for the ArtExtract project.
This module provides comprehensive visualization capabilities for multispectral data and model predictions.
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
import io
import base64
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultispectralVisualizer:
    """
    Visualization tools for multispectral data and model predictions.
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
            
        # Define spectral band wavelengths (in nm)
        self.band_wavelengths = {
            0: 450,  # Blue
            1: 500,  # Cyan
            2: 550,  # Green
            3: 600,  # Yellow
            4: 650,  # Red
            5: 700,  # Deep Red
            6: 750,  # Near IR
            7: 800   # IR
        }
        
        # Create a custom colormap for spectral visualization
        self.spectral_cmap = self._create_spectral_colormap()
        
    def _create_spectral_colormap(self) -> LinearSegmentedColormap:
        """Create a custom colormap for spectral visualization."""
        # Colors from visible spectrum (blue to red) plus infrared (purple)
        colors = [
            (0, 0, 1),      # 450nm - Blue
            (0, 1, 1),      # 500nm - Cyan
            (0, 1, 0),      # 550nm - Green
            (1, 1, 0),      # 600nm - Yellow
            (1, 0, 0),      # 650nm - Red
            (0.5, 0, 0),    # 700nm - Deep Red
            (0.5, 0, 0.5),  # 750nm - Near IR (represented as purple)
            (0.3, 0, 0.3)   # 800nm - IR (represented as dark purple)
        ]
        return LinearSegmentedColormap.from_list('spectral', colors)
        
    def visualize_spectral_bands(
        self,
        ms_masks: torch.Tensor,
        rgb_img: Optional[torch.Tensor] = None,
        title: str = 'Spectral Bands',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Visualize all spectral bands of a multispectral image.
        
        Args:
            ms_masks: Multispectral masks tensor of shape (8, H, W)
            rgb_img: Optional RGB image tensor of shape (3, H, W)
            title: Plot title
            save_path: Optional path to save the visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Convert tensors to numpy
        if isinstance(ms_masks, torch.Tensor):
            ms_masks = ms_masks.cpu().numpy()
            
        if rgb_img is not None and isinstance(rgb_img, torch.Tensor):
            rgb_img = rgb_img.cpu().numpy().transpose(1, 2, 0)
            rgb_img = np.clip(rgb_img, 0, 1)
            
        # Create figure
        if rgb_img is not None:
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(3, 3)
            
            # Plot RGB image
            ax_rgb = plt.subplot(gs[0, 0])
            ax_rgb.imshow(rgb_img)
            ax_rgb.set_title('RGB Image')
            ax_rgb.axis('off')
            
            # Plot spectral bands
            for i in range(8):
                row, col = (i // 3) + (1 if i < 3 else 0), i % 3
                ax = plt.subplot(gs[row, col])
                im = ax.imshow(ms_masks[i], cmap='inferno')
                ax.set_title(f'Band {i}: {self.band_wavelengths[i]}nm')
                ax.axis('off')
                
                # Add colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
        else:
            # Without RGB image, use a simpler layout
            fig, axes = plt.subplots(2, 4, figsize=figsize)
            axes = axes.flatten()
            
            for i, ax in enumerate(axes):
                im = ax.imshow(ms_masks[i], cmap='inferno')
                ax.set_title(f'Band {i}: {self.band_wavelengths[i]}nm')
                ax.axis('off')
                
                # Add colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Spectral bands visualization saved to {save_path}")
            
        return fig
        
    def visualize_band_differences(
        self,
        ms_masks: torch.Tensor,
        rgb_img: Optional[torch.Tensor] = None,
        band_pairs: Optional[List[Tuple[int, int]]] = None,
        title: str = 'Spectral Band Differences',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Visualize differences between spectral bands to highlight hidden features.
        
        Args:
            ms_masks: Multispectral masks tensor of shape (8, H, W)
            rgb_img: Optional RGB image tensor of shape (3, H, W)
            band_pairs: List of band index pairs to compare (default: key pairs for hidden content)
            title: Plot title
            save_path: Optional path to save the visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Convert tensors to numpy
        if isinstance(ms_masks, torch.Tensor):
            ms_masks = ms_masks.cpu().numpy()
            
        if rgb_img is not None and isinstance(rgb_img, torch.Tensor):
            rgb_img = rgb_img.cpu().numpy().transpose(1, 2, 0)
            rgb_img = np.clip(rgb_img, 0, 1)
            
        # Default band pairs that often reveal hidden content
        if band_pairs is None:
            band_pairs = [
                (0, 6),  # Blue vs Near IR
                (4, 7),  # Red vs IR
                (2, 6),  # Green vs Near IR
                (1, 5)   # Cyan vs Deep Red
            ]
            
        # Create figure
        if rgb_img is not None:
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(2, 3)
            
            # Plot RGB image
            ax_rgb = plt.subplot(gs[0, 0])
            ax_rgb.imshow(rgb_img)
            ax_rgb.set_title('RGB Image')
            ax_rgb.axis('off')
            
            # Plot band differences
            for i, (band1, band2) in enumerate(band_pairs[:5]):  # Limit to 5 pairs
                row, col = (i // 3) + (1 if i < 3 else 0), i % 3
                ax = plt.subplot(gs[row, col])
                
                # Compute normalized difference
                diff = ms_masks[band1] - ms_masks[band2]
                vmin, vmax = np.percentile(diff, [2, 98])  # Robust scaling
                
                im = ax.imshow(diff, cmap='coolwarm', vmin=vmin, vmax=vmax)
                ax.set_title(f'Band {band1} ({self.band_wavelengths[band1]}nm) - '
                           f'Band {band2} ({self.band_wavelengths[band2]}nm)')
                ax.axis('off')
                
                # Add colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
        else:
            # Without RGB image
            n_pairs = len(band_pairs)
            n_cols = min(3, n_pairs)
            n_rows = (n_pairs + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i, (band1, band2) in enumerate(band_pairs):
                if i < len(axes):
                    ax = axes[i]
                    
                    # Compute normalized difference
                    diff = ms_masks[band1] - ms_masks[band2]
                    vmin, vmax = np.percentile(diff, [2, 98])  # Robust scaling
                    
                    im = ax.imshow(diff, cmap='coolwarm', vmin=vmin, vmax=vmax)
                    ax.set_title(f'Band {band1} ({self.band_wavelengths[band1]}nm) - '
                               f'Band {band2} ({self.band_wavelengths[band2]}nm)')
                    ax.axis('off')
                    
                    # Add colorbar
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)
            
            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
                
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Band differences visualization saved to {save_path}")
            
        return fig
