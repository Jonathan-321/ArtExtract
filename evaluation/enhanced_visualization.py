"""
Enhanced visualization tools for the ArtExtract project.
This module provides advanced visualization techniques for multispectral data and hidden content extraction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import cv2
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedVisualization:
    """
    Advanced visualization tools for multispectral data and hidden content extraction.
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
            
        # Default band wavelengths (nm)
        self.band_wavelengths = [400, 450, 500, 550, 600, 650, 700, 800]
    
    def _tensor_to_numpy(self, tensor: torch.Tensor, is_image: bool = True) -> np.ndarray:
        """Convert tensor to numpy array with proper formatting for visualization."""
        if not isinstance(tensor, torch.Tensor):
            return tensor
            
        array = tensor.cpu().detach().numpy()
        
        if is_image and array.shape[0] in [1, 3]:  # CHW to HWC for images
            array = array.transpose(1, 2, 0)
            if array.shape[2] == 3:  # RGB image
                array = np.clip(array, 0, 1)
            elif array.shape[2] == 1:  # Grayscale image
                array = array.squeeze(2)
                
        return array
    
    def visualize_hidden_content_enhanced(
        self,
        rgb_img: torch.Tensor,
        ms_masks: torch.Tensor,
        reconstructed_img: torch.Tensor,
        title: str = 'Enhanced Hidden Content Detection',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Enhanced visualization of hidden content with difference maps and key spectral bands.
        
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
        rgb_img_np = self._tensor_to_numpy(rgb_img)
        ms_masks_np = self._tensor_to_numpy(ms_masks, is_image=False)
        reconstructed_img_np = self._tensor_to_numpy(reconstructed_img)
        
        # Create figure with grid layout
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1, 1])
        
        # Plot RGB image
        ax_rgb = fig.add_subplot(gs[0, 0])
        ax_rgb.imshow(rgb_img_np)
        ax_rgb.set_title('Original RGB Image')
        ax_rgb.axis('off')
        
        # Plot key multispectral bands
        # Select bands that typically reveal hidden content (IR and UV)
        ir_band_idx = -1  # Last band (typically IR)
        uv_band_idx = 0   # First band (typically UV/near-UV)
        
        # IR band visualization
        ax_ir = fig.add_subplot(gs[0, 1])
        ir_img = ms_masks_np[ir_band_idx]
        ax_ir.imshow(ir_img, cmap='inferno')
        ax_ir.set_title(f'IR Band ({self.band_wavelengths[ir_band_idx]}nm)')
        ax_ir.axis('off')
        
        # UV band visualization
        ax_uv = fig.add_subplot(gs[0, 2])
        uv_img = ms_masks_np[uv_band_idx]
        ax_uv.imshow(uv_img, cmap='viridis')
        ax_uv.set_title(f'UV Band ({self.band_wavelengths[uv_band_idx]}nm)')
        ax_uv.axis('off')
        
        # Reconstructed hidden image
        ax_recon = fig.add_subplot(gs[0, 3])
        ax_recon.imshow(reconstructed_img_np)
        ax_recon.set_title('Reconstructed Hidden Image')
        ax_recon.axis('off')
        
        # Create difference maps to highlight hidden content
        # 1. RGB to IR difference
        if len(rgb_img_np.shape) == 3 and rgb_img_np.shape[2] == 3:
            # Convert RGB to grayscale for comparison
            rgb_gray = np.mean(rgb_img_np, axis=2)
        else:
            rgb_gray = rgb_img_np
            
        # Normalize IR band for comparison
        ir_norm = (ir_img - ir_img.min()) / (ir_img.max() - ir_img.min() + 1e-8)
        
        # Calculate absolute difference
        diff_map = np.abs(rgb_gray - ir_norm)
        # Normalize difference map
        diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
        
        # Apply threshold to highlight significant differences
        threshold = 0.3
        diff_map_thresholded = np.copy(diff_map)
        diff_map_thresholded[diff_map < threshold] = 0
        
        # Plot difference map
        ax_diff = fig.add_subplot(gs[1, 0])
        ax_diff.imshow(diff_map, cmap='hot')
        ax_diff.set_title('Difference Map (RGB vs IR)')
        ax_diff.axis('off')
        
        # Plot thresholded difference map
        ax_diff_thresh = fig.add_subplot(gs[1, 1])
        ax_diff_thresh.imshow(diff_map_thresholded, cmap='hot')
        ax_diff_thresh.set_title(f'Thresholded Difference (>{threshold:.1f})')
        ax_diff_thresh.axis('off')
        
        # Create a composite visualization
        # Overlay thresholded difference on RGB image
        overlay = np.copy(rgb_img_np)
        if len(overlay.shape) == 2:
            overlay = np.stack([overlay] * 3, axis=2)
            
        # Create a red mask for the overlay
        red_mask = np.zeros_like(overlay)
        if len(red_mask.shape) == 3:
            red_mask[..., 0] = diff_map_thresholded  # Red channel
        
        # Blend the images
        alpha = 0.7
        composite = cv2.addWeighted(overlay, 1.0, red_mask, alpha, 0)
        
        # Plot composite
        ax_composite = fig.add_subplot(gs[1, 2])
        ax_composite.imshow(composite)
        ax_composite.set_title('Hidden Content Overlay')
        ax_composite.axis('off')
        
        # Plot PCA of multispectral bands to highlight differences
        # Reshape multispectral data for PCA
        h, w = ms_masks_np.shape[1], ms_masks_np.shape[2]
        ms_reshaped = ms_masks_np.reshape(ms_masks_np.shape[0], -1).T  # (H*W, bands)
        
        # Apply PCA
        pca = PCA(n_components=3)
        ms_pca = pca.fit_transform(ms_reshaped)
        
        # Reshape back to image
        ms_pca = ms_pca.reshape(h, w, 3)
        
        # Normalize for visualization
        ms_pca = (ms_pca - ms_pca.min()) / (ms_pca.max() - ms_pca.min() + 1e-8)
        
        # Plot PCA visualization
        ax_pca = fig.add_subplot(gs[1, 3])
        ax_pca.imshow(ms_pca)
        ax_pca.set_title('PCA of Multispectral Bands')
        ax_pca.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Enhanced hidden content visualization saved to {save_path}")
            
        return fig
    
    def visualize_property_detection_enhanced(
        self,
        rgb_img: torch.Tensor,
        properties: Dict[str, str],
        confidence_scores: Optional[Dict[str, float]] = None,
        title: str = 'Enhanced Property Detection',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Enhanced visualization of detected painting properties with visual indicators.
        
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
        rgb_img_np = self._tensor_to_numpy(rgb_img)
        
        # Create figure with grid layout
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1.5, 1])
        
        # Plot RGB image (larger)
        ax_rgb = fig.add_subplot(gs[:, 0])
        ax_rgb.imshow(rgb_img_np)
        ax_rgb.set_title('Painting', fontsize=14)
        ax_rgb.axis('off')
        
        # Plot properties as text with visual indicators
        ax_props = fig.add_subplot(gs[0, 1])
        ax_props.axis('off')
        ax_props.text(0.05, 0.95, 'Detected Properties:', fontsize=14, fontweight='bold', 
                      transform=ax_props.transAxes, va='top')
        
        # Define colors for different property types
        property_colors = {
            'pigment_type': '#1f77b4',  # blue
            'damage_type': '#d62728',   # red
            'restoration': '#2ca02c',   # green
            'hidden_content': '#9467bd' # purple
        }
        
        # Plot properties with colored boxes
        y_pos = 0.85
        for i, (prop, value) in enumerate(properties.items()):
            prop_name = prop.replace('_', ' ').title()
            color = property_colors.get(prop, '#7f7f7f')  # default gray
            
            # Add confidence score if available
            if confidence_scores and prop in confidence_scores:
                conf = confidence_scores[prop]
                conf_str = f" (Confidence: {conf:.2f})"
                # Adjust color opacity based on confidence
                alpha = max(0.3, conf)  # minimum opacity of 0.3
            else:
                conf_str = ""
                alpha = 0.7
            
            # Create colored box with property name
            box_text = f"{prop_name}: {value}{conf_str}"
            ax_props.text(0.05, y_pos, box_text, fontsize=12, transform=ax_props.transAxes,
                         bbox=dict(facecolor=color, alpha=alpha, pad=5, boxstyle='round'))
            y_pos -= 0.15
        
        # Create a visual summary of properties
        ax_summary = fig.add_subplot(gs[1, 1])
        ax_summary.axis('off')
        
        # Create a simple visual summary based on properties
        summary_text = []
        
        # Interpret pigment type
        if 'pigment_type' in properties:
            pigment = properties['pigment_type']
            if pigment == 'lead-based':
                summary_text.append("• Lead-based pigments suggest pre-20th century origin")
            elif pigment == 'oil-based':
                summary_text.append("• Oil-based pigments typical of classical paintings")
            elif pigment == 'acrylic':
                summary_text.append("• Acrylic pigments indicate modern artwork (post-1940s)")
            elif pigment == 'tempera':
                summary_text.append("• Tempera suggests medieval or early Renaissance work")
            else:
                summary_text.append(f"• Detected pigment: {pigment}")
        
        # Interpret damage type
        if 'damage_type' in properties:
            damage = properties['damage_type']
            if damage == 'cracking':
                summary_text.append("• Cracking indicates aging or environmental stress")
            elif damage == 'water-damage':
                summary_text.append("• Water damage suggests exposure to moisture")
            elif damage == 'fading':
                summary_text.append("• Fading indicates light exposure over time")
            elif damage == 'none':
                summary_text.append("• No significant damage detected")
            else:
                summary_text.append(f"• Detected damage: {damage}")
        
        # Interpret restoration
        if 'restoration' in properties:
            restoration = properties['restoration']
            if restoration == 'recent':
                summary_text.append("• Recent restoration work detected")
            elif restoration == 'historical':
                summary_text.append("• Historical restoration techniques present")
            elif restoration == 'none':
                summary_text.append("• No restoration detected, likely original state")
            else:
                summary_text.append(f"• Restoration status: {restoration}")
        
        # Interpret hidden content
        if 'hidden_content' in properties:
            hidden = properties['hidden_content']
            if hidden == 'underdrawing':
                summary_text.append("• Underdrawing detected beneath visible layers")
            elif hidden == 'pentimento':
                summary_text.append("• Pentimento shows artist's composition changes")
            elif hidden == 'earlier-painting':
                summary_text.append("• Earlier painting detected beneath current work")
            elif hidden == 'none':
                summary_text.append("• No hidden content detected")
            else:
                summary_text.append(f"• Hidden content: {hidden}")
        
        # Display summary
        ax_summary.text(0.05, 0.95, 'Analysis Summary:', fontsize=14, fontweight='bold',
                       transform=ax_summary.transAxes, va='top')
        
        y_pos = 0.85
        for line in summary_text:
            ax_summary.text(0.05, y_pos, line, fontsize=11, transform=ax_summary.transAxes)
            y_pos -= 0.12
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Enhanced property detection visualization saved to {save_path}")
            
        return fig
    
    def extract_hidden_features(
        self,
        rgb_img: torch.Tensor,
        ms_masks: torch.Tensor,
        threshold: float = 0.7,
        title: str = 'Hidden Feature Extraction',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Extract and visualize hidden features by analyzing differences across multispectral bands.
        
        Args:
            rgb_img: RGB image tensor of shape (3, H, W)
            ms_masks: Multispectral masks tensor of shape (8, H, W)
            threshold: Threshold for feature extraction (0.0-1.0)
            title: Plot title
            save_path: Optional path to save the visualization
            figsize: Figure size
            
        Returns:
            Tuple of (matplotlib figure, extracted features mask)
        """
        # Convert tensors to numpy
        rgb_img_np = self._tensor_to_numpy(rgb_img)
        ms_masks_np = self._tensor_to_numpy(ms_masks, is_image=False)
        
        # Calculate variance across spectral bands for each pixel
        spectral_variance = np.var(ms_masks_np, axis=0)
        
        # Normalize variance map
        variance_norm = (spectral_variance - spectral_variance.min()) / (spectral_variance.max() - spectral_variance.min() + 1e-8)
        
        # Apply threshold to create binary mask of hidden features
        feature_mask = np.copy(variance_norm)
        feature_mask[variance_norm < threshold] = 0
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # Plot RGB image
        ax_rgb = fig.add_subplot(gs[0, 0])
        ax_rgb.imshow(rgb_img_np)
        ax_rgb.set_title('Original RGB Image')
        ax_rgb.axis('off')
        
        # Plot variance map
        ax_var = fig.add_subplot(gs[0, 1])
        var_plot = ax_var.imshow(variance_norm, cmap='viridis')
        ax_var.set_title('Spectral Variance Map')
        ax_var.axis('off')
        
        # Add colorbar
        plt.colorbar(var_plot, ax=ax_var, fraction=0.046, pad=0.04)
        
        # Plot thresholded feature mask
        ax_mask = fig.add_subplot(gs[0, 2])
        ax_mask.imshow(feature_mask, cmap='hot')
        ax_mask.set_title(f'Feature Mask (Threshold: {threshold:.2f})')
        ax_mask.axis('off')
        
        # Create overlay of features on RGB image
        overlay = np.copy(rgb_img_np)
        if len(overlay.shape) == 2:
            overlay = np.stack([overlay] * 3, axis=2)
            
        # Create a highlight mask for the overlay (yellow)
        highlight_mask = np.zeros_like(overlay)
        if len(highlight_mask.shape) == 3:
            highlight_mask[..., 0] = feature_mask  # Red channel
            highlight_mask[..., 1] = feature_mask  # Green channel
        
        # Blend the images
        alpha = 0.7
        composite = cv2.addWeighted(overlay, 1.0, highlight_mask, alpha, 0)
        
        # Plot composite
        ax_composite = fig.add_subplot(gs[1, :])
        ax_composite.imshow(composite)
        ax_composite.set_title('Hidden Features Highlighted')
        ax_composite.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Hidden feature extraction saved to {save_path}")
            
        return fig, feature_mask
