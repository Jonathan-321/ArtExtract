"""
Visualization script for trained ArtExtract multispectral models.
This script loads trained models and creates accurate visualizations.
"""

import argparse
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import sys
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import models and visualization tools
from models.multispectral.property_detection import PaintingPropertyDetector
from models.multispectral.hidden_image_reconstruction import HiddenImageReconstructor
from evaluation.visualization_tools_part2 import AdvancedVisualization
from data.preprocessing.multispectral_dataset import MultispectralDataset

# Define wavelengths for multispectral bands (in nm)
BAND_WAVELENGTHS = [400, 450, 500, 550, 600, 650, 700, 750]

def load_models(args):
    """Load trained models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load property detector if available
    property_detector = None
    property_checkpoint_path = Path(args.checkpoint_dir) / 'property_detector' / 'best_property_detector_model.pth'
    if property_checkpoint_path.exists():
        logger.info(f"Loading property detector from {property_checkpoint_path}")
        property_detector = PaintingPropertyDetector(
            rgb_backbone='resnet18',
            ms_backbone='resnet18',
            pretrained=False,
            fusion_method='attention'
        ).to(device)
        
        checkpoint = torch.load(property_checkpoint_path, map_location=device)
        property_detector.load_state_dict(checkpoint['model_state_dict'])
        property_detector.eval()
        logger.info("Property detector loaded successfully")
    else:
        logger.warning(f"No property detector checkpoint found at {property_checkpoint_path}")
    
    # Load hidden image reconstructor if available
    hidden_reconstructor = None
    reconstructor_checkpoint_path = Path(args.checkpoint_dir) / 'hidden_reconstructor' / 'best_reconstructor_model.pth'
    if reconstructor_checkpoint_path.exists():
        logger.info(f"Loading hidden image reconstructor from {reconstructor_checkpoint_path}")
        hidden_reconstructor = HiddenImageReconstructor().to(device)
        
        checkpoint = torch.load(reconstructor_checkpoint_path, map_location=device)
        hidden_reconstructor.load_state_dict(checkpoint['model_state_dict'])
        hidden_reconstructor.eval()
        logger.info("Hidden image reconstructor loaded successfully")
    else:
        logger.warning(f"No hidden image reconstructor checkpoint found at {reconstructor_checkpoint_path}")
    
    return property_detector, hidden_reconstructor, device

def predict_properties(model, rgb_img, ms_masks, device):
    """Predict properties using the property detector model."""
    if model is None:
        # Return dummy properties and confidence scores if no model is available
        dummy_properties = {
            'pigment_type': 'unknown',
            'damage_type': 'unknown',
            'restoration': 'unknown',
            'hidden_content': 'unknown'
        }
        dummy_confidence = {
            'pigment_type': 0.0,
            'damage_type': 0.0,
            'restoration': 0.0,
            'hidden_content': 0.0
        }
        return dummy_properties, dummy_confidence
    
    # Prepare inputs
    rgb_img = rgb_img.unsqueeze(0).to(device)  # Add batch dimension
    ms_masks = ms_masks.unsqueeze(0).to(device)  # Add batch dimension
    
    # Use the model's predict_properties method if available
    if hasattr(model, 'predict_properties'):
        # This will return a dictionary mapping property names to lists of predicted categories
        predictions = model.predict_properties(rgb_img, ms_masks)
        
        # Convert batch predictions (lists) to single predictions
        properties = {k: v[0] for k, v in predictions.items()}
        
        # Get confidence scores
        with torch.no_grad():
            outputs = model(rgb_img, ms_masks)
            
        confidence_scores = {}
        for prop_name in properties.keys():
            logits_key = f'{prop_name}_logits'
            if logits_key in outputs:
                logits = outputs[logits_key]
                pred_idx = torch.argmax(logits, dim=1).item()
                confidence = torch.softmax(logits, dim=1)[0, pred_idx].item()
                confidence_scores[prop_name] = confidence
            else:
                confidence_scores[prop_name] = 0.0
                
        return properties, confidence_scores
    
    # Fallback to manual prediction if predict_properties is not available
    with torch.no_grad():
        outputs = model(rgb_img, ms_masks)
    
    # Process outputs
    properties = {}
    confidence_scores = {}
    
    # Extract property predictions from logits
    for key in outputs.keys():
        if key.endswith('_logits') and isinstance(outputs[key], torch.Tensor):
            prop_name = key.replace('_logits', '')
            logits = outputs[key]
            
            # Get predicted class index
            pred_idx = torch.argmax(logits, dim=1).item()
            
            # Get class name from PROPERTY_CATEGORIES if available
            if hasattr(model, 'PROPERTY_CATEGORIES') and prop_name in model.PROPERTY_CATEGORIES:
                categories = model.PROPERTY_CATEGORIES[prop_name]
                pred_class = categories[pred_idx]
            else:
                # Use a generic class name
                pred_class = f"Class_{pred_idx}"
            
            # Get confidence score
            confidence = torch.softmax(logits, dim=1)[0, pred_idx].item()
            
            properties[prop_name] = pred_class
            confidence_scores[prop_name] = confidence
    
    return properties, confidence_scores

def reconstruct_hidden_image(model, ms_masks, device):
    """Reconstruct hidden image using the hidden image reconstructor model."""
    if model is None:
        # Return dummy reconstruction if no model is available
        return torch.zeros_like(ms_masks[:3])  # Just return zeros in RGB shape
    
    # Prepare input
    ms_masks = ms_masks.unsqueeze(0).to(device)  # Add batch dimension
    
    # Get prediction
    with torch.no_grad():
        reconstructed_img = model(ms_masks)
        
        # Handle different output formats
        if isinstance(reconstructed_img, dict) and 'output' in reconstructed_img:
            reconstructed_img = reconstructed_img['output']
    
    # Remove batch dimension
    reconstructed_img = reconstructed_img.squeeze(0)
    
    # Ensure proper normalization for visualization
    if reconstructed_img.min() < 0 or reconstructed_img.max() > 1:
        reconstructed_img = (reconstructed_img - reconstructed_img.min()) / (reconstructed_img.max() - reconstructed_img.min() + 1e-8)
    
    return reconstructed_img

def visualize_sample(rgb_img, ms_masks, property_detector, hidden_reconstructor, device, output_dir, sample_idx):
    """Create visualizations for a sample using trained models."""
    # Predict properties
    properties, confidence_scores = predict_properties(property_detector, rgb_img, ms_masks, device)
    
    # Filter out non-property keys (like 'features', 'rgb_features', etc.)
    property_keys = ['pigment_type', 'damage_type', 'restoration', 'hidden_content']
    filtered_properties = {k: v for k, v in properties.items() if k in property_keys}
    filtered_confidence = {k: v for k, v in confidence_scores.items() if k in property_keys}
    
    # If we don't have any of the expected properties, use all properties
    if not filtered_properties:
        filtered_properties = properties
        filtered_confidence = confidence_scores
    
    logger.info(f"Predicted properties: {filtered_properties}")
    
    # Reconstruct hidden image
    reconstructed_img = reconstruct_hidden_image(hidden_reconstructor, ms_masks, device)
    
    # Create visualizer
    visualizer = AdvancedVisualization(save_dir=output_dir)
    visualizer.band_wavelengths = BAND_WAVELENGTHS
    
    # Create hidden content visualization
    logger.info("Creating hidden content visualization...")
    visualizer.visualize_hidden_content(
        rgb_img=rgb_img,
        ms_masks=ms_masks,
        reconstructed_img=reconstructed_img,
        title="Hidden Content Detection",
        save_path=output_dir / f'hidden_content_{sample_idx}.png'
    )
    
    # Create property visualization
    logger.info("Creating property visualization...")
    # Use the visualizer's property detection visualization
    visualizer.visualize_property_detection(
        rgb_img=rgb_img,
        properties=filtered_properties,
        confidence_scores=filtered_confidence,
        title="Painting Property Detection",
        save_path=output_dir / f'properties_{sample_idx}.png'
    )
    
    return filtered_properties, reconstructed_img

def main():
    """Main function to visualize trained models."""
    parser = argparse.ArgumentParser(description='Visualize ArtExtract multispectral models')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory with model checkpoints')
    parser.add_argument('--output_dir', type=str, default='./visualizations', help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    property_detector, hidden_reconstructor, device = load_models(args)
    
    # Load dataset
    dataset = MultispectralDataset(args.data_dir, split='val')
    
    if len(dataset) == 0:
        logger.error(f"No samples found in dataset at {args.data_dir}")
        return
    
    # Limit number of samples
    num_samples = min(args.num_samples, len(dataset))
    
    # Visualize samples
    for i in range(num_samples):
        logger.info(f"Processing sample {i+1}/{num_samples}")
        rgb_img, ms_masks, filename = dataset[i]
        
        logger.info(f"Sample: {filename}")
        logger.info(f"RGB image shape: {rgb_img.shape}")
        logger.info(f"MS masks shape: {ms_masks.shape}")
        
        visualize_sample(
            rgb_img=rgb_img,
            ms_masks=ms_masks,
            property_detector=property_detector,
            hidden_reconstructor=hidden_reconstructor,
            device=device,
            output_dir=output_dir,
            sample_idx=i
        )
    
    logger.info(f"Visualization complete! Results saved to {output_dir}")

if __name__ == '__main__':
    main()
