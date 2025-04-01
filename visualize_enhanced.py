#!/usr/bin/env python3
"""
Enhanced visualization script for ArtExtract trained models.
This script uses advanced visualization techniques to better highlight hidden content and properties.
"""

import argparse
import torch
import numpy as np
import logging
from pathlib import Path
import random
from tqdm import tqdm

from models.multispectral.property_detection import PaintingPropertyDetector
from models.multispectral.hidden_image_reconstruction import HiddenImageReconstructor
from data.multispectral.dataset import MultispectralDataset
from evaluation.enhanced_visualization import EnhancedVisualization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define band wavelengths (nm) for visualization
BAND_WAVELENGTHS = [400, 450, 500, 550, 600, 650, 700, 800]

def load_model(model_class, checkpoint_path, device):
    """Load a model from checkpoint."""
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        # Create model instance
        model = model_class()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        logger.info(f"Loaded model from {checkpoint_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

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
    """Create enhanced visualizations for a sample using trained models."""
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
        properties=filtered_properties,
        confidence_scores=filtered_confidence,
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
    
    return filtered_properties, reconstructed_img

def main():
    """Main function to visualize trained models with enhanced techniques."""
    parser = argparse.ArgumentParser(description='Visualize trained models with enhanced techniques')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--output_dir', type=str, default='./enhanced_visualizations', help='Path to output directory')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
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
        rgb_dir=data_dir / 'rgb_images',
        ms_dir=data_dir / 'ms_masks',
        transform=None  # No transforms for visualization
    )
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    
    # Load checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    property_detector = load_model(
        PaintingPropertyDetector,
        checkpoint_dir / 'property_detector.pth',
        device
    )
    
    hidden_reconstructor = load_model(
        HiddenImageReconstructor,
        checkpoint_dir / 'hidden_reconstructor.pth',
        device
    )
    
    # Select samples to visualize
    num_samples = min(args.num_samples, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    # Visualize samples
    for i, idx in enumerate(tqdm(sample_indices, desc="Visualizing samples")):
        # Get sample
        sample = dataset[idx]
        rgb_img, ms_masks = sample['rgb_img'], sample['ms_masks']
        
        # Log sample info
        logger.info(f"Processing sample {i+1}/{num_samples}")
        logger.info(f"Sample: {dataset.rgb_files[idx].name}")
        logger.info(f"RGB image shape: {rgb_img.shape}")
        logger.info(f"MS masks shape: {ms_masks.shape}")
        
        # Visualize sample
        visualize_sample(
            rgb_img=rgb_img,
            ms_masks=ms_masks,
            property_detector=property_detector,
            hidden_reconstructor=hidden_reconstructor,
            device=device,
            output_dir=output_dir,
            sample_idx=i
        )
    
    logger.info(f"Enhanced visualization complete! Results saved to {output_dir}")

if __name__ == '__main__':
    main()
