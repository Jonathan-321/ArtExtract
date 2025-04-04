"""
Comprehensive testing and visualization script for ArtExtract models.
This script tests all models and generates visualizations for the README.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from tqdm import tqdm
import json
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from models.style_classification.cnn_rnn_model import CNNRNNModel
from models.similarity_detection.similarity_model import PaintingSimilaritySystem
from models.multispectral.hidden_image_reconstruction import HiddenImageReconstructor

# Import evaluation tools
from evaluation.visualization import (
    plot_training_history, plot_confusion_matrix, plot_feature_space,
    plot_class_distribution, plot_similarity_heatmap, visualize_model_predictions
)
from evaluation.visualization_tools import MultispectralVisualizer
from evaluation.classification_metrics import calculate_metrics
from evaluation.similarity_metrics import (
    calculate_precision_at_k, calculate_mean_average_precision,
    calculate_ndcg_at_k, calculate_mean_reciprocal_rank
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test and visualize ArtExtract models')
    
    # Common parameters
    parser.add_argument('--output_dir', type=str, default='./visualization_results',
                        help='Directory to save visualization results')
    
    # Classification model parameters
    parser.add_argument('--classification_model_path', type=str,
                        help='Path to the trained classification model checkpoint')
    parser.add_argument('--classification_data_dir', type=str,
                        help='Directory containing classification test samples')
    
    # Similarity model parameters
    parser.add_argument('--similarity_model_path', type=str,
                        help='Path to the trained similarity model checkpoint')
    parser.add_argument('--similarity_data_dir', type=str,
                        help='Directory containing similarity test samples')
    
    # Multispectral model parameters
    parser.add_argument('--multispectral_model_path', type=str,
                        help='Path to the trained multispectral model checkpoint')
    parser.add_argument('--multispectral_data_dir', type=str,
                        help='Directory containing multispectral test samples')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation (cuda or cpu)')
    
    return parser.parse_args()


def test_classification_model(model_path, data_dir, output_dir, device):
    """Test classification model and generate visualizations."""
    logger.info("Testing classification model...")
    
    # Create output directory
    output_dir = Path(output_dir) / 'classification'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    if not model_path or not os.path.exists(model_path):
        logger.warning(f"Classification model path not provided or does not exist: {model_path}")
        return
    
    logger.info(f"Loading classification model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_args = checkpoint.get('args', {})
        
        # Create model with same architecture
        num_classes = model_args.get('num_classes', 10)
        cnn_backbone = model_args.get('cnn_backbone', 'resnet50')
        hidden_size = model_args.get('hidden_size', 256)
        dropout = model_args.get('dropout', 0.5)
        
        model = CNNRNNModel(
            num_classes=num_classes,
            cnn_backbone=cnn_backbone,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully: {num_classes} classes, {cnn_backbone} backbone")
    except Exception as e:
        logger.error(f"Error loading classification model: {e}")
        return
    
    # Load test images
    if not data_dir or not os.path.exists(data_dir):
        logger.warning(f"Classification data directory not provided or does not exist: {data_dir}")
        return
    
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load sample images
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        logger.warning(f"No images found in {data_dir}")
        return
    
    logger.info(f"Found {len(image_paths)} test images")
    
    # Process images and make predictions
    all_images = []
    all_predictions = []
    all_confidences = []
    
    for img_path in tqdm(image_paths[:20], desc="Processing images"):  # Limit to 20 images
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
            
            all_images.append(img)
            all_predictions.append(prediction.item())
            all_confidences.append(confidence.item())
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
    
    # Create visualization of predictions
    if all_images:
        logger.info("Generating classification visualizations...")
        
        # Create a figure with multiple images and their predictions
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        # Placeholder class names (replace with actual class names if available)
        class_names = [f"Class {i}" for i in range(num_classes)]
        
        for i, (img, pred, conf) in enumerate(zip(all_images, all_predictions, all_confidences)):
            if i >= len(axes):
                break
                
            axes[i].imshow(img)
            axes[i].set_title(f"Pred: {class_names[pred]}\nConf: {conf:.2f}")
            axes[i].axis('off')
        
        # Hide unused axes
        for i in range(len(all_images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "classification_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Classification visualizations saved to {output_dir}")
        
        # Return the path to the visualization for README
        return str(output_dir / "classification_predictions.png")
    
    return None


def test_similarity_model(model_path, data_dir, output_dir, device):
    """Test similarity model and generate visualizations."""
    logger.info("Testing similarity model...")
    
    # Create output directory
    output_dir = Path(output_dir) / 'similarity'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    if not model_path or not os.path.exists(model_path):
        logger.warning(f"Similarity model path not provided or does not exist: {model_path}")
        return
    
    logger.info(f"Loading similarity model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create similarity system
        similarity_system = PaintingSimilaritySystem(
            feature_extractor_type=checkpoint.get('feature_extractor_type', 'resnet50'),
            feature_dim=checkpoint.get('feature_dim', 2048),
            similarity_metric=checkpoint.get('similarity_metric', 'cosine')
        )
        
        # Load model weights if applicable
        if 'model_state_dict' in checkpoint:
            similarity_system.feature_extractor.load_state_dict(checkpoint['model_state_dict'])
        
        similarity_system.to(device)
        similarity_system.eval()
        
        logger.info(f"Similarity model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading similarity model: {e}")
        return
    
    # Load test images
    if not data_dir or not os.path.exists(data_dir):
        logger.warning(f"Similarity data directory not provided or does not exist: {data_dir}")
        return
    
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load sample images
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        logger.warning(f"No images found in {data_dir}")
        return
    
    logger.info(f"Found {len(image_paths)} test images")
    
    # Extract features from all images
    all_images = []
    all_features = []
    
    for img_path in tqdm(image_paths[:20], desc="Extracting features"):  # Limit to 20 images
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = similarity_system.extract_features(img_tensor)
            
            all_images.append(img)
            all_features.append(features.cpu().numpy())
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
    
    # Compute similarity matrix
    if all_features:
        logger.info("Computing similarity matrix...")
        all_features = np.vstack(all_features)
        similarity_matrix = similarity_system.compute_similarity_matrix(torch.tensor(all_features).to(device))
        similarity_matrix = similarity_matrix.cpu().numpy()
        
        # Generate visualizations
        logger.info("Generating similarity visualizations...")
        
        # Heatmap of similarity matrix
        plt.figure(figsize=(12, 10))
        sns_heatmap = plt.pcolor(similarity_matrix, cmap='viridis')
        plt.colorbar(sns_heatmap)
        plt.title('Similarity Matrix Heatmap')
        plt.tight_layout()
        plt.savefig(output_dir / "similarity_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize similar paintings
        num_query = min(5, len(all_images))
        fig, axes = plt.subplots(num_query, 6, figsize=(20, 4*num_query))
        
        for i in range(num_query):
            # Query image
            axes[i, 0].imshow(all_images[i])
            axes[i, 0].set_title("Query Image")
            axes[i, 0].axis('off')
            
            # Find top 5 similar images
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[::-1][1:6]  # Skip the first one (self)
            
            for j, idx in enumerate(top_indices):
                if idx < len(all_images):
                    axes[i, j+1].imshow(all_images[idx])
                    axes[i, j+1].set_title(f"Sim: {similarities[idx]:.2f}")
                    axes[i, j+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "similar_paintings.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Similarity visualizations saved to {output_dir}")
        
        # Return the path to the visualization for README
        return str(output_dir / "similar_paintings.png")
    
    return None


def test_multispectral_model(model_path, data_dir, output_dir, device):
    """Test multispectral model and generate visualizations."""
    logger.info("Testing multispectral model...")
    
    # Create output directory
    output_dir = Path(output_dir) / 'multispectral'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    if not model_path or not os.path.exists(model_path):
        logger.warning(f"Multispectral model path not provided or does not exist: {model_path}")
        return
    
    logger.info(f"Loading multispectral model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model
        model = HiddenImageReconstructor(
            in_channels=checkpoint.get('in_channels', 8),
            out_channels=checkpoint.get('out_channels', 3)
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        logger.info(f"Multispectral model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading multispectral model: {e}")
        return
    
    # Load test data
    if not data_dir or not os.path.exists(data_dir):
        logger.warning(f"Multispectral data directory not provided or does not exist: {data_dir}")
        return
    
    # Create visualizer
    visualizer = MultispectralVisualizer(save_dir=output_dir)
    
    # Look for multispectral data files (assuming .npy format)
    ms_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
               if f.lower().endswith('.npy')]
    
    if not ms_files:
        logger.warning(f"No multispectral data files found in {data_dir}")
        return
    
    logger.info(f"Found {len(ms_files)} multispectral data files")
    
    # Process multispectral data and generate reconstructions
    for i, ms_file in enumerate(tqdm(ms_files[:5], desc="Processing multispectral data")):  # Limit to 5 files
        try:
            # Load multispectral data
            ms_data = np.load(ms_file)
            ms_tensor = torch.tensor(ms_data, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Generate reconstruction
            with torch.no_grad():
                reconstruction = model(ms_tensor)
            
            # Convert to numpy for visualization
            ms_data = ms_tensor[0].cpu().numpy()
            reconstruction = reconstruction[0].cpu().numpy()
            
            # Visualize spectral bands
            visualizer.visualize_spectral_bands(
                ms_masks=ms_data,
                title=f"Spectral Bands - Sample {i+1}",
                save_path=output_dir / f"spectral_bands_{i+1}.png",
                figsize=(15, 10)
            )
            
            # Visualize band differences
            visualizer.visualize_band_differences(
                ms_masks=ms_data,
                title=f"Band Differences - Sample {i+1}",
                save_path=output_dir / f"band_differences_{i+1}.png",
                figsize=(15, 10)
            )
            
            # Visualize reconstruction
            plt.figure(figsize=(10, 8))
            plt.imshow(np.transpose(reconstruction, (1, 2, 0)))
            plt.title(f"Reconstructed Hidden Image - Sample {i+1}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / f"reconstruction_{i+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error processing multispectral data {ms_file}: {e}")
    
    logger.info(f"Multispectral visualizations saved to {output_dir}")
    
    # Return the path to the visualization for README
    return str(output_dir / "reconstruction_1.png")


def main():
    """Main function."""
    args = parse_args()
    
    # Create main output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test models and generate visualizations
    classification_viz = test_classification_model(
        args.classification_model_path,
        args.classification_data_dir,
        output_dir,
        args.device
    )
    
    similarity_viz = test_similarity_model(
        args.similarity_model_path,
        args.similarity_data_dir,
        output_dir,
        args.device
    )
    
    multispectral_viz = test_multispectral_model(
        args.multispectral_model_path,
        args.multispectral_data_dir,
        output_dir,
        args.device
    )
    
    # Generate summary of results
    results = {
        "classification_visualization": classification_viz,
        "similarity_visualization": similarity_viz,
        "multispectral_visualization": multispectral_viz
    }
    
    with open(output_dir / "visualization_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"All visualizations completed. Results saved to {output_dir}")
    
    # Print instructions for README update
    print("\n" + "="*80)
    print("VISUALIZATION RESULTS")
    print("="*80)
    print("To add these visualizations to your README, use the following paths:")
    if classification_viz:
        print(f"Classification model visualization: {classification_viz}")
    if similarity_viz:
        print(f"Similarity model visualization: {similarity_viz}")
    if multispectral_viz:
        print(f"Multispectral model visualization: {multispectral_viz}")
    print("="*80)


if __name__ == "__main__":
    main()
