"""
ArtExtract Demo Application

This script provides a simple interface to demonstrate the capabilities of the ArtExtract project,
including style classification and painting similarity detection.
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.style_classification.cnn_rnn_model import CNNRNNModel
from models.similarity_detection.feature_extraction import FeatureExtractor
from models.similarity_detection.similarity_model import PaintingSimilaritySystem
from models.style_classification.outlier_detection import IsolationForestDetector, visualize_outliers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArtExtractDemo:
    """Demo application for ArtExtract project."""
    
    def __init__(self, 
                classification_model_path: str,
                similarity_model_path: str,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the demo application.
        
        Args:
            classification_model_path: Path to the classification model checkpoint
            similarity_model_path: Path to the similarity model checkpoint
            device: Device to run the models on
        """
        self.device = device
        logger.info(f"Using device: {device}")
        
        # Load classification model
        self.load_classification_model(classification_model_path)
        
        # Load similarity model
        self.load_similarity_model(similarity_model_path)
        
        # Define image transforms
        self.transform = torch.nn.Sequential(
            torch.nn.Resize((224, 224)),
            torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    
    def load_classification_model(self, model_path: str) -> None:
        """
        Load the classification model.
        
        Args:
            model_path: Path to the model checkpoint
        """
        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model configuration
            config = checkpoint.get('config', {})
            num_classes = config.get('num_classes', 27)  # Default to 27 classes (WikiArt styles)
            cnn_backbone = config.get('cnn_backbone', 'resnet18')
            use_rnn = config.get('use_rnn', False)
            
            # Create model
            self.classification_model = CNNRNNModel(
                num_classes=num_classes,
                cnn_backbone=cnn_backbone,
                use_rnn=use_rnn
            )
            
            # Load model weights
            self.classification_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Move model to device
            self.classification_model.to(self.device)
            
            # Set model to evaluation mode
            self.classification_model.eval()
            
            # Load class names
            self.class_names = checkpoint.get('class_names', [f"Class {i}" for i in range(num_classes)])
            
            logger.info(f"Classification model loaded successfully with {num_classes} classes")
            
        except Exception as e:
            logger.error(f"Error loading classification model: {e}")
            self.classification_model = None
            self.class_names = []
    
    def load_similarity_model(self, model_path: str) -> None:
        """
        Load the similarity model.
        
        Args:
            model_path: Path to the model checkpoint
        """
        try:
            # Load feature extractor
            self.feature_extractor = FeatureExtractor(model_type='resnet50')
            
            # Load similarity system
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get features and image paths
            features = checkpoint.get('features', None)
            image_paths = checkpoint.get('image_paths', [])
            
            if features is not None and len(image_paths) > 0:
                # Create similarity system
                self.similarity_system = PaintingSimilaritySystem(
                    similarity_model=None,  # Will be created internally
                    features=features,
                    image_paths=image_paths
                )
                
                logger.info(f"Similarity model loaded successfully with {len(image_paths)} images")
            else:
                logger.warning("Similarity model loaded but no features or image paths found")
                self.similarity_system = None
                
        except Exception as e:
            logger.error(f"Error loading similarity model: {e}")
            self.feature_extractor = None
            self.similarity_system = None
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess an image for model input.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Convert to tensor
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        # Apply transforms
        img_tensor = self.transform(img_tensor.unsqueeze(0))
        
        return img_tensor
    
    def classify_artwork(self, image_path: str) -> dict:
        """
        Classify an artwork.
        
        Args:
            image_path: Path to the artwork image
            
        Returns:
            Dictionary with classification results
        """
        if self.classification_model is None:
            return {"error": "Classification model not loaded"}
        
        try:
            # Preprocess image
            img_tensor = self.preprocess_image(image_path)
            
            # Move to device
            img_tensor = img_tensor.to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.classification_model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
            # Get top-5 predictions
            top_probs, top_indices = torch.topk(probabilities, 5)
            
            # Create result
            result = {
                "predictions": [
                    {
                        "class": self.class_names[idx.item()],
                        "probability": prob.item()
                    }
                    for prob, idx in zip(top_probs, top_indices)
                ]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying artwork: {e}")
            return {"error": str(e)}
    
    def find_similar_artworks(self, image_path: str, k: int = 5) -> dict:
        """
        Find similar artworks.
        
        Args:
            image_path: Path to the query artwork image
            k: Number of similar artworks to find
            
        Returns:
            Dictionary with similarity results
        """
        if self.feature_extractor is None or self.similarity_system is None:
            return {"error": "Similarity model not loaded"}
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features_from_image(image_path)
            
            # Find similar artworks
            similar_indices, distances = self.similarity_system.find_similar_paintings_by_features(
                query_features=features,
                k=k
            )
            
            # Get image paths
            similar_paths = [self.similarity_system.image_paths[idx] for idx in similar_indices]
            
            # Create result
            result = {
                "similar_artworks": [
                    {
                        "path": path,
                        "distance": dist
                    }
                    for path, dist in zip(similar_paths, distances)
                ]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error finding similar artworks: {e}")
            return {"error": str(e)}
    
    def detect_outliers(self, features: np.ndarray, labels: np.ndarray) -> dict:
        """
        Detect outliers in the dataset.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            
        Returns:
            Dictionary with outlier detection results
        """
        try:
            # Create detector
            detector = IsolationForestDetector(contamination=0.05)
            
            # Fit detector
            detector.fit(features)
            
            # Get outlier indices
            outlier_indices = detector.get_outlier_indices(features)
            
            # Create result
            result = {
                "total_samples": len(features),
                "outliers": len(outlier_indices),
                "outlier_indices": outlier_indices.tolist()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return {"error": str(e)}


def create_gradio_interface(demo: ArtExtractDemo) -> gr.Interface:
    """
    Create a Gradio interface for the demo.
    
    Args:
        demo: ArtExtract demo instance
        
    Returns:
        Gradio interface
    """
    # Define classification function
    def classify(image):
        # Save image temporarily
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        # Classify artwork
        result = demo.classify_artwork(temp_path)
        
        # Format result
        if "error" in result:
            return f"Error: {result['error']}"
        
        # Create markdown table
        output = "## Style Classification Results\n\n"
        output += "| Style | Probability |\n"
        output += "|-------|-------------|\n"
        
        for pred in result["predictions"]:
            output += f"| {pred['class']} | {pred['probability']:.4f} |\n"
        
        return output
    
    # Define similarity function
    def find_similar(image):
        # Save image temporarily
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        # Find similar artworks
        result = demo.find_similar_artworks(temp_path)
        
        # Format result
        if "error" in result:
            return f"Error: {result['error']}"
        
        # Create markdown output
        output = "## Similar Artworks\n\n"
        
        for i, artwork in enumerate(result["similar_artworks"]):
            output += f"### {i+1}. Similarity Score: {1.0 - artwork['distance']:.4f}\n"
            output += f"![Similar Artwork]({artwork['path']})\n\n"
        
        return output
    
    # Create tabs for different functions
    with gr.Blocks() as interface:
        gr.Markdown("# ArtExtract Demo")
        gr.Markdown("Upload an artwork to classify its style or find similar paintings.")
        
        with gr.Tabs():
            with gr.TabItem("Style Classification"):
                with gr.Row():
                    with gr.Column():
                        input_image1 = gr.Image(label="Upload Artwork", type="pil")
                        classify_btn = gr.Button("Classify Artwork")
                    with gr.Column():
                        output_text1 = gr.Markdown(label="Classification Results")
                
                classify_btn.click(classify, inputs=input_image1, outputs=output_text1)
            
            with gr.TabItem("Similarity Detection"):
                with gr.Row():
                    with gr.Column():
                        input_image2 = gr.Image(label="Upload Artwork", type="pil")
                        similar_btn = gr.Button("Find Similar Artworks")
                    with gr.Column():
                        output_text2 = gr.Markdown(label="Similar Artworks")
                
                similar_btn.click(find_similar, inputs=input_image2, outputs=output_text2)
    
    return interface


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="ArtExtract Demo")
    parser.add_argument("--classification_model", type=str, required=True,
                        help="Path to the classification model checkpoint")
    parser.add_argument("--similarity_model", type=str, required=True,
                        help="Path to the similarity model checkpoint")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the demo on")
    parser.add_argument("--share", action="store_true",
                        help="Whether to create a public link")
    args = parser.parse_args()
    
    # Create demo
    demo = ArtExtractDemo(
        classification_model_path=args.classification_model,
        similarity_model_path=args.similarity_model
    )
    
    # Create interface
    interface = create_gradio_interface(demo)
    
    # Launch interface
    interface.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
