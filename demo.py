#!/usr/bin/env python3
"""
Interactive demo for the ArtExtract project.
This script demonstrates both the classification and similarity detection capabilities.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import gradio as gr
import os
import faiss
from torchvision import transforms

from models.classification.cnn_rnn_classifier import CNNRNNClassifier
from models.similarity.painting_similarity import PaintingSimilarityDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArtExtractDemo:
    """Demo class for ArtExtract project."""
    
    def __init__(
        self,
        classification_checkpoint: str = None,
        similarity_checkpoint: str = None,
        similarity_index: str = None,
        painting_ids_file: str = None,
        data_dir: str = None,
        device: str = None
    ):
        """
        Initialize the demo.
        
        Args:
            classification_checkpoint: Path to classification model checkpoint
            similarity_checkpoint: Path to similarity model checkpoint
            similarity_index: Path to similarity index file
            painting_ids_file: Path to painting IDs file
            data_dir: Path to data directory
            device: Device to use
        """
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize classification model
        self.classification_model = None
        self.idx_to_class = {}
        if classification_checkpoint and os.path.exists(classification_checkpoint):
            self._init_classification_model(classification_checkpoint)
        
        # Initialize similarity model
        self.similarity_model = None
        self.painting_ids = []
        self.data_dir = Path(data_dir) if data_dir else None
        
        if similarity_checkpoint and os.path.exists(similarity_checkpoint):
            self._init_similarity_model(
                similarity_checkpoint, 
                similarity_index, 
                painting_ids_file
            )
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _init_classification_model(self, checkpoint_path):
        """
        Initialize the classification model.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Get model configuration
            if 'args' in checkpoint:
                args = checkpoint['args']
                backbone = args.get('backbone', 'resnet50')
                rnn_hidden_size = args.get('rnn_hidden_size', 512)
                rnn_num_layers = args.get('rnn_num_layers', 2)
                dropout = args.get('dropout', 0.5)
                num_classes = args.get('num_classes', {})
            else:
                # Use default configuration
                backbone = 'resnet50'
                rnn_hidden_size = 512
                rnn_num_layers = 2
                dropout = 0.5
                
                # Try to infer num_classes from model state dict
                state_dict = checkpoint['model_state_dict']
                num_classes = {}
                for key in state_dict.keys():
                    if key.startswith('classifiers.') and key.endswith('.weight'):
                        attr_name = key.split('.')[1]
                        output_dim = state_dict[key].shape[0]
                        num_classes[attr_name] = output_dim
            
            # Create model
            self.classification_model = CNNRNNClassifier(
                num_classes=num_classes,
                backbone=backbone,
                pretrained=False,
                rnn_hidden_size=rnn_hidden_size,
                rnn_num_layers=rnn_num_layers,
                dropout=dropout
            )
            
            # Load weights
            self.classification_model.load_state_dict(checkpoint['model_state_dict'])
            self.classification_model = self.classification_model.to(self.device)
            self.classification_model.eval()
            
            # Load class mappings if available
            if 'idx_to_class' in checkpoint:
                self.idx_to_class = checkpoint['idx_to_class']
            
            logger.info(f"Classification model loaded from {checkpoint_path}")
            logger.info(f"Model attributes: {list(num_classes.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load classification model: {e}")
            self.classification_model = None
    
    def _init_similarity_model(self, checkpoint_path, index_path, painting_ids_file):
        """
        Initialize the similarity model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            index_path: Path to similarity index file
            painting_ids_file: Path to painting IDs file
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Get model configuration
            if 'args' in checkpoint:
                args = checkpoint['args']
                backbone = args.get('backbone', 'resnet50')
                feature_layer = args.get('feature_layer', 'avgpool')
                use_face_detection = args.get('use_face_detection', False)
                use_pose_estimation = args.get('use_pose_estimation', False)
            else:
                # Use default configuration
                backbone = 'resnet50'
                feature_layer = 'avgpool'
                use_face_detection = False
                use_pose_estimation = False
            
            # Create model
            self.similarity_model = PaintingSimilarityDetector(
                backbone=backbone,
                pretrained=False,
                feature_layer=feature_layer,
                device=self.device,
                use_face_detection=use_face_detection,
                use_pose_estimation=use_pose_estimation
            )
            
            # Load weights if available
            if 'model_state_dict' in checkpoint:
                self.similarity_model.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load index if available
            if index_path and os.path.exists(index_path):
                self.similarity_model.faiss_index = faiss.read_index(index_path)
                logger.info(f"Similarity index loaded from {index_path}")
            
            # Load painting IDs if available
            if painting_ids_file and os.path.exists(painting_ids_file):
                with open(painting_ids_file, 'r') as f:
                    self.painting_ids = json.load(f)
                self.similarity_model.painting_ids = self.painting_ids
                logger.info(f"Painting IDs loaded from {painting_ids_file}")
            
            logger.info(f"Similarity model loaded from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load similarity model: {e}")
            self.similarity_model = None
    
    def classify_artwork(self, image):
        """
        Classify artwork using the CNN-RNN model.
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            Dictionary of classification results
        """
        if self.classification_model is None:
            return {"error": "Classification model not loaded"}
        
        try:
            # Load image if path is provided
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            # Apply transform
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                logits = self.classification_model(img_tensor)
                
                results = {}
                for attr_name, attr_logits in logits.items():
                    # Get predicted class
                    _, predicted = attr_logits.max(1)
                    predicted_idx = predicted.item()
                    
                    # Get class name if available
                    if attr_name in self.idx_to_class and predicted_idx in self.idx_to_class[attr_name]:
                        predicted_class = self.idx_to_class[attr_name][predicted_idx]
                    else:
                        predicted_class = f"Class {predicted_idx}"
                    
                    # Get confidence
                    probabilities = torch.nn.functional.softmax(attr_logits, dim=1)
                    confidence = probabilities[0, predicted_idx].item()
                    
                    results[attr_name] = {
                        "class": predicted_class,
                        "confidence": confidence
                    }
                
                return results
                
        except Exception as e:
            logger.error(f"Error in classify_artwork: {e}")
            return {"error": str(e)}
    
    def find_similar_artworks(self, image, k=5):
        """
        Find similar artworks using the similarity model.
        
        Args:
            image: PIL Image or path to image
            k: Number of similar artworks to return
            
        Returns:
            List of similar artwork information
        """
        if self.similarity_model is None:
            return {"error": "Similarity model not loaded"}
        
        if self.similarity_model.faiss_index is None:
            return {"error": "Similarity index not loaded"}
        
        try:
            # Load image if path is provided
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            # Apply transform
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            features = self.similarity_model.extract_features(img_tensor)
            features_np = features.cpu().numpy()
            
            # Find similar paintings
            similar_ids, similarity_scores = self.similarity_model.find_similar(features_np, k=k)
            
            # Get similar painting information
            similar_info = []
            for i, (painting_id, score) in enumerate(zip(similar_ids, similarity_scores)):
                # Find image path
                if self.data_dir:
                    # Try to find the image in the data directory
                    img_path = None
                    for ext in ['.jpg', '.jpeg', '.png']:
                        candidate = self.data_dir / 'images' / f"{painting_id}{ext}"
                        if candidate.exists():
                            img_path = str(candidate)
                            break
                else:
                    img_path = None
                
                similar_info.append({
                    "id": painting_id,
                    "similarity": score,
                    "path": img_path
                })
            
            return similar_info
            
        except Exception as e:
            logger.error(f"Error in find_similar_artworks: {e}")
            return {"error": str(e)}
    
    def analyze_artwork(self, image):
        """
        Analyze artwork using both classification and similarity models.
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            Dictionary of analysis results
        """
        # Classify artwork
        classification_results = self.classify_artwork(image)
        
        # Find similar artworks
        similarity_results = self.find_similar_artworks(image)
        
        return {
            "classification": classification_results,
            "similarity": similarity_results
        }


def setup_gradio_interface(demo):
    """
    Set up Gradio interface for the demo.
    
    Args:
        demo: ArtExtractDemo instance
        
    Returns:
        Gradio interface
    """
    def classify_artwork(image):
        results = demo.classify_artwork(image)
        
        # Format results for display
        output = ""
        for attr, info in results.items():
            if "error" in info:
                output += f"Error: {info['error']}\n"
            else:
                output += f"**{attr.capitalize()}**: {info['class']} (Confidence: {info['confidence']:.2f})\n"
        
        return output
    
    def find_similar_artworks(image):
        results = demo.find_similar_artworks(image, k=5)
        
        if isinstance(results, dict) and "error" in results:
            return [None] * 5, ["Error: " + results["error"]] * 5
        
        # Prepare images and captions
        images = []
        captions = []
        
        for info in results:
            if "path" in info and info["path"]:
                try:
                    img = Image.open(info["path"])
                    images.append(img)
                    captions.append(f"ID: {info['id']}, Similarity: {info['similarity']:.3f}")
                except Exception as e:
                    images.append(None)
                    captions.append(f"Error loading image: {e}")
            else:
                images.append(None)
                captions.append(f"ID: {info['id']}, Similarity: {info['similarity']:.3f} (Image not available)")
        
        # Pad with None if less than 5 results
        while len(images) < 5:
            images.append(None)
            captions.append("")
        
        return images, captions
    
    def analyze_artwork(image):
        results = demo.analyze_artwork(image)
        
        # Format classification results
        classification_output = ""
        for attr, info in results["classification"].items():
            if "error" in info:
                classification_output += f"Error: {info['error']}\n"
            else:
                classification_output += f"**{attr.capitalize()}**: {info['class']} (Confidence: {info['confidence']:.2f})\n"
        
        # Get similar artworks
        similarity_results = results["similarity"]
        
        if isinstance(similarity_results, dict) and "error" in similarity_results:
            similarity_images = [None] * 5
            similarity_captions = ["Error: " + similarity_results["error"]] * 5
        else:
            # Prepare images and captions
            similarity_images = []
            similarity_captions = []
            
            for info in similarity_results:
                if "path" in info and info["path"]:
                    try:
                        img = Image.open(info["path"])
                        similarity_images.append(img)
                        similarity_captions.append(f"ID: {info['id']}, Similarity: {info['similarity']:.3f}")
                    except Exception as e:
                        similarity_images.append(None)
                        similarity_captions.append(f"Error loading image: {e}")
                else:
                    similarity_images.append(None)
                    similarity_captions.append(f"ID: {info['id']}, Similarity: {info['similarity']:.3f} (Image not available)")
            
            # Pad with None if less than 5 results
            while len(similarity_images) < 5:
                similarity_images.append(None)
                similarity_captions.append("")
        
        return classification_output, similarity_images, similarity_captions
    
    # Create tabs for different functionalities
    with gr.Blocks(title="ArtExtract Demo") as interface:
        gr.Markdown("# ArtExtract: Art Classification and Similarity Detection")
        gr.Markdown("Upload an artwork to analyze it using our AI models.")
        
        with gr.Tabs():
            with gr.TabItem("All-in-One Analysis"):
                with gr.Row():
                    input_image = gr.Image(type="pil", label="Upload Artwork")
                
                analyze_button = gr.Button("Analyze Artwork")
                
                with gr.Row():
                    classification_output = gr.Markdown(label="Classification Results")
                
                gr.Markdown("### Similar Artworks")
                with gr.Row():
                    similar_images = [gr.Image(type="pil", label=f"Similar {i+1}") for i in range(5)]
                
                with gr.Row():
                    similar_captions = [gr.Markdown(label=f"Info {i+1}") for i in range(5)]
                
                analyze_button.click(
                    analyze_artwork,
                    inputs=[input_image],
                    outputs=[classification_output, *similar_images, *similar_captions]
                )
            
            with gr.TabItem("Classification Only"):
                with gr.Row():
                    classify_image = gr.Image(type="pil", label="Upload Artwork")
                
                classify_button = gr.Button("Classify Artwork")
                
                with gr.Row():
                    classify_output = gr.Markdown(label="Classification Results")
                
                classify_button.click(
                    classify_artwork,
                    inputs=[classify_image],
                    outputs=[classify_output]
                )
            
            with gr.TabItem("Similarity Only"):
                with gr.Row():
                    similarity_image = gr.Image(type="pil", label="Upload Artwork")
                
                similarity_button = gr.Button("Find Similar Artworks")
                
                gr.Markdown("### Similar Artworks")
                with gr.Row():
                    similarity_output_images = [gr.Image(type="pil", label=f"Similar {i+1}") for i in range(5)]
                
                with gr.Row():
                    similarity_output_captions = [gr.Markdown(label=f"Info {i+1}") for i in range(5)]
                
                similarity_button.click(
                    find_similar_artworks,
                    inputs=[similarity_image],
                    outputs=[similarity_output_images, similarity_output_captions]
                )
        
        gr.Markdown("## About ArtExtract")
        gr.Markdown("""
        ArtExtract is a project that combines computer vision and deep learning to analyze artworks.
        
        ### Features:
        - **Classification**: Identify style, artist, and genre of artworks
        - **Similarity Detection**: Find artworks with similar visual features
        - **Outlier Detection**: Identify unusual or misclassified artworks
        
        ### Models:
        - CNN-RNN architecture for classification
        - Feature extraction with pretrained CNNs for similarity detection
        """)
    
    return interface


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ArtExtract Demo')
    
    # Model arguments
    parser.add_argument('--classification_checkpoint', type=str, default=None,
                        help='Path to classification model checkpoint')
    parser.add_argument('--similarity_checkpoint', type=str, default=None,
                        help='Path to similarity model checkpoint')
    parser.add_argument('--similarity_index', type=str, default=None,
                        help='Path to similarity index file')
    parser.add_argument('--painting_ids', type=str, default=None,
                        help='Path to painting IDs file')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory')
    
    # Other arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port for Gradio server')
    parser.add_argument('--share', action='store_true',
                        help='Create a public link for the interface')
    
    return parser.parse_args()


def main():
    """Main function to run the demo."""
    args = parse_args()
    
    # Create demo
    demo = ArtExtractDemo(
        classification_checkpoint=args.classification_checkpoint,
        similarity_checkpoint=args.similarity_checkpoint,
        similarity_index=args.similarity_index,
        painting_ids_file=args.painting_ids,
        data_dir=args.data_dir,
        device=args.device
    )
    
    # Set up Gradio interface
    interface = setup_gradio_interface(demo)
    
    # Launch interface
    interface.launch(server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
