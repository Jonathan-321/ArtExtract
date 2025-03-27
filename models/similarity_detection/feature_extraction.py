"""
Feature extraction module for painting similarity detection.
This module implements methods for extracting features from paintings for similarity analysis.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from pathlib import Path
import os
from tqdm import tqdm
import pickle
from transformers import CLIPProcessor, CLIPModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Base class for feature extractors.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the feature extractor.
        
        Args:
            device: Device to run the model on
        """
        self.device = device
        logger.info(f"Initialized feature extractor on device: {device}")
    
    def extract_features(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extract features from images.
        
        Args:
            images: List of PIL images
            
        Returns:
            Array of features
        """
        raise NotImplementedError("Subclasses must implement extract_features")
    
    def extract_features_from_dataloader(self, dataloader: DataLoader) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from a dataloader.
        
        Args:
            dataloader: DataLoader with images
            
        Returns:
            Tuple of (features, image_paths)
        """
        raise NotImplementedError("Subclasses must implement extract_features_from_dataloader")
    
    def save_features(self, features: np.ndarray, image_paths: List[str], output_path: str) -> None:
        """
        Save extracted features to disk.
        
        Args:
            features: Array of features
            image_paths: List of image paths
            output_path: Path to save the features
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = {
            'features': features,
            'image_paths': image_paths
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved features to {output_path}")
    
    @staticmethod
    def load_features(input_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load features from disk.
        
        Args:
            input_path: Path to load the features from
            
        Returns:
            Tuple of (features, image_paths)
        """
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        features = data['features']
        image_paths = data['image_paths']
        
        logger.info(f"Loaded features from {input_path}")
        
        return features, image_paths


class CNNFeatureExtractor(FeatureExtractor):
    """
    Feature extractor using a pre-trained CNN.
    """
    
    def __init__(self, 
                model_name: str = 'resnet50', 
                pretrained: bool = True,
                layer_name: Optional[str] = None,
                img_size: Tuple[int, int] = (224, 224),
                normalize: bool = True,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the CNN feature extractor.
        
        Args:
            model_name: Name of the pre-trained model
            pretrained: Whether to use pre-trained weights
            layer_name: Name of the layer to extract features from (if None, use the penultimate layer)
            img_size: Image size for input to the model
            normalize: Whether to normalize the images
            device: Device to run the model on
        """
        super().__init__(device)
        
        self.model_name = model_name
        self.layer_name = layer_name
        self.img_size = img_size
        
        # Initialize model
        self._init_model(model_name, pretrained)
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
        
        # Create transforms
        self._create_transforms(normalize)
        
        logger.info(f"Initialized CNN feature extractor with {model_name} model")
    
    def _init_model(self, model_name: str, pretrained: bool) -> None:
        """
        Initialize the model.
        
        Args:
            model_name: Name of the pre-trained model
            pretrained: Whether to use pre-trained weights
        """
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            # Remove the final fully connected layer
            self.feature_dim = self.model.fc.in_features
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        elif model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            # Remove the final fully connected layer
            self.feature_dim = self.model.fc.in_features
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            # Use the features part of the model
            self.feature_dim = self.model.classifier[0].in_features
            self.model = self.model.features
        
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = self.model.classifier[1].in_features
            # Remove the classifier
            self.model = self.model.features
        
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
    
    def _create_transforms(self, normalize: bool) -> None:
        """
        Create image transforms.
        
        Args:
            normalize: Whether to normalize the images
        """
        transform_list = [
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ]
        
        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        self.transform = transforms.Compose(transform_list)
    
    def extract_features(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extract features from images.
        
        Args:
            images: List of PIL images
            
        Returns:
            Array of features
        """
        # Apply transforms
        tensors = [self.transform(img).unsqueeze(0) for img in images]
        batch = torch.cat(tensors, dim=0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(batch)
            features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
        
        return features.cpu().numpy()
    
    def extract_features_from_dataloader(self, dataloader: DataLoader) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from a dataloader.
        
        Args:
            dataloader: DataLoader with images
            
        Returns:
            Tuple of (features, image_paths)
        """
        all_features = []
        all_paths = []
        
        with torch.no_grad():
            for batch, paths in tqdm(dataloader, desc="Extracting features"):
                batch = batch.to(self.device)
                
                # Extract features
                features = self.model(batch)
                features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
                
                all_features.append(features.cpu().numpy())
                all_paths.extend(paths)
        
        return np.vstack(all_features), all_paths


class CLIPFeatureExtractor(FeatureExtractor):
    """
    Feature extractor using the CLIP model.
    
    CLIP (Contrastive Language-Image Pre-training) is a model that learns
    visual concepts from natural language supervision.
    """
    
    def __init__(self, 
                model_name: str = 'openai/clip-vit-base-patch32', 
                img_size: Tuple[int, int] = (224, 224),
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the CLIP feature extractor.
        
        Args:
            model_name: Name of the CLIP model
            img_size: Image size for input to the model
            device: Device to run the model on
        """
        super().__init__(device)
        
        self.model_name = model_name
        self.img_size = img_size
        
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info(f"Initialized CLIP feature extractor with {model_name} model")
    
    def extract_features(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extract features from images.
        
        Args:
            images: List of PIL images
            
        Returns:
            Array of features
        """
        # Process images
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        
        # Normalize features
        features = outputs / outputs.norm(dim=1, keepdim=True)
        
        return features.cpu().numpy()
    
    def extract_features_from_dataloader(self, dataloader: DataLoader) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from a dataloader.
        
        Args:
            dataloader: DataLoader with images
            
        Returns:
            Tuple of (features, image_paths)
        """
        all_features = []
        all_paths = []
        
        with torch.no_grad():
            for batch, paths in tqdm(dataloader, desc="Extracting features"):
                # Process images
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                
                # Extract features
                outputs = self.model.get_image_features(**inputs)
                
                # Normalize features
                features = outputs / outputs.norm(dim=1, keepdim=True)
                
                all_features.append(features.cpu().numpy())
                all_paths.extend(paths)
        
        return np.vstack(all_features), all_paths


class ImageDataset(Dataset):
    """
    Dataset for loading images for feature extraction.
    """
    
    def __init__(self, 
                image_paths: List[str], 
                transform: Optional[Callable] = None):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of image paths
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Union[Image.Image, torch.Tensor], str]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, image_path)
        """
        img_path = self.image_paths[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return img, img_path
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a placeholder image
            placeholder = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
            if self.transform:
                placeholder = self.transform(placeholder)
            
            return placeholder, img_path


def create_feature_extractor(extractor_type: str, **kwargs) -> FeatureExtractor:
    """
    Create a feature extractor.
    
    Args:
        extractor_type: Type of feature extractor ('cnn' or 'clip')
        **kwargs: Additional arguments for the feature extractor
        
    Returns:
        Feature extractor instance
    """
    if extractor_type == 'cnn':
        return CNNFeatureExtractor(**kwargs)
    elif extractor_type == 'clip':
        return CLIPFeatureExtractor(**kwargs)
    else:
        raise ValueError(f"Unsupported feature extractor type: {extractor_type}")


if __name__ == "__main__":
    # Example usage
    print("Feature Extraction module")
    print("Use this module to extract features from paintings for similarity analysis.")
