"""
Dataset implementation for art classification.
This module provides classes for loading and preprocessing art images for
the CNN-RNN classifier.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import logging
from pathlib import Path
import random
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArtDataset(Dataset):
    """Dataset for art classification tasks."""
    
    def __init__(
        self,
        data_dir,
        split='train',
        transform=True,
        target_size=(224, 224),
        test_mode=False
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Whether to apply transformations
            target_size: Image size after preprocessing
            test_mode: Whether to use a simplified structure for testing
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_size = target_size
        self.transform = transform
        self.test_mode = test_mode
        
        # Define transformations
        if self.transform:
            if split == 'train':
                self.transform_pipeline = transforms.Compose([
                    transforms.Resize(self.target_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform_pipeline = transforms.Compose([
                    transforms.Resize(self.target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform_pipeline = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor()
            ])
        
        # Load dataset
        self.load_dataset()
    
    def load_dataset(self):
        """Load dataset from directory."""
        logger.info(f"Loading {self.split} dataset from {self.data_dir}")
        
        # Initialize data and attributes
        self.samples = []  # List of (image_path, labels_dict) tuples
        
        # Try to load metadata if it exists
        metadata_path = self.data_dir / 'metadata.json'
        self.class_names = {}
        
        if metadata_path.exists():
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.class_names = metadata.get('class_names', {})
            
            # Filter samples for current split
            all_samples = metadata.get('samples', [])
            split_samples = [s for s in all_samples if s.get('split') == self.split]
            
            for sample in split_samples:
                image_path = self.data_dir / sample['path']
                if image_path.exists():
                    labels = sample['labels']
                    self.samples.append((str(image_path), labels))
        
        # If no metadata or test mode, use a simple directory structure
        elif self.test_mode or not metadata_path.exists():
            # Assuming structure: data_dir/split/attribute/class/image.jpg
            # For test mode, we assume: data_dir/style/class/image.jpg
            
            # Default attribute is 'style' for test mode
            attributes = ['style'] if self.test_mode else os.listdir(self.data_dir)
            
            for attr in attributes:
                attr_dir = self.data_dir / attr if not self.test_mode else self.data_dir
                
                if not attr_dir.is_dir():
                    continue
                
                # Get class names for this attribute
                self.class_names[attr] = sorted([d.name for d in attr_dir.iterdir() if d.is_dir()])
                
                # Map class names to indices
                class_to_idx = {name: idx for idx, name in enumerate(self.class_names[attr])}
                
                # Load samples
                for class_name in self.class_names[attr]:
                    class_dir = attr_dir / class_name
                    
                    if not class_dir.is_dir():
                        continue
                    
                    for img_path in class_dir.glob('*.jpg') + class_dir.glob('*.jpeg') + class_dir.glob('*.png'):
                        if img_path.is_file():
                            labels = {attr: class_to_idx[class_name]}
                            self.samples.append((str(img_path), labels))
        
        logger.info(f"Loaded {len(self.samples)} samples for {self.split} split")
        
        if not self.samples:
            logger.warning(f"No samples found for {self.split} split in {self.data_dir}")
        
        if self.class_names:
            for attr, classes in self.class_names.items():
                logger.info(f"Attribute '{attr}' has {len(classes)} classes: {classes}")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            tuple: (image, labels)
                image: Tensor of shape (C, H, W)
                labels: Dictionary mapping attribute names to class indices
        """
        image_path, labels = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform_pipeline(image)
            
            # Convert labels to tensors
            label_tensors = {attr: torch.tensor(class_idx, dtype=torch.long)
                           for attr, class_idx in labels.items()}
            
            return image, label_tensors
            
        except Exception as e:
            logger.error(f"Error loading sample {image_path}: {e}")
            # Return a random valid sample as fallback
            return self.__getitem__(random.randint(0, len(self) - 1))


class ArtSimilarityDataset(Dataset):
    """Dataset for art similarity tasks."""
    
    def __init__(
        self,
        data_dir,
        transform=True,
        target_size=(224, 224)
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the dataset
            transform: Whether to apply transformations
            target_size: Image size after preprocessing
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.transform = transform
        
        # Define transformations
        if self.transform:
            self.transform_pipeline = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform_pipeline = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor()
            ])
        
        # Load dataset
        self.load_dataset()
    
    def load_dataset(self):
        """Load dataset from directory."""
        logger.info(f"Loading similarity dataset from {self.data_dir}")
        
        # Initialize data
        self.samples = []  # List of (image_path, metadata) tuples
        self.metadata = {}
        
        # Try to load metadata if it exists
        metadata_path = self.data_dir / 'metadata.json'
        
        if metadata_path.exists():
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Get samples from metadata
            for item in self.metadata.get('items', []):
                image_path = self.data_dir / item['path']
                if image_path.exists():
                    self.samples.append((str(image_path), item))
        else:
            # Just load all images
            for img_path in self.data_dir.glob('**/*.jpg') + self.data_dir.glob('**/*.jpeg') + self.data_dir.glob('**/*.png'):
                if img_path.is_file():
                    self.samples.append((str(img_path), {}))
        
        logger.info(f"Loaded {len(self.samples)} samples for similarity analysis")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            tuple: (image, metadata)
                image: Tensor of shape (C, H, W)
                metadata: Dictionary with image metadata
        """
        image_path, metadata = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform_pipeline(image)
            
            return image, metadata
            
        except Exception as e:
            logger.error(f"Error loading sample {image_path}: {e}")
            # Return a random valid sample as fallback
            return self.__getitem__(random.randint(0, len(self) - 1)) 