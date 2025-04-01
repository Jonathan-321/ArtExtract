"""
WikiArt dataset loader for the ArtExtract project.
This module handles loading and preprocessing of the WikiArt dataset for art classification.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WikiArtDataset(Dataset):
    """Dataset class for loading WikiArt images with style, artist, and genre labels."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        attributes: List[str] = ['style', 'artist', 'genre'],
        transform: Optional[Callable] = None,
        limit_samples: Optional[int] = None
    ):
        """
        Initialize the WikiArt dataset.
        
        Args:
            data_dir: Root directory containing the WikiArt dataset
            split: 'train', 'val', or 'test'
            attributes: List of attributes to include ('style', 'artist', 'genre')
            transform: Optional transform to be applied to images
            limit_samples: Optional limit on the number of samples (for testing)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.attributes = attributes
        
        # Set up transforms
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if split == 'val' or split == 'test':
            # No augmentation for validation/test
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Create class mappings
        self.class_to_idx = {}
        self.idx_to_class = {}
        for attr in self.attributes:
            classes = sorted(list(set(item[attr] for item in self.metadata)))
            self.class_to_idx[attr] = {cls: i for i, cls in enumerate(classes)}
            self.idx_to_class[attr] = {i: cls for i, cls in enumerate(classes)}
        
        # Filter by split
        self.samples = [item for item in self.metadata if item['split'] == split]
        
        # Limit samples if specified
        if limit_samples and limit_samples < len(self.samples):
            self.samples = random.sample(self.samples, limit_samples)
            
        logger.info(f"Loaded {len(self.samples)} {split} samples from WikiArt dataset")
        for attr in self.attributes:
            logger.info(f"Number of {attr} classes: {len(self.class_to_idx[attr])}")
    
    def _load_metadata(self) -> List[Dict]:
        """
        Load metadata from JSON file or create it if it doesn't exist.
        
        Returns:
            List of metadata dictionaries
        """
        metadata_file = self.data_dir / 'metadata.json'
        
        if metadata_file.exists():
            # Load existing metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_file}")
                return metadata
        else:
            # Create metadata from directory structure
            logger.info(f"Creating metadata from directory structure")
            metadata = []
            
            # WikiArt is organized as style/artist/painting.jpg
            for style_dir in self.data_dir.glob('*'):
                if not style_dir.is_dir():
                    continue
                    
                style = style_dir.name
                
                for artist_dir in style_dir.glob('*'):
                    if not artist_dir.is_dir():
                        continue
                        
                    artist = artist_dir.name
                    
                    for img_path in artist_dir.glob('*.jpg'):
                        # Extract genre from filename if available
                        # Format: artist_title_genre.jpg
                        parts = img_path.stem.split('_')
                        genre = parts[-1] if len(parts) >= 3 else 'unknown'
                        
                        metadata.append({
                            'path': str(img_path.relative_to(self.data_dir)),
                            'style': style,
                            'artist': artist,
                            'genre': genre,
                            'split': self._assign_split(img_path.name)  # Assign split based on filename hash
                        })
            
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
                logger.info(f"Saved metadata to {metadata_file}")
                
            return metadata
    
    def _assign_split(self, filename: str) -> str:
        """
        Assign a split (train/val/test) based on filename hash.
        This ensures consistent splits across runs.
        
        Args:
            filename: Image filename
            
        Returns:
            Split name ('train', 'val', or 'test')
        """
        # Use hash of filename to determine split
        hash_val = hash(filename) % 100
        
        if hash_val < 80:
            return 'train'
        elif hash_val < 90:
            return 'val'
        else:
            return 'test'
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, labels) where labels is a dictionary mapping
            attribute names to class indices
        """
        sample = self.samples[idx]
        img_path = self.data_dir / sample['path']
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a random valid sample instead
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Apply transform
        if self.transform:
            img = self.transform(img)
        
        # Create labels dictionary
        labels = {}
        for attr in self.attributes:
            if attr in sample and sample[attr] in self.class_to_idx[attr]:
                labels[attr] = torch.tensor(self.class_to_idx[attr][sample[attr]], dtype=torch.long)
            else:
                # Use a default value for missing attributes
                labels[attr] = torch.tensor(-1, dtype=torch.long)
        
        return img, labels
    
    def get_class_weights(self) -> Dict[str, torch.Tensor]:
        """
        Calculate class weights to handle imbalanced data.
        
        Returns:
            Dictionary mapping attribute names to class weight tensors
        """
        weights = {}
        
        for attr in self.attributes:
            # Count samples per class
            class_counts = {}
            for sample in self.samples:
                if attr in sample and sample[attr] in self.class_to_idx[attr]:
                    cls_idx = self.class_to_idx[attr][sample[attr]]
                    class_counts[cls_idx] = class_counts.get(cls_idx, 0) + 1
            
            # Calculate weights (inverse of frequency)
            num_classes = len(self.class_to_idx[attr])
            weight_tensor = torch.ones(num_classes)
            
            for cls_idx in range(num_classes):
                count = class_counts.get(cls_idx, 0)
                if count > 0:
                    weight_tensor[cls_idx] = len(self.samples) / (num_classes * count)
            
            # Normalize weights
            weight_tensor = weight_tensor / weight_tensor.sum() * num_classes
            
            weights[attr] = weight_tensor
        
        return weights


def create_wikiart_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    attributes: List[str] = ['style', 'artist', 'genre'],
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    limit_samples: Optional[Dict[str, int]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, torch.Tensor]]:
    """
    Create train, validation, and test dataloaders for WikiArt.
    
    Args:
        data_dir: Root directory containing the WikiArt dataset
        batch_size: Batch size for the dataloaders
        num_workers: Number of workers for data loading
        attributes: List of attributes to include
        train_transform: Optional custom transform for training data
        val_transform: Optional custom transform for validation/test data
        limit_samples: Optional dict mapping split names to sample limits
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    # Create datasets
    train_dataset = WikiArtDataset(
        data_dir=data_dir,
        split='train',
        attributes=attributes,
        transform=train_transform,
        limit_samples=limit_samples.get('train') if limit_samples else None
    )
    
    val_dataset = WikiArtDataset(
        data_dir=data_dir,
        split='val',
        attributes=attributes,
        transform=val_transform,
        limit_samples=limit_samples.get('val') if limit_samples else None
    )
    
    test_dataset = WikiArtDataset(
        data_dir=data_dir,
        split='test',
        attributes=attributes,
        transform=val_transform,
        limit_samples=limit_samples.get('test') if limit_samples else None
    )
    
    # Calculate class weights
    class_weights = train_dataset.get_class_weights()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_weights
