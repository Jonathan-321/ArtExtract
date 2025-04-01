"""
National Gallery dataset loader for the ArtExtract project.
This module handles loading and preprocessing of the National Gallery dataset for painting similarity detection.
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
import requests
from tqdm import tqdm
import zipfile
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NationalGalleryDataset(Dataset):
    """Dataset class for loading National Gallery images with metadata."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        download: bool = False,
        limit_samples: Optional[int] = None
    ):
        """
        Initialize the National Gallery dataset.
        
        Args:
            data_dir: Root directory for the dataset
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied to images
            download: Whether to download the dataset if not found
            limit_samples: Optional limit on the number of samples (for testing)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
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
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset if requested and not already downloaded
        if download and not (self.data_dir / 'metadata.json').exists():
            self._download_dataset()
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Filter by split
        self.samples = [item for item in self.metadata if item['split'] == split]
        
        # Limit samples if specified
        if limit_samples and limit_samples < len(self.samples):
            self.samples = random.sample(self.samples, limit_samples)
            
        logger.info(f"Loaded {len(self.samples)} {split} samples from National Gallery dataset")
    
    def _download_dataset(self):
        """Download the National Gallery dataset."""
        logger.info("Downloading National Gallery dataset...")
        
        # URLs for data and metadata
        metadata_url = "https://github.com/NationalGalleryOfArt/opendata/raw/main/data/objects.csv"
        images_url = "https://github.com/NationalGalleryOfArt/opendata/raw/main/data/images/objects/thumbnail.zip"
        
        # Download metadata
        logger.info("Downloading metadata...")
        metadata_df = pd.read_csv(metadata_url)
        
        # Download images
        logger.info("Downloading images...")
        response = requests.get(images_url, stream=True)
        response.raise_for_status()
        
        # Extract images
        logger.info("Extracting images...")
        images_dir = self.data_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(images_dir)
        
        # Process metadata
        logger.info("Processing metadata...")
        metadata = []
        
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            # Check if image exists
            img_id = str(row['objectid'])
            img_path = images_dir / f"{img_id}.jpg"
            
            if not img_path.exists():
                continue
                
            # Extract relevant metadata
            item = {
                'id': img_id,
                'path': str(img_path.relative_to(self.data_dir)),
                'title': row.get('title', ''),
                'artist': row.get('attribution', ''),
                'date': row.get('displaydate', ''),
                'medium': row.get('medium', ''),
                'classification': row.get('classification', ''),
                'split': self._assign_split(img_id)  # Assign split based on ID
            }
            
            metadata.append(item)
        
        # Save processed metadata
        with open(self.data_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
            
        logger.info(f"Downloaded and processed {len(metadata)} samples")
    
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
            
            # Look for images
            images_dir = self.data_dir / 'images'
            if not images_dir.exists():
                logger.warning(f"Images directory not found: {images_dir}")
                return metadata
                
            for img_path in images_dir.glob('*.jpg'):
                img_id = img_path.stem
                
                metadata.append({
                    'id': img_id,
                    'path': str(img_path.relative_to(self.data_dir)),
                    'split': self._assign_split(img_id)  # Assign split based on ID
                })
            
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
                logger.info(f"Saved metadata to {metadata_file}")
                
            return metadata
    
    def _assign_split(self, img_id: str) -> str:
        """
        Assign a split (train/val/test) based on image ID.
        This ensures consistent splits across runs.
        
        Args:
            img_id: Image ID
            
        Returns:
            Split name ('train', 'val', or 'test')
        """
        # Use hash of ID to determine split
        hash_val = hash(img_id) % 100
        
        if hash_val < 80:
            return 'train'
        elif hash_val < 90:
            return 'val'
        else:
            return 'test'
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, metadata)
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
        
        return img, sample


def create_national_gallery_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    download: bool = False,
    limit_samples: Optional[Dict[str, int]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for National Gallery dataset.
    
    Args:
        data_dir: Root directory for the dataset
        batch_size: Batch size for the dataloaders
        num_workers: Number of workers for data loading
        train_transform: Optional custom transform for training data
        val_transform: Optional custom transform for validation/test data
        download: Whether to download the dataset if not found
        limit_samples: Optional dict mapping split names to sample limits
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = NationalGalleryDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform,
        download=download,
        limit_samples=limit_samples.get('train') if limit_samples else None
    )
    
    val_dataset = NationalGalleryDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform,
        download=False,  # Only download once
        limit_samples=limit_samples.get('val') if limit_samples else None
    )
    
    test_dataset = NationalGalleryDataset(
        data_dir=data_dir,
        split='test',
        transform=val_transform,
        download=False,  # Only download once
        limit_samples=limit_samples.get('test') if limit_samples else None
    )
    
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
    
    return train_loader, val_loader, test_loader
