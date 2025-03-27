"""
Data Preprocessing module for ArtExtract project.
This module handles data augmentation, normalization, and dataset splitting.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import albumentations as A
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArtDatasetPreprocessor:
    """
    Class for preprocessing art datasets.
    """
    
    def __init__(self, 
                 img_size: Tuple[int, int] = (224, 224),
                 use_augmentation: bool = True,
                 normalize: bool = True,
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        """
        Initialize the preprocessor.
        
        Args:
            img_size: Target image size (width, height)
            use_augmentation: Whether to use data augmentation
            normalize: Whether to normalize the images
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        """
        self.img_size = img_size
        self.use_augmentation = use_augmentation
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
        # Create transforms
        self._create_transforms()
    
    def _create_transforms(self) -> None:
        """Create transformation pipelines for training and validation."""
        # Base transforms (always applied)
        base_transforms = [
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ]
        
        # Normalization (optional)
        if self.normalize:
            base_transforms.append(transforms.Normalize(mean=self.mean, std=self.std))
        
        # Validation/test transforms
        self.val_transforms = transforms.Compose(base_transforms)
        
        # Training transforms with augmentation (optional)
        if self.use_augmentation:
            train_transforms = [
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
            ]
            
            if self.normalize:
                train_transforms.append(transforms.Normalize(mean=self.mean, std=self.std))
            
            self.train_transforms = transforms.Compose(train_transforms)
        else:
            self.train_transforms = self.val_transforms
    
    def get_transforms(self, is_training: bool = True) -> Callable:
        """
        Get the appropriate transforms based on whether we're training or not.
        
        Args:
            is_training: Whether the transforms are for training data
            
        Returns:
            Composition of transforms
        """
        return self.train_transforms if is_training else self.val_transforms


class ArtDataset(Dataset):
    """
    PyTorch Dataset for art images.
    """
    
    def __init__(self, 
                 dataframe: pd.DataFrame,
                 transform: Optional[Callable] = None,
                 target_column: str = 'label'):
        """
        Initialize the dataset.
        
        Args:
            dataframe: DataFrame containing image paths and labels
            transform: Optional transform to apply to images
            target_column: Column name for the target labels
        """
        self.dataframe = dataframe
        self.transform = transform
        self.target_column = target_column
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image tensor, label tensor)
        """
        # Get image path and label
        row = self.dataframe.iloc[idx]
        img_path = row['image_path']
        label = row[self.target_column]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transforms if available
        if self.transform:
            img = self.transform(img)
        
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label


def create_data_loaders(dataframe: pd.DataFrame,
                       preprocessor: ArtDatasetPreprocessor,
                       batch_size: int = 32,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       num_workers: int = 4,
                       target_column: str = 'label',
                       random_seed: int = 42) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        dataframe: DataFrame containing image paths and labels
        preprocessor: ArtDatasetPreprocessor instance
        batch_size: Batch size for data loaders
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        num_workers: Number of workers for data loading
        target_column: Column name for the target labels
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing train, val, and test DataLoaders
    """
    # Create label encoder
    unique_labels = sorted(dataframe[target_column].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Encode labels
    dataframe = dataframe.copy()
    dataframe[target_column] = dataframe[target_column].map(label_to_idx)
    # Verify ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError("Train, validation, and test ratios must sum to 1")
    
    # Calculate split sizes
    total_size = len(dataframe)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Create datasets with appropriate transforms
    train_dataset = ArtDataset(
        dataframe=dataframe.sample(frac=1, random_state=random_seed).reset_index(drop=True),
        transform=preprocessor.get_transforms(is_training=True),
        target_column=target_column
    )
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        train_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Override transforms for validation and test datasets
    val_transforms = preprocessor.get_transforms(is_training=False)
    val_dataset.dataset.transform = val_transforms
    test_dataset.dataset.transform = val_transforms
    
    # Create data loaders
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
    
    logger.info(f"Created data loaders with {train_size} training, {val_size} validation, and {test_size} test samples")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def save_processed_data(dataframe: pd.DataFrame, output_path: str, filename: str = 'processed_data.csv') -> None:
    """
    Save processed data to disk.
    
    Args:
        dataframe: DataFrame to save
        output_path: Directory to save the data to
        filename: Name of the output file
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / filename
    dataframe.to_csv(output_file, index=False)
    logger.info(f"Saved processed data to {output_file}")


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing module")
    print("Use this module to preprocess art datasets for the ArtExtract project.")
