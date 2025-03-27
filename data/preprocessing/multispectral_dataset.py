"""
Dataset loader for RGB images and their corresponding multispectral masks.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class MultispectralDataset(Dataset):
    """Dataset class for loading RGB images and their corresponding multispectral masks."""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        """
        Initialize the MultispectralDataset.
        
        Args:
            data_dir (str): Root directory containing the dataset
            split (str): 'train' or 'val'
            transform: Optional transform to be applied to RGB images
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Mask transform (only resize and convert to tensor)
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        # Get paths
        self.rgb_dir = self.data_dir / split / 'rgb_images'
        self.ms_dir = self.data_dir / split / 'ms_masks'
        
        # Get all RGB image files (supporting multiple formats)
        self.rgb_files = sorted([f for f in os.listdir(self.rgb_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        # Create mapping from RGB image to its MS masks
        self.ms_mapping = self._create_ms_mapping()
        
    def _create_ms_mapping(self) -> Dict[str, List[str]]:
        """Create a mapping from RGB images to their corresponding MS masks."""
        mapping = {}
        for rgb_file in self.rgb_files:
            # Remove extension to get base name
            # Remove _RGB and extension to get base name
            base_name = rgb_file.replace('_RGB', '').rsplit('.', 1)[0]
            
            # Find all corresponding MS masks
            ms_files = sorted([
                f for f in os.listdir(self.ms_dir)
                if f.startswith(base_name + '_ms_')
            ])
            
            if len(ms_files) != 8:
                print(f"Warning: {base_name} has {len(ms_files)} masks instead of 8")
                continue
                
            mapping[rgb_file] = ms_files
        
        return mapping
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.rgb_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (rgb_image, ms_masks) where:
                - rgb_image is a tensor of shape (3, H, W)
                - ms_masks is a tensor of shape (8, H, W)
        """
        # Get RGB image
        rgb_file = self.rgb_files[idx]
        rgb_path = self.rgb_dir / rgb_file
        rgb_image = Image.open(rgb_path).convert('RGB')
        
        # Apply transform to RGB image
        if self.transform:
            rgb_image = self.transform(rgb_image)
            
        # Get corresponding MS masks
        ms_files = self.ms_mapping[rgb_file]
        ms_masks = []
        
        for ms_file in ms_files:
            ms_path = self.ms_dir / ms_file
            ms_mask = Image.open(ms_path).convert('L')  # Convert to grayscale
            ms_mask = self.mask_transform(ms_mask)
            ms_masks.append(ms_mask)
            
        # Stack MS masks into a single tensor (8, H, W)
        ms_masks = torch.stack(ms_masks)
        # Remove the extra channel dimension
        ms_masks = ms_masks.squeeze(1)
        
        return rgb_image, ms_masks, self.rgb_files[idx]

def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform=None,
    val_transform=None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir (str): Root directory containing the dataset
        batch_size (int): Batch size for the dataloaders
        num_workers (int): Number of workers for data loading
        train_transform: Optional custom transform for training data
        val_transform: Optional custom transform for validation data
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = MultispectralDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = MultispectralDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
