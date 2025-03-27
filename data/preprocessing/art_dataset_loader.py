"""
Art Dataset Loader for ArtExtract project.
This module handles loading and basic processing of art datasets.
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WikiArtDataset:
    """
    Class for handling the WikiArt dataset from ArtGAN.
    
    This dataset contains artworks categorized by style, artist, and genre.
    Source: https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the WikiArt dataset loader.
        
        Args:
            base_dir: Base directory containing the WikiArt dataset
        """
        self.base_dir = Path(base_dir)
        self.style_dir = self.base_dir / 'style'
        self.artist_dir = self.base_dir / 'artist'
        self.genre_dir = self.base_dir / 'genre'
        
        # Verify dataset existence
        self._verify_dataset()
        
        # Initialize class mappings
        self.style_to_idx = {}
        self.idx_to_style = {}
        self.artist_to_idx = {}
        self.idx_to_artist = {}
        self.genre_to_idx = {}
        self.idx_to_genre = {}
        
        # Load metadata
        self._load_metadata()
    
    def _verify_dataset(self) -> None:
        """Verify that the dataset directories exist."""
        if not self.base_dir.exists():
            logger.error(f"Dataset base directory {self.base_dir} does not exist.")
            raise FileNotFoundError(f"Dataset base directory {self.base_dir} does not exist.")
        
        # Check if at least one of the category directories exists
        if not any([self.style_dir.exists(), self.artist_dir.exists(), self.genre_dir.exists()]):
            logger.error(f"None of the category directories exist in {self.base_dir}.")
            raise FileNotFoundError(f"None of the category directories exist in {self.base_dir}.")
    
    def _load_metadata(self) -> None:
        """Load metadata and create class mappings."""
        # Load styles
        if self.style_dir.exists():
            styles = [d.name for d in self.style_dir.iterdir() if d.is_dir()]
            self.style_to_idx = {style: idx for idx, style in enumerate(sorted(styles))}
            self.idx_to_style = {idx: style for style, idx in self.style_to_idx.items()}
            logger.info(f"Loaded {len(styles)} styles from {self.style_dir}")
        
        # Load artists
        if self.artist_dir.exists():
            artists = [d.name for d in self.artist_dir.iterdir() if d.is_dir()]
            self.artist_to_idx = {artist: idx for idx, artist in enumerate(sorted(artists))}
            self.idx_to_artist = {idx: artist for artist, idx in self.artist_to_idx.items()}
            logger.info(f"Loaded {len(artists)} artists from {self.artist_dir}")
        
        # Load genres
        if self.genre_dir.exists():
            genres = [d.name for d in self.genre_dir.iterdir() if d.is_dir()]
            self.genre_to_idx = {genre: idx for idx, genre in enumerate(sorted(genres))}
            self.idx_to_genre = {idx: genre for genre, idx in self.genre_to_idx.items()}
            logger.info(f"Loaded {len(genres)} genres from {self.genre_dir}")
    
    def create_dataframe(self, category: str = 'style') -> pd.DataFrame:
        """
        Create a DataFrame with image paths and labels.
        
        Args:
            category: One of 'style', 'artist', or 'genre'
            
        Returns:
            DataFrame with columns 'image_path' and 'label'
        """
        if category == 'style':
            category_dir = self.style_dir
            category_to_idx = self.style_to_idx
        elif category == 'artist':
            category_dir = self.artist_dir
            category_to_idx = self.artist_to_idx
        elif category == 'genre':
            category_dir = self.genre_dir
            category_to_idx = self.genre_to_idx
        else:
            raise ValueError(f"Invalid category: {category}. Must be one of 'style', 'artist', or 'genre'.")
        
        if not category_dir.exists():
            raise FileNotFoundError(f"Category directory {category_dir} does not exist.")
        
        data = []
        for class_dir in category_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_idx = category_to_idx[class_name]
                
                for img_path in class_dir.glob('*.jpg'):
                    data.append({
                        'image_path': str(img_path),
                        'label': class_idx,
                        'label_name': class_name
                    })
        
        df = pd.DataFrame(data)
        logger.info(f"Created DataFrame with {len(df)} images for category '{category}'")
        return df
    
    def load_image(self, image_path: str, target_size: Tuple[int, int] = None) -> Image.Image:
        """
        Load an image from the dataset.
        
        Args:
            image_path: Path to the image
            target_size: Optional tuple (width, height) to resize the image
            
        Returns:
            PIL Image object
        """
        try:
            img = Image.open(image_path).convert('RGB')
            if target_size:
                img = img.resize(target_size, Image.LANCZOS)
            return img
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise


class NationalGalleryDataset:
    """
    Class for handling the National Gallery of Art dataset.
    
    Source: https://github.com/NationalGalleryOfArt/opendata
    """
    
    def __init__(self, base_dir: str, metadata_file: str):
        """
        Initialize the National Gallery dataset loader.
        
        Args:
            base_dir: Base directory containing the image files
            metadata_file: Path to the metadata JSON file
        """
        self.base_dir = Path(base_dir)
        self.metadata_file = Path(metadata_file)
        
        # Verify dataset existence
        self._verify_dataset()
        
        # Load metadata
        self.metadata = self._load_metadata()
    
    def _verify_dataset(self) -> None:
        """Verify that the dataset directory and metadata file exist."""
        if not self.base_dir.exists():
            logger.error(f"Dataset base directory {self.base_dir} does not exist.")
            raise FileNotFoundError(f"Dataset base directory {self.base_dir} does not exist.")
        
        if not self.metadata_file.exists():
            logger.error(f"Metadata file {self.metadata_file} does not exist.")
            raise FileNotFoundError(f"Metadata file {self.metadata_file} does not exist.")
    
    def _load_metadata(self) -> Dict:
        """Load metadata from JSON file."""
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {self.metadata_file}")
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata from {self.metadata_file}: {e}")
            raise
    
    def create_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame with image paths and metadata.
        
        Returns:
            DataFrame with columns for image path and metadata fields
        """
        data = []
        
        for item in self.metadata:
            # Extract relevant fields (adjust based on actual metadata structure)
            item_data = {
                'object_id': item.get('objectID', ''),
                'title': item.get('title', ''),
                'artist': item.get('artist', ''),
                'date': item.get('date', ''),
                'medium': item.get('medium', ''),
                'dimensions': item.get('dimensions', ''),
                'classification': item.get('classification', ''),
                'image_path': str(self.base_dir / f"{item.get('objectID', '')}.jpg")
            }
            
            # Check if image file exists
            if Path(item_data['image_path']).exists():
                data.append(item_data)
        
        df = pd.DataFrame(data)
        logger.info(f"Created DataFrame with {len(df)} images from National Gallery dataset")
        return df
    
    def load_image(self, image_path: str, target_size: Tuple[int, int] = None) -> Image.Image:
        """
        Load an image from the dataset.
        
        Args:
            image_path: Path to the image
            target_size: Optional tuple (width, height) to resize the image
            
        Returns:
            PIL Image object
        """
        try:
            img = Image.open(image_path).convert('RGB')
            if target_size:
                img = img.resize(target_size, Image.LANCZOS)
            return img
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    print("Art Dataset Loader module")
    print("Use this module to load and process art datasets for the ArtExtract project.")
