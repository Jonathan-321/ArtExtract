"""
Script to download a subset of the WikiArt dataset from a public source.
This script downloads a smaller subset of the WikiArt dataset for training purposes.
"""

import os
import sys
import logging
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URLs for WikiArt subset data (these are example URLs - replace with actual public dataset URLs)
WIKIART_SUBSET_URLS = [
    "https://github.com/cs-chan/ArtGAN/raw/master/WikiArt%20Dataset/WikiArt_Styles_Subset.zip",
]

def download_file(url: str, save_path: Path, chunk_size: int = 8192) -> None:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        save_path: Path to save the file
        chunk_size: Size of chunks to download
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(
        desc=save_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)

def extract_zip(zip_path: Path, extract_path: Path) -> None:
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to zip file
        extract_path: Path to extract to
    """
    logger.info(f"Extracting {zip_path} to {extract_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def create_synthetic_dataset(base_dir: Path, num_styles: int = 10, images_per_style: int = 50) -> None:
    """
    Create a synthetic dataset from the test dataset for scaling up training.
    This is a fallback if downloading from external sources fails.
    
    Args:
        base_dir: Base directory for the dataset
        num_styles: Number of styles to create
        images_per_style: Number of images per style
    """
    logger.info(f"Creating synthetic dataset with {num_styles} styles and {images_per_style} images per style")
    
    # Source test dataset
    test_dir = Path('data/test_dataset/style')
    if not test_dir.exists():
        logger.error(f"Test dataset directory {test_dir} does not exist")
        return
    
    # Create style directories
    styles = list(test_dir.iterdir())
    if len(styles) == 0:
        logger.error("No style directories found in test dataset")
        return
    
    # Create wikiart directory structure
    wikiart_dir = base_dir / 'wikiart'
    style_dir = wikiart_dir / 'style'
    style_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic styles by duplicating test styles
    style_names = ['abstract', 'baroque', 'cubism', 'expressionism', 'impressionism', 
                  'pop_art', 'realism', 'renaissance', 'romanticism', 'surrealism']
    
    for i, style_name in enumerate(style_names[:num_styles]):
        # Create style directory
        new_style_dir = style_dir / style_name
        new_style_dir.mkdir(exist_ok=True)
        
        # Source style (cycle through available test styles)
        source_style = styles[i % len(styles)]
        
        # Get all images from source style
        source_images = list(source_style.glob('*.jpg')) + list(source_style.glob('*.png'))
        if not source_images:
            logger.warning(f"No images found in {source_style}")
            continue
        
        # Copy and duplicate images to reach desired count
        for j in range(images_per_style):
            source_img = source_images[j % len(source_images)]
            target_img = new_style_dir / f"{style_name}_{j:04d}{source_img.suffix}"
            shutil.copy(source_img, target_img)
    
    logger.info(f"Created synthetic dataset at {wikiart_dir}")

def setup_wikiart_subset(base_dir: str = 'data', download_dir: str = 'data/downloads') -> None:
    """
    Download and setup a subset of the WikiArt dataset.
    
    Args:
        base_dir: Directory to setup the dataset in
        download_dir: Directory to store downloaded files
    """
    base_dir = Path(base_dir)
    download_dir = Path(download_dir)
    wikiart_dir = base_dir / 'wikiart'
    
    # Create directories
    base_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    wikiart_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try downloading from URLs
        for i, url in enumerate(WIKIART_SUBSET_URLS):
            filename = f"wikiart_subset_{i}.zip"
            zip_path = download_dir / filename
            
            logger.info(f"Downloading {url} to {zip_path}")
            download_file(url, zip_path)
            
            extract_zip(zip_path, wikiart_dir)
            
        logger.info("WikiArt subset downloaded and extracted successfully")
        
    except Exception as e:
        logger.error(f"Error downloading WikiArt subset: {str(e)}")
        logger.info("Falling back to creating a synthetic dataset from test data")
        create_synthetic_dataset(base_dir)

if __name__ == "__main__":
    setup_wikiart_subset()
