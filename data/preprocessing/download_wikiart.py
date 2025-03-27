"""
Script to download and setup the WikiArt dataset.
"""

import os
import sys
import logging
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Instructions for downloading WikiArt dataset
KAGGLE_DATASET = "ipythonx/wikiart-gangogh-creating-art-gan"

INSTRUCTIONS = """
To download the WikiArt dataset:

1. Install Kaggle CLI and authenticate:
   pip install kaggle
   Place kaggle.json in ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json

2. Download the dataset:
   kaggle datasets download -d ipythonx/wikiart-gangogh-creating-art-gan
   
3. Extract to data/wikiart directory
"""

def download_wikiart_kaggle(download_dir: Path) -> None:
    """Download WikiArt dataset using Kaggle CLI."""
    try:
        import kaggle
        logger.info("Downloading WikiArt dataset from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(download_dir),
            unzip=True
        )
        return True
    except Exception as e:
        logger.error(f"Error downloading from Kaggle: {str(e)}")
        logger.info("\nManual download instructions:\n" + INSTRUCTIONS)
        return False

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

def setup_wikiart_dataset(base_dir: str = 'data/wikiart', download_dir: str = 'data/downloads') -> None:
    """
    Download and setup the WikiArt dataset using Kaggle.
    
    Args:
        base_dir: Directory to setup the dataset in
        download_dir: Directory to store downloaded files
    """
    base_dir = Path(base_dir)
    download_dir = Path(download_dir)
    
    # Create directories
    base_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Try downloading from Kaggle
    success = download_wikiart_kaggle(download_dir)
    
    if success:
        logger.info("WikiArt dataset downloaded and extracted successfully")
    else:
        logger.warning("Please follow the manual download instructions above")

if __name__ == "__main__":
    setup_wikiart_dataset()
