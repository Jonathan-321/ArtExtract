"""
Script to download the refined WikiArt dataset from the ArtGAN paper.
This script downloads the dataset from Google Drive and sets it up for training.
"""

import os
import sys
import logging
import requests
import zipfile
import gdown
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URLs for WikiArt refined dataset
WIKIART_DATASET_URL = "https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view?usp=drivesdk"
WIKIART_DATASET_ALTERNATIVE_URL = "http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip"
WIKIART_CSV_URL = "https://drive.google.com/file/d/1uug57zp13wJDwb2nuHOQfR2Odr0hh1a8/view?usp=sharing"

def download_file_from_google_drive(url: str, output_path: Path) -> bool:
    """
    Download a file from Google Drive.
    
    Args:
        url: Google Drive URL
        output_path: Path to save the file
        
    Returns:
        True if download was successful, False otherwise
    """
    try:
        logger.info(f"Downloading from Google Drive: {url}")
        gdown.download(url, str(output_path), quiet=False)
        return True
    except Exception as e:
        logger.error(f"Error downloading from Google Drive: {str(e)}")
        return False

def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        chunk_size: Size of chunks to download
        
    Returns:
        True if download was successful, False otherwise
    """
    try:
        logger.info(f"Downloading from URL: {url}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        logger.error(f"Error downloading from URL: {str(e)}")
        return False

def extract_zip(zip_path: Path, extract_path: Path) -> bool:
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to zip file
        extract_path: Path to extract to
        
    Returns:
        True if extraction was successful, False otherwise
    """
    try:
        logger.info(f"Extracting {zip_path} to {extract_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        return True
    except Exception as e:
        logger.error(f"Error extracting zip file: {str(e)}")
        return False

def setup_wikiart_refined(base_dir: str = 'data', download_dir: str = 'data/downloads') -> bool:
    """
    Download and setup the refined WikiArt dataset.
    
    Args:
        base_dir: Directory to setup the dataset in
        download_dir: Directory to store downloaded files
        
    Returns:
        True if setup was successful, False otherwise
    """
    base_dir = Path(base_dir)
    download_dir = Path(download_dir)
    wikiart_dir = base_dir / 'wikiart_refined'
    
    # Create directories
    base_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    wikiart_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    dataset_zip_path = download_dir / "wikiart_refined.zip"
    csv_zip_path = download_dir / "wikiart_csv.zip"
    
    # Check if dataset already exists
    if (wikiart_dir / "wikiart").exists():
        logger.info(f"WikiArt refined dataset already exists at {wikiart_dir}")
        return True
    
    # Try downloading the dataset
    success = False
    
    # Try Google Drive first
    if not dataset_zip_path.exists():
        logger.info("Attempting to download from Google Drive...")
        success = download_file_from_google_drive(WIKIART_DATASET_URL, dataset_zip_path)
    else:
        logger.info(f"Dataset zip already exists at {dataset_zip_path}")
        success = True
    
    # If Google Drive fails, try alternative URL
    if not success and not dataset_zip_path.exists():
        logger.info("Google Drive download failed, trying alternative URL...")
        success = download_file(WIKIART_DATASET_ALTERNATIVE_URL, dataset_zip_path)
    
    # If download failed, exit
    if not success and not dataset_zip_path.exists():
        logger.error("Failed to download WikiArt dataset")
        return False
    
    # Download CSV files
    csv_success = False
    if not csv_zip_path.exists():
        logger.info("Downloading CSV files...")
        csv_success = download_file_from_google_drive(WIKIART_CSV_URL, csv_zip_path)
    else:
        logger.info(f"CSV zip already exists at {csv_zip_path}")
        csv_success = True
    
    # Extract dataset
    if dataset_zip_path.exists():
        logger.info("Extracting dataset...")
        if not extract_zip(dataset_zip_path, wikiart_dir):
            logger.error("Failed to extract dataset")
            return False
    
    # Extract CSV files
    if csv_success and csv_zip_path.exists():
        logger.info("Extracting CSV files...")
        if not extract_zip(csv_zip_path, wikiart_dir):
            logger.warning("Failed to extract CSV files, but dataset extraction succeeded")
    
    logger.info(f"WikiArt refined dataset setup complete at {wikiart_dir}")
    return True

def verify_dataset(wikiart_dir: Path) -> bool:
    """
    Verify that the dataset was downloaded and extracted correctly.
    
    Args:
        wikiart_dir: Path to the WikiArt directory
        
    Returns:
        True if verification was successful, False otherwise
    """
    # Check if main directories exist
    if not (wikiart_dir / "wikiart").exists():
        logger.error(f"WikiArt directory not found at {wikiart_dir / 'wikiart'}")
        return False
    
    # Check if CSV files exist
    csv_files = [
        "style_train.csv",
        "style_val.csv",
        "artist_train.csv",
        "artist_val.csv",
        "genre_train.csv",
        "genre_val.csv"
    ]
    
    missing_files = [f for f in csv_files if not (wikiart_dir / f).exists()]
    if missing_files:
        logger.warning(f"Missing CSV files: {missing_files}")
        logger.warning("Some CSV files are missing, but the dataset may still be usable")
    
    # Count images
    image_count = 0
    for ext in ['*.jpg', '*.png']:
        image_count += len(list((wikiart_dir / "wikiart").glob(f"**/{ext}")))
    
    logger.info(f"Found {image_count} images in the dataset")
    
    return True

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Download and setup the refined WikiArt dataset')
    parser.add_argument('--base_dir', type=str, default='data', help='Directory to setup the dataset in')
    parser.add_argument('--download_dir', type=str, default='data/downloads', help='Directory to store downloaded files')
    args = parser.parse_args()
    
    # Setup dataset
    if setup_wikiart_refined(args.base_dir, args.download_dir):
        # Verify dataset
        verify_dataset(Path(args.base_dir) / 'wikiart_refined')
        logger.info("WikiArt refined dataset setup complete")
    else:
        logger.error("Failed to setup WikiArt refined dataset")
        sys.exit(1)
