"""
Script to download the refined WikiArt dataset from the ArtGAN paper.
This script downloads the dataset from Google Drive and sets it up for training.
Uses the --fuzzy option for gdown to handle Google Drive links better.
"""

import os
import sys
import logging
import requests
import zipfile
import gdown
import subprocess
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URLs for WikiArt refined dataset
WIKIART_DATASET_URL = "https://drive.google.com/uc?id=1vTChp3nU5GQeLkPwotrybpUGUXj12BTK"
WIKIART_DATASET_ALTERNATIVE_URL = "http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip"
WIKIART_CSV_URL = "https://drive.google.com/uc?id=1uug57zp13wJDwb2nuHOQfR2Odr0hh1a8"

def download_file_from_google_drive(url: str, output_path: Path) -> bool:
    """
    Download a file from Google Drive using gdown with fuzzy option.
    
    Args:
        url: Google Drive URL
        output_path: Path to save the file
        
    Returns:
        True if download was successful, False otherwise
    """
    try:
        logger.info(f"Downloading from Google Drive: {url}")
        result = gdown.download(url, str(output_path), quiet=False, fuzzy=True)
        return result is not None
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

def extract_archive(archive_path: Path, extract_path: Path) -> bool:
    """
    Extract an archive file using multiple methods.
    Tries different extraction methods in case one fails.
    
    Args:
        archive_path: Path to archive file
        extract_path: Path to extract to
        
    Returns:
        True if extraction was successful, False otherwise
    """
    # Try zipfile first
    try:
        logger.info(f"Attempting to extract with zipfile: {archive_path}")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        return True
    except Exception as e:
        logger.info(f"zipfile extraction failed: {str(e)}")
    
    # Try using unzip command
    try:
        logger.info(f"Attempting to extract with unzip command: {archive_path}")
        subprocess.run(["unzip", "-o", str(archive_path), "-d", str(extract_path)], check=True)
        return True
    except Exception as e:
        logger.info(f"unzip command failed: {str(e)}")
    
    # Try using tar command
    try:
        logger.info(f"Attempting to extract with tar command: {archive_path}")
        subprocess.run(["tar", "-xf", str(archive_path), "-C", str(extract_path)], check=True)
        return True
    except Exception as e:
        logger.info(f"tar command failed: {str(e)}")
    
    # Try using 7z if available
    try:
        logger.info(f"Attempting to extract with 7z command: {archive_path}")
        subprocess.run(["7z", "x", str(archive_path), f"-o{str(extract_path)}"], check=True)
        return True
    except Exception as e:
        logger.info(f"7z command failed: {str(e)}")
    
    logger.error(f"All extraction methods failed for {archive_path}")
    return False

def download_using_wget(url: str, output_path: Path) -> bool:
    """
    Download a file using wget command.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        
    Returns:
        True if download was successful, False otherwise
    """
    try:
        logger.info(f"Downloading using wget: {url}")
        cmd = ["wget", "-O", str(output_path), url]
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        logger.error(f"Error downloading using wget: {str(e)}")
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
    
    # If download failed, try wget
    if not success and not dataset_zip_path.exists():
        logger.info("Direct download failed, trying wget...")
        success = download_using_wget(WIKIART_DATASET_ALTERNATIVE_URL, dataset_zip_path)
    
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
        if not extract_archive(dataset_zip_path, wikiart_dir):
            logger.error("Failed to extract dataset")
            return False
    
    # Extract CSV files
    if csv_success and csv_zip_path.exists():
        logger.info("Extracting CSV files...")
        if not extract_archive(csv_zip_path, wikiart_dir):
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

def create_synthetic_dataset(base_dir: Path, num_styles: int = 10, images_per_style: int = 50) -> None:
    """
    Create a synthetic dataset from the test dataset for scaling up training.
    This is a fallback if downloading from external sources fails.
    
    Args:
        base_dir: Base directory for the dataset
        num_styles: Number of styles to create
        images_per_style: Number of images per style
    """
    import shutil
    
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
    wikiart_dir = base_dir / 'wikiart_refined'
    wikiart_img_dir = wikiart_dir / 'wikiart'
    wikiart_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic styles by duplicating test styles
    style_names = ['abstract', 'baroque', 'cubism', 'expressionism', 'impressionism', 
                  'pop_art', 'realism', 'renaissance', 'romanticism', 'surrealism']
    
    # Create CSV files
    style_train_data = []
    style_val_data = []
    style_class_data = []
    
    for i, style_name in enumerate(style_names[:num_styles]):
        # Create style directory
        new_style_dir = wikiart_img_dir / style_name
        new_style_dir.mkdir(exist_ok=True)
        
        # Add to class data
        style_class_data.append(f"{i} {style_name}")
        
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
            
            # Add to CSV data (80% train, 20% val)
            rel_path = f"{style_name}/{target_img.name}"
            if j < int(images_per_style * 0.8):
                style_train_data.append(f"{rel_path},{i}")
            else:
                style_val_data.append(f"{rel_path},{i}")
    
    # Write CSV files
    with open(wikiart_dir / "style_train.csv", "w") as f:
        f.write("\n".join(style_train_data))
    
    with open(wikiart_dir / "style_val.csv", "w") as f:
        f.write("\n".join(style_val_data))
    
    with open(wikiart_dir / "style_class.txt", "w") as f:
        f.write("\n".join(style_class_data))
    
    logger.info(f"Created synthetic dataset at {wikiart_dir}")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Download and setup the refined WikiArt dataset')
    parser.add_argument('--base_dir', type=str, default='data', help='Directory to setup the dataset in')
    parser.add_argument('--download_dir', type=str, default='data/downloads', help='Directory to store downloaded files')
    parser.add_argument('--synthetic', action='store_true', help='Create synthetic dataset instead of downloading')
    args = parser.parse_args()
    
    if args.synthetic:
        # Create synthetic dataset
        create_synthetic_dataset(Path(args.base_dir))
    else:
        # Setup dataset
        if setup_wikiart_refined(args.base_dir, args.download_dir):
            # Verify dataset
            verify_dataset(Path(args.base_dir) / 'wikiart_refined')
            logger.info("WikiArt refined dataset setup complete")
        else:
            logger.error("Failed to setup WikiArt refined dataset")
            logger.info("Falling back to creating a synthetic dataset")
            create_synthetic_dataset(Path(args.base_dir))
