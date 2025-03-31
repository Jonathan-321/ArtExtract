"""
Script to extract the WikiArt dataset from an existing zip file.
This script extracts the dataset and sets it up for training.
"""

import os
import sys
import logging
import zipfile
import subprocess
import shutil
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def setup_wikiart_dataset(zip_path: Path, extract_dir: Path, skip_extraction: bool = False) -> bool:
    """
    Extract and setup the WikiArt dataset from an existing zip file.
    
    Args:
        zip_path: Path to the WikiArt zip file
        extract_dir: Directory to extract the dataset to
        skip_extraction: Skip extraction and just process the dataset
        
    Returns:
        True if setup was successful, False otherwise
    """
    # Create extraction directory
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # If skip_extraction is True, assume dataset is already extracted
    if skip_extraction:
        logger.info(f"Skipping extraction, processing dataset at {extract_dir}")
    else:
        # Check if zip file exists
        if not zip_path.exists():
            logger.error(f"WikiArt zip file not found at {zip_path}")
            return False
        
        # Extract dataset
        logger.info(f"Extracting WikiArt dataset from {zip_path} to {extract_dir}")
        if not extract_archive(zip_path, extract_dir):
            logger.error("Failed to extract WikiArt dataset")
            return False
    
    # Verify extraction
    if not verify_dataset(extract_dir):
        logger.error("Failed to verify WikiArt dataset")
        return False
    
    logger.info(f"WikiArt dataset setup complete at {extract_dir}")
    return True

def verify_dataset(extract_dir: Path) -> bool:
    """
    Verify that the dataset was extracted correctly.
    
    Args:
        extract_dir: Directory where the dataset was extracted
        
    Returns:
        True if verification was successful, False otherwise
    """
    # Check if wikiart directory exists
    wikiart_dir = extract_dir / "wikiart"
    if not wikiart_dir.exists():
        # Check if the zip contained a parent directory
        potential_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if potential_dirs:
            # Find the directory that might contain the dataset
            for potential_dir in potential_dirs:
                if (potential_dir / "wikiart").exists():
                    wikiart_dir = potential_dir / "wikiart"
                    break
                elif any(potential_dir.glob("**/train")) or any(potential_dir.glob("**/val")):
                    wikiart_dir = potential_dir
                    break
        
        if not wikiart_dir.exists():
            logger.error(f"WikiArt directory not found at {extract_dir}")
            return False
    
    # Count images
    image_count = 0
    for ext in ['*.jpg', '*.png']:
        image_count += len(list(wikiart_dir.glob(f"**/{ext}")))
    
    logger.info(f"Found {image_count} images in the dataset")
    
    # Check for train/val directories
    train_dir = wikiart_dir / "train"
    val_dir = wikiart_dir / "val"
    
    if train_dir.exists() and val_dir.exists():
        logger.info("Found train and validation directories")
        
        # Count styles/classes
        train_styles = [d for d in train_dir.iterdir() if d.is_dir()]
        val_styles = [d for d in val_dir.iterdir() if d.is_dir()]
        
        logger.info(f"Found {len(train_styles)} style classes in training set")
        logger.info(f"Found {len(val_styles)} style classes in validation set")
        
        # Create CSV files if they don't exist
        create_csv_files(wikiart_dir, train_dir, val_dir)
    else:
        # Look for style directories directly
        style_dirs = [d for d in wikiart_dir.iterdir() if d.is_dir()]
        if style_dirs:
            logger.info(f"Found {len(style_dirs)} potential style directories")
            
            # Create train/val split
            logger.info("Creating train/val split from style directories")
            create_train_val_split(wikiart_dir, style_dirs)
        else:
            logger.warning("Could not find style directories or train/val split")
    
    return True

def create_csv_files(wikiart_dir: Path, train_dir: Path, val_dir: Path) -> None:
    """
    Create CSV files for training and validation if they don't exist.
    
    Args:
        wikiart_dir: Path to the WikiArt directory
        train_dir: Path to the training directory
        val_dir: Path to the validation directory
    """
    # Check if CSV files already exist
    style_train_csv = wikiart_dir / "style_train.csv"
    style_val_csv = wikiart_dir / "style_val.csv"
    style_class_txt = wikiart_dir / "style_class.txt"
    
    if style_train_csv.exists() and style_val_csv.exists():
        logger.info("CSV files already exist")
        return
    
    logger.info("Creating CSV files for training and validation")
    
    # Get style directories
    train_styles = [d for d in train_dir.iterdir() if d.is_dir()]
    val_styles = [d for d in val_dir.iterdir() if d.is_dir()]
    
    # Create style to index mapping
    style_to_idx = {style.name: i for i, style in enumerate(sorted(train_styles, key=lambda x: x.name))}
    
    # Create CSV files
    train_data = []
    val_data = []
    class_data = []
    
    # Add training data
    for style_dir in train_styles:
        style_name = style_dir.name
        style_idx = style_to_idx[style_name]
        
        # Add to class data
        class_data.append(f"{style_idx} {style_name}")
        
        # Add images to training data
        for img_path in style_dir.glob("*.jpg"):
            rel_path = f"train/{style_name}/{img_path.name}"
            train_data.append(f"{rel_path},{style_idx}")
        
        for img_path in style_dir.glob("*.png"):
            rel_path = f"train/{style_name}/{img_path.name}"
            train_data.append(f"{rel_path},{style_idx}")
    
    # Add validation data
    for style_dir in val_styles:
        style_name = style_dir.name
        if style_name in style_to_idx:
            style_idx = style_to_idx[style_name]
            
            # Add images to validation data
            for img_path in style_dir.glob("*.jpg"):
                rel_path = f"val/{style_name}/{img_path.name}"
                val_data.append(f"{rel_path},{style_idx}")
            
            for img_path in style_dir.glob("*.png"):
                rel_path = f"val/{style_name}/{img_path.name}"
                val_data.append(f"{rel_path},{style_idx}")
    
    # Write CSV files
    with open(style_train_csv, "w") as f:
        f.write("\n".join(train_data))
    
    with open(style_val_csv, "w") as f:
        f.write("\n".join(val_data))
    
    with open(style_class_txt, "w") as f:
        f.write("\n".join(class_data))
    
    logger.info(f"Created CSV files: {style_train_csv}, {style_val_csv}, {style_class_txt}")
    logger.info(f"Training examples: {len(train_data)}, Validation examples: {len(val_data)}")

def create_train_val_split(wikiart_dir: Path, style_dirs: list) -> None:
    """
    Create train/val split from style directories.
    
    Args:
        wikiart_dir: Path to the WikiArt directory
        style_dirs: List of style directories
    """
    import random
    
    # Create train/val directories
    train_dir = wikiart_dir / "train"
    val_dir = wikiart_dir / "val"
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Create style to index mapping
    style_to_idx = {style.name: i for i, style in enumerate(sorted(style_dirs, key=lambda x: x.name))}
    
    # Create CSV files
    train_data = []
    val_data = []
    class_data = []
    
    # Add class data
    for style_name, style_idx in style_to_idx.items():
        class_data.append(f"{style_idx} {style_name}")
    
    # Split data into train/val
    for style_dir in style_dirs:
        style_name = style_dir.name
        style_idx = style_to_idx[style_name]
        
        # Create style directories in train/val
        train_style_dir = train_dir / style_name
        val_style_dir = val_dir / style_name
        
        train_style_dir.mkdir(exist_ok=True)
        val_style_dir.mkdir(exist_ok=True)
        
        # Get all images
        images = list(style_dir.glob("*.jpg")) + list(style_dir.glob("*.png"))
        
        # Shuffle images
        random.shuffle(images)
        
        # Split 80/20
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy images to train/val
        for img_path in train_images:
            target_path = train_style_dir / img_path.name
            if not target_path.exists():
                shutil.copy(img_path, target_path)
            
            # Add to training data
            rel_path = f"train/{style_name}/{img_path.name}"
            train_data.append(f"{rel_path},{style_idx}")
        
        for img_path in val_images:
            target_path = val_style_dir / img_path.name
            if not target_path.exists():
                shutil.copy(img_path, target_path)
            
            # Add to validation data
            rel_path = f"val/{style_name}/{img_path.name}"
            val_data.append(f"{rel_path},{style_idx}")
    
    # Write CSV files
    with open(wikiart_dir / "style_train.csv", "w") as f:
        f.write("\n".join(train_data))
    
    with open(wikiart_dir / "style_val.csv", "w") as f:
        f.write("\n".join(val_data))
    
    with open(wikiart_dir / "style_class.txt", "w") as f:
        f.write("\n".join(class_data))
    
    logger.info(f"Created train/val split with {len(train_data)} training examples and {len(val_data)} validation examples")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Extract and setup the WikiArt dataset')
    parser.add_argument('--zip_path', type=str, default='~/Downloads/wikiart.zip', help='Path to the WikiArt zip file')
    parser.add_argument('--extract_dir', type=str, default='data/wikiart', help='Directory to extract the dataset to')
    parser.add_argument('--skip_extraction', action='store_true', help='Skip extraction and just process the dataset')
    args = parser.parse_args()
    
    # Expand user directory
    zip_path = Path(os.path.expanduser(args.zip_path))
    extract_dir = Path(args.extract_dir)
    
    # Setup dataset
    if setup_wikiart_dataset(zip_path, extract_dir, args.skip_extraction):
        logger.info("WikiArt dataset setup complete")
    else:
        logger.error("Failed to setup WikiArt dataset")
        sys.exit(1)
