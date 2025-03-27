"""
Setup script for ArtExtract project.
This script helps with project setup, including creating necessary directories.
"""

import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Setup ArtExtract project')
    
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Base directory for data')
    parser.add_argument('--create_dirs', action='store_true',
                        help='Create directory structure')
    
    return parser.parse_args()


def create_directory_structure(base_dir):
    """Create the project directory structure."""
    logger.info(f"Creating directory structure in {base_dir}")
    
    # Define directory structure
    directories = [
        'raw/wikiart/style',
        'raw/wikiart/artist',
        'raw/wikiart/genre',
        'raw/national_gallery/images',
        'processed/wikiart',
        'processed/national_gallery/features',
        'metadata'
    ]
    
    # Create directories
    for directory in directories:
        dir_path = Path(base_dir) / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    logger.info("Directory structure created successfully")


def main():
    """Main setup function."""
    args = parse_args()
    
    # Create directory structure if requested
    if args.create_dirs:
        create_directory_structure(args.data_dir)
    
    logger.info("Setup completed successfully")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download datasets:")
    print("   - WikiArt: https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md")
    print("   - National Gallery: https://github.com/NationalGalleryOfArt/opendata")
    print("3. Place datasets in the appropriate directories")
    print("4. Run preprocessing scripts")
    print("5. Train models")
    print("\nSee README.md for more detailed instructions")


if __name__ == "__main__":
    main()
