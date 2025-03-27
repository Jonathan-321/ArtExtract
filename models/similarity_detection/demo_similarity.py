"""
Demo script for the painting similarity model.
This script demonstrates how to use the trained similarity model to find similar paintings.
"""

import argparse
import os
import logging
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import sys
from tqdm import tqdm

# Add project root to path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.similarity_detection.feature_extraction import FeatureExtractor
from models.similarity_detection.similarity_model import PaintingSimilaritySystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Demo for painting similarity model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained painting similarity system')
    
    # Query arguments
    parser.add_argument('--query_image', type=str, default=None,
                        help='Path to a query image (if not provided, a random image from the database will be used)')
    parser.add_argument('--query_idx', type=int, default=None,
                        help='Index of a query image in the database (if not provided and query_image is not provided, a random image will be used)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save the results')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of similar paintings to retrieve')
    
    # Feature extraction arguments (only used if query_image is provided)
    parser.add_argument('--feature_extractor', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet', 'clip'],
                        help='Feature extractor to use for query image')
    parser.add_argument('--feature_extractor_path', type=str, default=None,
                        help='Path to a saved feature extractor (if not provided, a new one will be created)')
    
    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode, allowing the user to select multiple query images')
    
    return parser.parse_args()


def load_similarity_system(model_path):
    """
    Load a trained painting similarity system.
    
    Args:
        model_path: Path to the trained system
        
    Returns:
        Loaded PaintingSimilaritySystem
    """
    logger.info(f"Loading painting similarity system from {model_path}")
    
    # Load the system
    system = PaintingSimilaritySystem.load_system(model_path)
    
    logger.info(f"Loaded painting similarity system with {len(system.image_paths)} paintings")
    
    return system


def load_or_create_feature_extractor(feature_extractor_type, feature_extractor_path=None):
    """
    Load or create a feature extractor.
    
    Args:
        feature_extractor_type: Type of feature extractor
        feature_extractor_path: Path to a saved feature extractor
        
    Returns:
        FeatureExtractor
    """
    if feature_extractor_path and os.path.exists(feature_extractor_path):
        logger.info(f"Loading feature extractor from {feature_extractor_path}")
        
        with open(feature_extractor_path, 'rb') as f:
            feature_extractor = pickle.load(f)
    else:
        logger.info(f"Creating new {feature_extractor_type} feature extractor")
        
        # Create feature extractor
        feature_extractor = FeatureExtractor(model_type=feature_extractor_type)
        
        # Save feature extractor if path is provided
        if feature_extractor_path:
            os.makedirs(os.path.dirname(feature_extractor_path), exist_ok=True)
            
            with open(feature_extractor_path, 'wb') as f:
                pickle.dump(feature_extractor, f)
            
            logger.info(f"Saved feature extractor to {feature_extractor_path}")
    
    return feature_extractor


def extract_features_from_image(feature_extractor, image_path):
    """
    Extract features from an image.
    
    Args:
        feature_extractor: FeatureExtractor
        image_path: Path to the image
        
    Returns:
        Feature vector
    """
    logger.info(f"Extracting features from {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Extract features
    features = feature_extractor.extract_features_from_image(image)
    
    return features


def find_similar_paintings(similarity_system, query_feature=None, query_path=None, query_idx=None, k=5):
    """
    Find paintings similar to a query.
    
    Args:
        similarity_system: PaintingSimilaritySystem
        query_feature: Feature vector of the query painting
        query_path: Path to the query painting
        query_idx: Index of the query painting in the database
        k: Number of similar paintings to retrieve
        
    Returns:
        Dictionary with similar paintings information
    """
    # Find similar paintings
    if query_idx is not None:
        # Use an existing painting in the database
        result = similarity_system.find_similar_paintings(query_idx=query_idx, k=k)
    elif query_feature is not None and query_path is not None:
        # Use a new painting not in the database
        result = similarity_system.find_similar_to_new_painting(
            query_feature=query_feature,
            query_path=query_path,
            k=k
        )
    else:
        raise ValueError("Either query_idx or (query_feature and query_path) must be provided")
    
    return result


def visualize_similar_paintings(result, output_path=None, figsize=(15, 10)):
    """
    Visualize similar paintings.
    
    Args:
        result: Result from find_similar_paintings
        output_path: Path to save the visualization
        figsize: Figure size
    """
    # Load images
    query_img = Image.open(result['query_path']).convert('RGB')
    similar_imgs = [Image.open(path).convert('RGB') for path in result['similar_paths']]
    
    # Get similarities
    similarities = result['similarities']
    
    # Create figure
    fig, axes = plt.subplots(1, len(similar_imgs) + 1, figsize=figsize)
    
    # Plot query image
    axes[0].imshow(query_img)
    axes[0].set_title('Query')
    axes[0].axis('off')
    
    # Plot similar images
    for i, (img, sim) in enumerate(zip(similar_imgs, similarities)):
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f'Similarity: {sim:.3f}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Visualization saved to {output_path}")
    
    plt.show()


def print_similar_paintings_info(result):
    """
    Print information about similar paintings.
    
    Args:
        result: Result from find_similar_paintings
    """
    print("\nSimilar Paintings:")
    print(f"Query: {result['query_path']}")
    
    # Print metadata if available
    if 'query_metadata' in result:
        print("\nQuery Metadata:")
        for key, value in result['query_metadata'].items():
            print(f"  {key}: {value}")
    
    print("\nSimilar Paintings:")
    for i, (path, sim) in enumerate(zip(result['similar_paths'], result['similarities'])):
        print(f"{i+1}. {path} (Similarity: {sim:.3f})")
        
        # Print metadata if available
        if 'similar_metadata' in result:
            metadata = result['similar_metadata'][i]
            print("  Metadata:")
            for key, value in metadata.items():
                print(f"    {key}: {value}")
        
        print()


def run_demo(args):
    """
    Run the demo.
    
    Args:
        args: Command line arguments
    """
    # Load similarity system
    similarity_system = load_similarity_system(args.model_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine query
    if args.query_image:
        # Load feature extractor
        feature_extractor = load_or_create_feature_extractor(
            feature_extractor_type=args.feature_extractor,
            feature_extractor_path=args.feature_extractor_path
        )
        
        # Extract features from query image
        query_feature = extract_features_from_image(feature_extractor, args.query_image)
        
        # Find similar paintings
        result = find_similar_paintings(
            similarity_system=similarity_system,
            query_feature=query_feature,
            query_path=args.query_image,
            k=args.k
        )
    elif args.query_idx is not None:
        # Use specified index
        result = find_similar_paintings(
            similarity_system=similarity_system,
            query_idx=args.query_idx,
            k=args.k
        )
    else:
        # Use random index
        query_idx = np.random.randint(len(similarity_system.image_paths))
        logger.info(f"Using random query index: {query_idx}")
        
        result = find_similar_paintings(
            similarity_system=similarity_system,
            query_idx=query_idx,
            k=args.k
        )
    
    # Print information
    print_similar_paintings_info(result)
    
    # Visualize similar paintings
    output_path = os.path.join(args.output_dir, 'similar_paintings.png')
    visualize_similar_paintings(result, output_path=output_path)
    
    return result


def run_interactive_demo(args):
    """
    Run the demo in interactive mode.
    
    Args:
        args: Command line arguments
    """
    # Load similarity system
    similarity_system = load_similarity_system(args.model_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load feature extractor if needed
    feature_extractor = None
    
    while True:
        print("\nPainting Similarity Demo")
        print("1. Use a random painting from the database")
        print("2. Use a specific painting from the database (by index)")
        print("3. Use a new painting (provide path)")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            # Use random index
            query_idx = np.random.randint(len(similarity_system.image_paths))
            logger.info(f"Using random query index: {query_idx}")
            
            result = find_similar_paintings(
                similarity_system=similarity_system,
                query_idx=query_idx,
                k=args.k
            )
            
            # Print information
            print_similar_paintings_info(result)
            
            # Visualize similar paintings
            output_path = os.path.join(args.output_dir, f'similar_paintings_random_{query_idx}.png')
            visualize_similar_paintings(result, output_path=output_path)
        
        elif choice == '2':
            # Get index from user
            try:
                query_idx = int(input(f"Enter index (0-{len(similarity_system.image_paths)-1}): "))
                
                if query_idx < 0 or query_idx >= len(similarity_system.image_paths):
                    print(f"Index must be between 0 and {len(similarity_system.image_paths)-1}")
                    continue
                
                result = find_similar_paintings(
                    similarity_system=similarity_system,
                    query_idx=query_idx,
                    k=args.k
                )
                
                # Print information
                print_similar_paintings_info(result)
                
                # Visualize similar paintings
                output_path = os.path.join(args.output_dir, f'similar_paintings_idx_{query_idx}.png')
                visualize_similar_paintings(result, output_path=output_path)
            
            except ValueError:
                print("Invalid index")
        
        elif choice == '3':
            # Get path from user
            query_image = input("Enter path to query image: ")
            
            if not os.path.exists(query_image):
                print(f"File not found: {query_image}")
                continue
            
            # Load feature extractor if not already loaded
            if feature_extractor is None:
                feature_extractor = load_or_create_feature_extractor(
                    feature_extractor_type=args.feature_extractor,
                    feature_extractor_path=args.feature_extractor_path
                )
            
            # Extract features from query image
            query_feature = extract_features_from_image(feature_extractor, query_image)
            
            # Find similar paintings
            result = find_similar_paintings(
                similarity_system=similarity_system,
                query_feature=query_feature,
                query_path=query_image,
                k=args.k
            )
            
            # Print information
            print_similar_paintings_info(result)
            
            # Visualize similar paintings
            output_path = os.path.join(args.output_dir, f'similar_paintings_custom_{os.path.basename(query_image)}.png')
            visualize_similar_paintings(result, output_path=output_path)
        
        elif choice == '4':
            # Exit
            print("Exiting demo")
            break
        
        else:
            print("Invalid choice")


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Run demo
    if args.interactive:
        run_interactive_demo(args)
    else:
        run_demo(args)
