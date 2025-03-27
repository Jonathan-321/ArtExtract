"""
Training script for the painting similarity model.
This script trains a similarity model using features extracted from paintings.
"""

import argparse
import os
import logging
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# Add project root to path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.similarity_detection.feature_extraction import FeatureExtractor
from models.similarity_detection.similarity_model import (
    create_similarity_model,
    PaintingSimilaritySystem
)
from evaluation.similarity_metrics import SimilarityEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a painting similarity model')
    
    # Data arguments
    parser.add_argument('--features_path', type=str, required=True,
                        help='Path to the extracted features file')
    parser.add_argument('--metadata_path', type=str, default=None,
                        help='Path to the metadata file (optional)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save the trained model and results')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='faiss',
                        choices=['cosine', 'faiss'],
                        help='Type of similarity model to use')
    parser.add_argument('--index_type', type=str, default='L2',
                        choices=['L2', 'IP', 'Cosine'],
                        help='Type of index to use for Faiss (only used if model_type is faiss)')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for Faiss index (only used if model_type is faiss)')
    
    # Evaluation arguments
    parser.add_argument('--eval_split', type=float, default=0.1,
                        help='Fraction of data to use for evaluation')
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20],
                        help='k values for precision@k and NDCG@k')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize similar paintings for a few examples')
    parser.add_argument('--num_vis_examples', type=int, default=5,
                        help='Number of examples to visualize')
    
    return parser.parse_args()


def load_features(features_path):
    """
    Load extracted features from a file.
    
    Args:
        features_path: Path to the features file
        
    Returns:
        Tuple of (features, image_paths)
    """
    logger.info(f"Loading features from {features_path}")
    
    with open(features_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check if the file contains a dictionary with 'features' and 'image_paths' keys
    if isinstance(data, dict) and 'features' in data and 'image_paths' in data:
        features = data['features']
        image_paths = data['image_paths']
    else:
        # Assume the file contains a tuple of (features, image_paths)
        features, image_paths = data
    
    logger.info(f"Loaded features for {len(image_paths)} images")
    
    return features, image_paths


def load_metadata(metadata_path):
    """
    Load metadata from a file.
    
    Args:
        metadata_path: Path to the metadata file
        
    Returns:
        Metadata DataFrame or None if path is None
    """
    if metadata_path is None:
        return None
    
    logger.info(f"Loading metadata from {metadata_path}")
    
    # Determine file format based on extension
    ext = os.path.splitext(metadata_path)[1].lower()
    if ext == '.csv':
        metadata = pd.read_csv(metadata_path)
    elif ext == '.json':
        metadata = pd.read_json(metadata_path)
    elif ext in ['.pkl', '.pickle']:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    else:
        raise ValueError(f"Unsupported metadata file format: {ext}")
    
    logger.info(f"Loaded metadata with {len(metadata)} entries")
    
    return metadata


def create_train_eval_split(features, image_paths, eval_split, seed):
    """
    Create train and evaluation splits.
    
    Args:
        features: Feature vectors
        image_paths: Image paths
        eval_split: Fraction of data to use for evaluation
        seed: Random seed
        
    Returns:
        Tuple of (train_indices, eval_indices)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create indices
    indices = np.arange(len(image_paths))
    
    # Shuffle indices
    np.random.shuffle(indices)
    
    # Split indices
    split_idx = int(len(indices) * (1 - eval_split))
    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]
    
    logger.info(f"Created train-eval split: {len(train_indices)} train, {len(eval_indices)} eval")
    
    return train_indices, eval_indices


def train_similarity_model(args):
    """
    Train a similarity model.
    
    Args:
        args: Command line arguments
        
    Returns:
        Trained PaintingSimilaritySystem
    """
    # Load features and metadata
    features, image_paths = load_features(args.features_path)
    metadata = load_metadata(args.metadata_path)
    
    # Create train-eval split
    train_indices, eval_indices = create_train_eval_split(
        features, image_paths, args.eval_split, args.seed
    )
    
    # Create similarity model
    logger.info(f"Creating {args.model_type} similarity model")
    
    if args.model_type == 'faiss':
        # Get feature dimension
        feature_dim = features.shape[1]
        
        # Create Faiss model
        similarity_model = create_similarity_model(
            'faiss',
            feature_dim=feature_dim,
            index_type=args.index_type,
            use_gpu=args.use_gpu
        )
    else:
        # Create cosine similarity model
        similarity_model = create_similarity_model('cosine')
    
    # Create painting similarity system using training data
    train_features = features[train_indices]
    train_image_paths = [image_paths[i] for i in train_indices]
    
    # Create metadata subset for training if metadata is available
    train_metadata = None
    if metadata is not None:
        # Assuming metadata has a column that can be matched with image paths
        # Adjust as needed based on your metadata structure
        filename_to_metadata = {}
        
        # Try different approaches to match image paths with metadata
        if 'image_path' in metadata.columns:
            filename_to_metadata = {row['image_path']: row for _, row in metadata.iterrows()}
            train_metadata = pd.DataFrame([
                filename_to_metadata[path] for path in train_image_paths
                if path in filename_to_metadata
            ])
        elif 'filename' in metadata.columns:
            filename_to_metadata = {row['filename']: row for _, row in metadata.iterrows()}
            train_metadata = pd.DataFrame([
                filename_to_metadata[os.path.basename(path)] for path in train_image_paths
                if os.path.basename(path) in filename_to_metadata
            ])
        elif 'object_id' in metadata.columns:
            # Try to extract object ID from filename
            filename_to_metadata = {row['object_id']: row for _, row in metadata.iterrows()}
            train_metadata = pd.DataFrame([
                filename_to_metadata[os.path.splitext(os.path.basename(path))[0]] for path in train_image_paths
                if os.path.splitext(os.path.basename(path))[0] in filename_to_metadata
            ])
    
    # Create painting similarity system
    similarity_system = PaintingSimilaritySystem(
        similarity_model=similarity_model,
        features=train_features,
        image_paths=train_image_paths,
        metadata=train_metadata
    )
    
    logger.info(f"Created painting similarity system with {len(train_image_paths)} paintings")
    
    # Evaluate the model if eval_split > 0
    if args.eval_split > 0:
        evaluate_similarity_model(
            similarity_system=similarity_system,
            features=features,
            image_paths=image_paths,
            train_indices=train_indices,
            eval_indices=eval_indices,
            k_values=args.k_values,
            output_dir=args.output_dir
        )
    
    # Visualize similar paintings for a few examples if requested
    if args.visualize:
        visualize_similar_paintings(
            similarity_system=similarity_system,
            num_examples=args.num_vis_examples,
            output_dir=args.output_dir
        )
    
    # Save the trained model
    save_model(similarity_system, args.output_dir)
    
    return similarity_system


def evaluate_similarity_model(similarity_system, features, image_paths, train_indices, eval_indices, k_values, output_dir):
    """
    Evaluate a similarity model.
    
    Args:
        similarity_system: Trained PaintingSimilaritySystem
        features: All feature vectors
        image_paths: All image paths
        train_indices: Indices for training data
        eval_indices: Indices for evaluation data
        k_values: k values for precision@k and NDCG@k
        output_dir: Directory to save evaluation results
    """
    logger.info("Evaluating similarity model")
    
    # Create evaluator
    evaluator = SimilarityEvaluator()
    
    # For each evaluation query, find similar paintings in the training set
    all_recommended_items = []
    
    # We'll use all training indices as relevant items for simplicity
    # In a real scenario, you would have ground truth relevance information
    all_relevant_items = [train_indices.tolist() for _ in eval_indices]
    
    # Find similar paintings for each evaluation query
    for query_idx in tqdm(eval_indices, desc="Evaluating queries"):
        # Get query feature
        query_feature = features[query_idx]
        
        # Find similar paintings in the training set
        # We use the find_similar_to_new_painting method since the query is not in the system
        result = similarity_system.find_similar_to_new_painting(
            query_feature=query_feature,
            query_path=image_paths[query_idx],
            k=max(k_values)
        )
        
        # Get indices of similar paintings
        similar_indices = result['similar_indices']
        
        # Add to recommended items
        all_recommended_items.append(similar_indices)
    
    # Evaluate the model
    results = evaluator.evaluate_similarity_model(
        all_relevant_items=all_relevant_items,
        all_recommended_items=all_recommended_items,
        k_values=k_values
    )
    
    # Log results
    logger.info(f"Evaluation results:")
    logger.info(f"MAP: {results['map']:.4f}")
    logger.info(f"MRR: {results['mrr']:.4f}")
    for k, precision in results['precision_at_k'].items():
        logger.info(f"{k}: {precision:.4f}")
    
    # Plot precision@k
    os.makedirs(output_dir, exist_ok=True)
    evaluator.plot_precision_at_k(
        results=results,
        save_path=os.path.join(output_dir, 'precision_at_k.png')
    )
    
    # Save results to JSON
    evaluator.save_results_to_json(
        results=results,
        output_path=os.path.join(output_dir, 'evaluation_results.json')
    )


def visualize_similar_paintings(similarity_system, num_examples, output_dir):
    """
    Visualize similar paintings for a few examples.
    
    Args:
        similarity_system: Trained PaintingSimilaritySystem
        num_examples: Number of examples to visualize
        output_dir: Directory to save visualizations
    """
    logger.info(f"Visualizing similar paintings for {num_examples} examples")
    
    # Create output directory for visualizations
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get random indices
    num_paintings = len(similarity_system.image_paths)
    indices = np.random.choice(num_paintings, num_examples, replace=False)
    
    # Visualize similar paintings for each example
    for i, idx in enumerate(indices):
        # Visualize similar paintings
        save_path = os.path.join(vis_dir, f'similar_paintings_{i}.png')
        similarity_system.visualize_similar_paintings(
            query_idx=idx,
            k=5,
            save_path=save_path
        )


def save_model(similarity_system, output_dir):
    """
    Save the trained model.
    
    Args:
        similarity_system: Trained PaintingSimilaritySystem
        output_dir: Directory to save the model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, 'painting_similarity_system.pkl')
    similarity_system.save_system(model_path)
    
    logger.info(f"Saved painting similarity system to {model_path}")


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Train similarity model
    similarity_system = train_similarity_model(args)
