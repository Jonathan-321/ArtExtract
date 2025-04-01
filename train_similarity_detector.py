#!/usr/bin/env python3
"""
Training script for Painting Similarity Detector.
This script extracts features from paintings and builds a similarity index.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import faiss
import cv2
from datetime import datetime

from models.similarity.painting_similarity import PaintingSimilarityDetector
from data.similarity.national_gallery_dataset import create_national_gallery_dataloaders

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Painting Similarity Detector')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to National Gallery dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--download', action='store_true',
                        help='Download dataset if not found')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='CNN backbone architecture')
    parser.add_argument('--feature_layer', type=str, default='avgpool',
                        help='Layer to extract features from')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone')
    parser.add_argument('--use_face_detection', action='store_true',
                        help='Use face detection for portrait similarity')
    parser.add_argument('--use_pose_estimation', action='store_true',
                        help='Use pose estimation for pose similarity')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./model_checkpoints/similarity',
                        help='Directory to save index and visualizations')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--limit_samples', type=int, default=None,
                        help='Limit number of samples (for testing)')
    parser.add_argument('--num_similar', type=int, default=5,
                        help='Number of similar paintings to visualize')
    
    return parser.parse_args()


def extract_features(model, data_loader, device):
    """
    Extract features from all paintings in the dataset.
    
    Args:
        model: Feature extractor model
        data_loader: Data loader
        device: Device to use
        
    Returns:
        Tuple of (features, painting_ids, metadata)
    """
    features = []
    painting_ids = []
    metadata = []
    
    with torch.no_grad():
        for batch_idx, (images, batch_metadata) in enumerate(tqdm(data_loader, desc="Extracting features")):
            # Extract features
            batch_features = model.extract_features(images.to(device))
            
            # Store features and metadata
            features.append(batch_features.cpu().numpy())
            
            for meta in batch_metadata:
                painting_ids.append(meta['id'])
                metadata.append(meta)
    
    # Concatenate features
    features = np.vstack(features)
    
    return features, painting_ids, metadata


def visualize_similar_paintings(model, data_loader, device, save_dir, num_similar=5):
    """
    Visualize similar paintings for a few examples.
    
    Args:
        model: Similarity detector model
        data_loader: Data loader
        device: Device to use
        save_dir: Directory to save visualizations
        num_similar: Number of similar paintings to visualize
    """
    # Extract features from all paintings
    logger.info("Extracting features for visualization...")
    features, painting_ids, metadata = extract_features(model, data_loader, device)
    
    # Build index
    logger.info("Building index for visualization...")
    model.build_index(features, painting_ids)
    
    # Select a few random paintings as queries
    num_queries = min(5, len(painting_ids))
    query_indices = np.random.choice(len(painting_ids), num_queries, replace=False)
    
    # Create visualization directory
    vis_dir = Path(save_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize similar paintings for each query
    for i, idx in enumerate(query_indices):
        query_id = painting_ids[idx]
        query_features = features[idx].reshape(1, -1)
        
        # Find similar paintings
        similar_ids, similarity_scores = model.find_similar(query_features, k=num_similar+1)
        
        # Skip the first result (which is the query itself)
        similar_ids = similar_ids[1:]
        similarity_scores = similarity_scores[1:]
        
        # Get metadata for query and similar paintings
        query_meta = next(meta for meta in metadata if meta['id'] == query_id)
        similar_meta = [next(meta for meta in metadata if meta['id'] == sim_id) for sim_id in similar_ids]
        
        # Load images
        query_img_path = Path(data_loader.dataset.data_dir) / query_meta['path']
        query_img = cv2.imread(str(query_img_path))
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        similar_imgs = []
        for meta in similar_meta:
            img_path = Path(data_loader.dataset.data_dir) / meta['path']
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            similar_imgs.append(img)
        
        # Visualize
        save_path = vis_dir / f'similar_paintings_{i+1}.png'
        model.visualize_similarity(
            query_img=query_img,
            similar_imgs=similar_imgs,
            similarity_scores=similarity_scores,
            save_path=save_path
        )
        
        # Log information
        logger.info(f"Query painting: {query_meta.get('title', 'Unknown')}")
        logger.info(f"Similar paintings:")
        for j, (meta, score) in enumerate(zip(similar_meta, similarity_scores)):
            logger.info(f"  {j+1}. {meta.get('title', 'Unknown')} (Similarity: {score:.3f})")
    
    logger.info(f"Visualizations saved to {vis_dir}")


def analyze_dataset(model, data_loader, device, save_dir):
    """
    Analyze the dataset and visualize feature space.
    
    Args:
        model: Similarity detector model
        data_loader: Data loader
        device: Device to use
        save_dir: Directory to save visualizations
    """
    # Extract features from all paintings
    logger.info("Extracting features for analysis...")
    features, painting_ids, metadata = extract_features(model, data_loader, device)
    
    # Create analysis directory
    analysis_dir = Path(save_dir) / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze similarity distribution
    logger.info("Analyzing similarity distribution...")
    stats = model.analyze_similarity_distribution(
        all_features=features,
        save_path=analysis_dir / 'similarity_distribution.png'
    )
    
    # Log statistics
    logger.info("Similarity statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Extract labels for visualization
    labels = []
    for meta in metadata:
        # Use classification or medium as label
        label = meta.get('classification', meta.get('medium', 'unknown'))
        labels.append(label)
    
    # Visualize feature space using PCA
    logger.info("Visualizing feature space using PCA...")
    model.visualize_feature_space(
        features=features,
        labels=labels,
        method='pca',
        save_path=analysis_dir / 'feature_space_pca.png'
    )
    
    # Visualize feature space using t-SNE (if not too many samples)
    if len(features) <= 1000:
        logger.info("Visualizing feature space using t-SNE...")
        model.visualize_feature_space(
            features=features,
            labels=labels,
            method='tsne',
            save_path=analysis_dir / 'feature_space_tsne.png'
        )
    
    logger.info(f"Analysis saved to {analysis_dir}")


def main():
    """Main function to train the painting similarity detector."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    limit_samples_dict = {'train': args.limit_samples, 'val': args.limit_samples // 5 if args.limit_samples else None}
    train_loader, val_loader, test_loader = create_national_gallery_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        limit_samples=limit_samples_dict
    )
    
    # Create model
    logger.info(f"Creating Painting Similarity Detector with {args.backbone} backbone...")
    model = PaintingSimilarityDetector(
        backbone=args.backbone,
        pretrained=args.pretrained,
        feature_layer=args.feature_layer,
        device=device,
        use_face_detection=args.use_face_detection,
        use_pose_estimation=args.use_pose_estimation
    )
    
    # Extract features and build index
    logger.info("Extracting features and building index...")
    features, painting_ids, metadata = extract_features(model, train_loader, device)
    
    # Save features and metadata
    logger.info("Saving features and metadata...")
    np.save(save_dir / 'features.npy', features)
    with open(save_dir / 'painting_ids.json', 'w') as f:
        json.dump(painting_ids, f)
    
    # Build and save index
    logger.info("Building and saving index...")
    model.build_index(features, painting_ids)
    
    # Save index
    faiss.write_index(model.faiss_index, str(save_dir / 'similarity_index.faiss'))
    
    # Visualize similar paintings
    logger.info("Visualizing similar paintings...")
    visualize_similar_paintings(
        model=model,
        data_loader=val_loader,
        device=device,
        save_dir=save_dir,
        num_similar=args.num_similar
    )
    
    # Analyze dataset
    logger.info("Analyzing dataset...")
    analyze_dataset(
        model=model,
        data_loader=train_loader,
        device=device,
        save_dir=save_dir
    )
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
