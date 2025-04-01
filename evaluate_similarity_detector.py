#!/usr/bin/env python3
"""
Evaluation script for Painting Similarity Detector.
This script evaluates a trained similarity detector on the test set and identifies outliers.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import faiss
import cv2
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import os

from models.similarity.painting_similarity import PaintingSimilarityDetector
from data.similarity.national_gallery_dataset import create_national_gallery_dataloaders

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Painting Similarity Detector')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to National Gallery dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='CNN backbone architecture')
    parser.add_argument('--feature_layer', type=str, default='avgpool',
                        help='Layer to extract features from')
    parser.add_argument('--use_face_detection', action='store_true',
                        help='Use face detection for portrait similarity')
    parser.add_argument('--use_pose_estimation', action='store_true',
                        help='Use pose estimation for pose similarity')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--index_file', type=str, required=True,
                        help='Path to similarity index file')
    parser.add_argument('--painting_ids_file', type=str, required=True,
                        help='Path to painting IDs file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results/similarity',
                        help='Directory to save evaluation results')
    
    # Evaluation arguments
    parser.add_argument('--num_queries', type=int, default=100,
                        help='Number of query paintings to evaluate')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top similar paintings to retrieve')
    parser.add_argument('--outlier_threshold', type=float, default=0.3,
                        help='Threshold for outlier detection')
    parser.add_argument('--num_outliers', type=int, default=10,
                        help='Number of outliers to visualize')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def evaluate_similarity_search(model, test_loader, device, top_k, num_queries, output_dir):
    """
    Evaluate similarity search performance.
    
    Args:
        model: Similarity detector model
        test_loader: Test data loader
        device: Device to use
        top_k: Number of top similar paintings to retrieve
        num_queries: Number of query paintings to evaluate
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Extract features from all paintings in the test set
    logger.info("Extracting features from test set...")
    all_features = []
    all_metadata = []
    
    with torch.no_grad():
        for batch_idx, (images, batch_metadata) in enumerate(tqdm(test_loader, desc="Extracting features")):
            # Extract features
            batch_features = model.extract_features(images.to(device))
            
            # Store features and metadata
            all_features.append(batch_features.cpu().numpy())
            all_metadata.extend(batch_metadata)
    
    # Concatenate features
    all_features = np.vstack(all_features)
    
    # Select random query paintings
    num_queries = min(num_queries, len(all_metadata))
    query_indices = np.random.choice(len(all_metadata), num_queries, replace=False)
    
    # Create directory for query visualizations
    query_dir = Path(output_dir) / 'query_results'
    query_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each query
    precision_at_k = []
    average_precision = []
    
    for i, idx in enumerate(tqdm(query_indices, desc="Evaluating queries")):
        query_features = all_features[idx].reshape(1, -1)
        query_metadata = all_metadata[idx]
        
        # Find similar paintings
        similar_indices, similarity_scores = model.find_similar_from_features(
            query_features, all_features, k=top_k+1)
        
        # Skip the first result (which is the query itself)
        similar_indices = similar_indices[1:]
        similarity_scores = similarity_scores[1:]
        
        # Get metadata for similar paintings
        similar_metadata = [all_metadata[j] for j in similar_indices]
        
        # Calculate precision based on matching attributes
        # We consider a match if at least one of these attributes matches
        relevant_attributes = ['artist', 'classification', 'medium', 'school']
        
        # Create binary relevance array
        relevance = []
        for sim_meta in similar_metadata:
            is_relevant = False
            for attr in relevant_attributes:
                if (attr in query_metadata and attr in sim_meta and 
                    query_metadata[attr] == sim_meta[attr] and 
                    query_metadata[attr] != ''):
                    is_relevant = True
                    break
            relevance.append(int(is_relevant))
        
        # Calculate precision@k
        precision = sum(relevance) / len(relevance)
        precision_at_k.append(precision)
        
        # Calculate average precision
        if sum(relevance) > 0:
            ap = average_precision_score(relevance, similarity_scores)
            average_precision.append(ap)
        
        # Visualize query results (for a subset of queries)
        if i < 10:  # Only visualize the first 10 queries
            # Load query image
            query_img_path = os.path.join(test_loader.dataset.data_dir, query_metadata['path'])
            query_img = cv2.imread(query_img_path)
            query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
            
            # Load similar images
            similar_imgs = []
            for meta in similar_metadata:
                img_path = os.path.join(test_loader.dataset.data_dir, meta['path'])
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                similar_imgs.append(img)
            
            # Visualize
            save_path = query_dir / f'query_{i+1}.png'
            model.visualize_similarity(
                query_img=query_img,
                similar_imgs=similar_imgs,
                similarity_scores=similarity_scores,
                save_path=save_path,
                relevance=relevance
            )
    
    # Calculate mean metrics
    mean_precision_at_k = np.mean(precision_at_k)
    mean_average_precision = np.mean(average_precision) if average_precision else 0
    
    return {
        'mean_precision_at_k': mean_precision_at_k,
        'mean_average_precision': mean_average_precision,
        'precision_at_k': precision_at_k,
        'average_precision': average_precision
    }


def identify_outliers(model, test_loader, device, threshold, num_outliers, output_dir):
    """
    Identify outlier paintings based on similarity scores.
    
    Args:
        model: Similarity detector model
        test_loader: Test data loader
        device: Device to use
        threshold: Threshold for outlier detection
        num_outliers: Number of outliers to visualize
        output_dir: Directory to save outlier visualizations
        
    Returns:
        List of outlier indices and scores
    """
    # Extract features from all paintings in the test set
    logger.info("Extracting features for outlier detection...")
    all_features = []
    all_metadata = []
    
    with torch.no_grad():
        for batch_idx, (images, batch_metadata) in enumerate(tqdm(test_loader, desc="Extracting features")):
            # Extract features
            batch_features = model.extract_features(images.to(device))
            
            # Store features and metadata
            all_features.append(batch_features.cpu().numpy())
            all_metadata.extend(batch_metadata)
    
    # Concatenate features
    all_features = np.vstack(all_features)
    
    # Calculate average similarity to nearest neighbors
    logger.info("Calculating outlier scores...")
    outlier_scores = []
    
    for i, features in enumerate(tqdm(all_features, desc="Calculating outlier scores")):
        # Reshape features
        query_features = features.reshape(1, -1)
        
        # Find similar paintings
        _, similarity_scores = model.find_similar_from_features(
            query_features, all_features, k=11)
        
        # Skip the first result (which is the query itself)
        similarity_scores = similarity_scores[1:11]
        
        # Calculate average similarity
        avg_similarity = np.mean(similarity_scores)
        
        # Calculate outlier score (lower similarity = higher outlier score)
        outlier_score = 1 - avg_similarity
        outlier_scores.append(outlier_score)
    
    # Find outliers
    outlier_indices = np.argsort(outlier_scores)[::-1]
    sorted_outlier_scores = np.array(outlier_scores)[outlier_indices]
    
    # Filter by threshold
    threshold_outlier_indices = outlier_indices[sorted_outlier_scores > threshold]
    threshold_outlier_scores = sorted_outlier_scores[sorted_outlier_scores > threshold]
    
    # Create directory for outlier visualizations
    outlier_dir = Path(output_dir) / 'outliers'
    outlier_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize outliers
    num_to_visualize = min(num_outliers, len(threshold_outlier_indices))
    
    for i in range(num_to_visualize):
        idx = threshold_outlier_indices[i]
        score = threshold_outlier_scores[i]
        metadata = all_metadata[idx]
        
        # Load outlier image
        img_path = os.path.join(test_loader.dataset.data_dir, metadata['path'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Outlier {i+1}\nTitle: {metadata.get('title', 'Unknown')}\nOutlier Score: {score:.4f}")
        plt.axis('off')
        
        # Save
        save_path = outlier_dir / f'outlier_{i+1}.png'
        plt.savefig(save_path)
        plt.close()
    
    logger.info(f"Found {len(threshold_outlier_indices)} outliers with threshold {threshold}")
    logger.info(f"Visualized {num_to_visualize} outliers")
    
    return {
        'outlier_indices': threshold_outlier_indices.tolist(),
        'outlier_scores': threshold_outlier_scores.tolist()
    }


def visualize_feature_space(model, test_loader, device, output_dir):
    """
    Visualize feature space using dimensionality reduction.
    
    Args:
        model: Similarity detector model
        test_loader: Test data loader
        device: Device to use
        output_dir: Directory to save visualizations
    """
    # Extract features from all paintings in the test set
    logger.info("Extracting features for visualization...")
    all_features = []
    all_metadata = []
    
    with torch.no_grad():
        for batch_idx, (images, batch_metadata) in enumerate(tqdm(test_loader, desc="Extracting features")):
            # Extract features
            batch_features = model.extract_features(images.to(device))
            
            # Store features and metadata
            all_features.append(batch_features.cpu().numpy())
            all_metadata.extend(batch_metadata)
    
    # Concatenate features
    all_features = np.vstack(all_features)
    
    # Create directory for visualizations
    vis_dir = Path(output_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract labels for visualization
    # We'll use different attributes for coloring
    for attr in ['classification', 'medium', 'school']:
        # Skip if attribute is not available
        if not all(attr in meta for meta in all_metadata):
            continue
        
        # Get labels
        labels = [meta.get(attr, 'unknown') for meta in all_metadata]
        
        # Count occurrences of each label
        label_counts = {}
        for label in labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        
        # Keep only labels with at least 5 occurrences
        common_labels = {label for label, count in label_counts.items() if count >= 5}
        
        # Replace uncommon labels with 'other'
        labels = ['other' if label not in common_labels else label for label in labels]
        
        # Visualize using PCA
        logger.info(f"Visualizing feature space using PCA (colored by {attr})...")
        model.visualize_feature_space(
            features=all_features,
            labels=labels,
            method='pca',
            save_path=vis_dir / f'feature_space_pca_{attr}.png',
            title=f'Feature Space (PCA) - Colored by {attr.capitalize()}'
        )
        
        # Visualize using t-SNE (if not too many samples)
        if len(all_features) <= 1000:
            logger.info(f"Visualizing feature space using t-SNE (colored by {attr})...")
            model.visualize_feature_space(
                features=all_features,
                labels=labels,
                method='tsne',
                save_path=vis_dir / f'feature_space_tsne_{attr}.png',
                title=f'Feature Space (t-SNE) - Colored by {attr.capitalize()}'
            )
    
    logger.info(f"Visualizations saved to {vis_dir}")


def main():
    """Main function to evaluate the painting similarity detector."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'eval_args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    _, _, test_loader = create_national_gallery_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    logger.info(f"Creating Painting Similarity Detector with {args.backbone} backbone...")
    model = PaintingSimilarityDetector(
        backbone=args.backbone,
        pretrained=False,  # No need for pretrained weights when loading checkpoint
        feature_layer=args.feature_layer,
        device=device,
        use_face_detection=args.use_face_detection,
        use_pose_estimation=args.use_pose_estimation
    )
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load index
    logger.info(f"Loading index: {args.index_file}")
    model.faiss_index = faiss.read_index(args.index_file)
    
    # Load painting IDs
    logger.info(f"Loading painting IDs: {args.painting_ids_file}")
    with open(args.painting_ids_file, 'r') as f:
        model.painting_ids = json.load(f)
    
    # Evaluate similarity search
    logger.info("Evaluating similarity search...")
    similarity_metrics = evaluate_similarity_search(
        model=model,
        test_loader=test_loader,
        device=device,
        top_k=args.top_k,
        num_queries=args.num_queries,
        output_dir=output_dir
    )
    
    # Log similarity metrics
    logger.info("Similarity search metrics:")
    logger.info(f"  Mean Precision@{args.top_k}: {similarity_metrics['mean_precision_at_k']:.4f}")
    logger.info(f"  Mean Average Precision: {similarity_metrics['mean_average_precision']:.4f}")
    
    # Save similarity metrics
    with open(output_dir / 'similarity_metrics.json', 'w') as f:
        json.dump(similarity_metrics, f, indent=4)
    
    # Identify outliers
    logger.info("Identifying outliers...")
    outlier_results = identify_outliers(
        model=model,
        test_loader=test_loader,
        device=device,
        threshold=args.outlier_threshold,
        num_outliers=args.num_outliers,
        output_dir=output_dir
    )
    
    # Save outlier results
    with open(output_dir / 'outlier_results.json', 'w') as f:
        json.dump(outlier_results, f, indent=4)
    
    # Visualize feature space
    logger.info("Visualizing feature space...")
    visualize_feature_space(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir
    )
    
    logger.info(f"Evaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
