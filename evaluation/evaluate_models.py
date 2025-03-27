"""
Unified evaluation script for ArtExtract models.
This script can evaluate both classification and similarity models.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from tqdm import tqdm
import pickle

# Add parent directory to path to import from project
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.style_classification.cnn_rnn_model import CNNRNNModel
from models.similarity_detection.similarity_model import (
    create_similarity_model, PaintingSimilaritySystem
)
from data.preprocessing.art_dataset_loader import WikiArtDataset, NationalGalleryDataset
from data.preprocessing.data_preprocessing import ArtDatasetPreprocessor, create_data_loaders
from evaluation.classification_metrics import calculate_metrics, plot_confusion_matrix, plot_roc_curve
from evaluation.similarity_metrics import (
    calculate_precision_at_k, calculate_mean_average_precision,
    calculate_ndcg_at_k, calculate_mean_reciprocal_rank,
    plot_precision_recall_curve, plot_similarity_distribution
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate ArtExtract models')
    
    # Common parameters
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['classification', 'similarity'],
                        help='Type of model to evaluate')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    
    # Classification model parameters
    parser.add_argument('--data_dir', type=str,
                        help='Directory containing the dataset (required for classification)')
    parser.add_argument('--category', type=str, default='style',
                        choices=['style', 'artist', 'genre'],
                        help='Classification category (for classification model)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    
    # Similarity model parameters
    parser.add_argument('--features_path', type=str,
                        help='Path to the features file (required for similarity)')
    parser.add_argument('--ground_truth_path', type=str,
                        help='Path to ground truth similarity data (optional for similarity)')
    parser.add_argument('--k_values', type=str, default='1,5,10',
                        help='Comma-separated list of k values for precision@k and NDCG@k')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    return parser.parse_args()


def evaluate_classification_model(args):
    """Evaluate a classification model."""
    # Create output directory
    output_dir = Path(args.output_dir) / 'classification'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading classification model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model_args = checkpoint['args']
    
    # Create model with same architecture
    model = CNNRNNModel(
        num_classes=model_args.get('num_classes', 0),  # Will be updated based on dataset
        cnn_backbone=model_args.get('cnn_backbone', 'resnet50'),
        hidden_size=model_args.get('hidden_size', 256),
        dropout=model_args.get('dropout', 0.5)
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    # Load dataset
    logger.info(f"Loading WikiArt dataset from {args.data_dir}")
    dataset = WikiArtDataset(args.data_dir)
    df = dataset.create_dataframe(category=args.category)
    
    # Update model's num_classes if needed
    num_classes = len(df['label'].unique())
    if model.fc.out_features != num_classes:
        logger.warning(f"Model's output dimension ({model.fc.out_features}) doesn't match dataset classes ({num_classes})")
    
    # Create data preprocessor and loaders
    logger.info("Creating data loaders")
    preprocessor = ArtDatasetPreprocessor(img_size=(224, 224), use_augmentation=False)
    dataloaders = create_data_loaders(
        dataframe=df,
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    
    criterion = torch.nn.CrossEntropyLoss()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloaders['test'], desc='Evaluation')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / total,
                'acc': 100 * correct / total
            })
    
    test_loss = running_loss / total
    test_acc = 100 * correct / total
    
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Get class names
    if args.category == 'style':
        idx_to_class = dataset.idx_to_style
    elif args.category == 'artist':
        idx_to_class = dataset.idx_to_artist
    else:  # genre
        idx_to_class = dataset.idx_to_genre
    
    class_names = [idx_to_class[i] for i in range(num_classes)]
    
    # Calculate and save metrics
    metrics = calculate_metrics(all_labels, all_predictions, all_probabilities, list(range(num_classes)))
    metrics['test_loss'] = test_loss
    metrics['test_accuracy'] = test_acc
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        all_labels, all_predictions, 
        class_names=class_names,
        output_path=output_dir / 'confusion_matrix.png'
    )
    
    # Plot ROC curve
    plot_roc_curve(
        all_labels, all_probabilities, 
        class_names=class_names,
        output_path=output_dir / 'roc_curve.png'
    )
    
    # Plot per-class accuracy
    plt.figure(figsize=(12, 8))
    per_class_acc = metrics['per_class_metrics']
    class_acc = [per_class_acc[str(i)]['accuracy'] for i in range(num_classes)]
    
    # Sort by accuracy
    sorted_indices = np.argsort(class_acc)
    sorted_class_names = [class_names[i] for i in sorted_indices]
    sorted_class_acc = [class_acc[i] for i in sorted_indices]
    
    plt.barh(range(len(sorted_class_names)), sorted_class_acc)
    plt.yticks(range(len(sorted_class_names)), sorted_class_names)
    plt.xlabel('Accuracy')
    plt.title(f'Per-class Accuracy ({args.category.capitalize()})')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_accuracy.png')
    
    logger.info(f"Evaluation results saved to {output_dir}")
    
    return metrics


def evaluate_similarity_model(args):
    """Evaluate a similarity model."""
    # Create output directory
    output_dir = Path(args.output_dir) / 'similarity'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features
    logger.info(f"Loading features from {args.features_path}")
    with open(args.features_path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    image_paths = data['image_paths']
    
    # Load model
    logger.info(f"Loading similarity model from {args.model_path}")
    with open(args.model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model_type = model_data.get('model_type', 'cosine')
    feature_dim = features.shape[1]
    
    # Create similarity model
    similarity_model = create_similarity_model(model_type, feature_dim=feature_dim)
    
    # Load model parameters if applicable
    if hasattr(similarity_model, 'load_state'):
        similarity_model.load_state(model_data.get('model_state', {}))
    
    # Create similarity system
    similarity_system = PaintingSimilaritySystem(
        similarity_model=similarity_model,
        features=features,
        image_paths=image_paths
    )
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(',')]
    max_k = max(k_values)
    
    # Evaluate model
    logger.info(f"Evaluating similarity model with k values: {k_values}")
    
    # If ground truth is available, calculate metrics
    metrics = {}
    if args.ground_truth_path and os.path.exists(args.ground_truth_path):
        logger.info(f"Loading ground truth from {args.ground_truth_path}")
        with open(args.ground_truth_path, 'rb') as f:
            ground_truth = pickle.load(f)
        
        # Calculate metrics
        precision_at_k = {}
        ndcg_at_k = {}
        
        for k in k_values:
            precision_at_k[k] = calculate_precision_at_k(
                similarity_system, ground_truth, k=k
            )
            ndcg_at_k[k] = calculate_ndcg_at_k(
                similarity_system, ground_truth, k=k
            )
        
        map_score = calculate_mean_average_precision(
            similarity_system, ground_truth, k=max_k
        )
        
        mrr_score = calculate_mean_reciprocal_rank(
            similarity_system, ground_truth
        )
        
        metrics = {
            'precision_at_k': precision_at_k,
            'ndcg_at_k': ndcg_at_k,
            'mean_average_precision': map_score,
            'mean_reciprocal_rank': mrr_score
        }
        
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Plot precision-recall curve
        plot_precision_recall_curve(
            similarity_system, ground_truth,
            output_path=output_dir / 'precision_recall_curve.png'
        )
    
    # Generate similarity distribution
    logger.info("Generating similarity distribution")
    
    # Sample random pairs for distribution
    num_samples = min(1000, len(features))
    indices = np.random.choice(len(features), size=num_samples, replace=False)
    
    similarities = []
    for i in tqdm(range(len(indices))):
        for j in range(i+1, len(indices)):
            idx1, idx2 = indices[i], indices[j]
            sim = similarity_system.calculate_similarity(idx1, idx2)
            similarities.append(sim)
    
    # Plot similarity distribution
    plot_similarity_distribution(
        similarities,
        output_path=output_dir / 'similarity_distribution.png'
    )
    
    # Generate example similar paintings
    logger.info("Generating example similar paintings")
    
    # Sample random queries
    num_queries = min(5, len(features))
    query_indices = np.random.choice(len(features), size=num_queries, replace=False)
    
    example_results = {}
    for i, query_idx in enumerate(query_indices):
        result = similarity_system.find_similar_paintings(query_idx, k=max_k)
        example_results[f'query_{i}'] = {
            'query_path': image_paths[query_idx],
            'similar_paths': [image_paths[idx] for idx in result['indices']],
            'similarities': result['similarities'].tolist()
        }
    
    with open(output_dir / 'example_results.json', 'w') as f:
        json.dump(example_results, f, indent=4)
    
    logger.info(f"Evaluation results saved to {output_dir}")
    
    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(Path(args.output_dir) / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Evaluate model based on type
    if args.model_type == 'classification':
        if not args.data_dir:
            logger.error("--data_dir is required for classification model evaluation")
            return
        
        metrics = evaluate_classification_model(args)
    else:  # similarity
        if not args.features_path:
            logger.error("--features_path is required for similarity model evaluation")
            return
        
        metrics = evaluate_similarity_model(args)
    
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
