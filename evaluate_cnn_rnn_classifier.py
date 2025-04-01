#!/usr/bin/env python3
"""
Evaluation script for CNN-RNN art classifier.
This script evaluates a trained CNN-RNN model on the test set and identifies outliers.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from models.classification.cnn_rnn_classifier import CNNRNNClassifier
from data.classification.wikiart_dataset import create_wikiart_dataloaders

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate CNN-RNN art classifier')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to WikiArt dataset')
    parser.add_argument('--attributes', type=str, nargs='+', default=['style', 'artist', 'genre'],
                        help='Art attributes to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='CNN backbone architecture')
    parser.add_argument('--rnn_hidden_size', type=int, default=512,
                        help='Hidden size of RNN layers')
    parser.add_argument('--rnn_num_layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    
    # Outlier detection arguments
    parser.add_argument('--outlier_method', type=str, default='softmax_uncertainty',
                        choices=['softmax_uncertainty', 'entropy'],
                        help='Method for outlier detection')
    parser.add_argument('--outlier_threshold', type=float, default=0.5,
                        help='Threshold for outlier detection')
    parser.add_argument('--num_outliers', type=int, default=10,
                        help='Number of outliers to visualize')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def evaluate_model(model, data_loader, device, attributes):
    """
    Evaluate model on the dataset.
    
    Args:
        model: CNN-RNN classifier model
        data_loader: Data loader
        device: Device to use
        attributes: List of attributes to evaluate
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Initialize metrics
    correct = {attr: 0 for attr in attributes}
    total = 0
    
    # Initialize lists for confusion matrix
    all_preds = {attr: [] for attr in attributes}
    all_targets = {attr: [] for attr in attributes}
    
    # Initialize lists for outlier detection
    all_outlier_scores = {attr: [] for attr in attributes}
    all_sample_indices = []
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
            data = data.to(device)
            # Move each target tensor to device
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            outputs = model(data)
            
            # Calculate accuracy for each attribute
            for attr in attributes:
                if attr not in targets:
                    continue
                    
                _, predicted = outputs[attr].max(1)
                correct[attr] += predicted.eq(targets[attr]).sum().item()
                
                # Store predictions and targets for confusion matrix
                all_preds[attr].extend(predicted.cpu().numpy())
                all_targets[attr].extend(targets[attr].cpu().numpy())
            
            # Calculate outlier scores
            outlier_scores = model.get_outlier_scores(data, method='softmax_uncertainty')
            
            # Store outlier scores and sample indices
            for attr in attributes:
                if attr not in outlier_scores:
                    continue
                    
                all_outlier_scores[attr].extend(outlier_scores[attr].cpu().numpy())
            
            # Store sample indices
            batch_indices = list(range(batch_idx * data_loader.batch_size, 
                                       min((batch_idx + 1) * data_loader.batch_size, 
                                           len(data_loader.dataset))))
            all_sample_indices.extend(batch_indices)
            
            total += data.size(0)
    
    # Calculate accuracy
    accuracy = {attr: 100. * correct[attr] / total for attr in attributes if attr in correct}
    
    # Create confusion matrices
    confusion_matrices = {}
    for attr in attributes:
        if attr not in all_preds or attr not in all_targets:
            continue
            
        confusion_matrices[attr] = confusion_matrix(all_targets[attr], all_preds[attr])
    
    # Create classification reports
    classification_reports = {}
    for attr in attributes:
        if attr not in all_preds or attr not in all_targets:
            continue
            
        classification_reports[attr] = classification_report(
            all_targets[attr], all_preds[attr], output_dict=True)
    
    # Find outliers
    outliers = {}
    for attr in attributes:
        if attr not in all_outlier_scores:
            continue
            
        # Sort by outlier score (descending)
        sorted_indices = np.argsort(all_outlier_scores[attr])[::-1]
        outlier_indices = [all_sample_indices[i] for i in sorted_indices]
        outlier_scores = [all_outlier_scores[attr][i] for i in sorted_indices]
        
        outliers[attr] = {
            'indices': outlier_indices,
            'scores': outlier_scores
        }
    
    return {
        'accuracy': accuracy,
        'confusion_matrices': confusion_matrices,
        'classification_reports': classification_reports,
        'outliers': outliers
    }


def plot_confusion_matrix(cm, class_names, title, save_path):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def visualize_outliers(data_loader, outliers, attribute, num_outliers, save_dir):
    """
    Visualize outlier samples.
    
    Args:
        data_loader: Data loader
        outliers: Dictionary of outlier information
        attribute: Attribute to visualize outliers for
        num_outliers: Number of outliers to visualize
        save_dir: Directory to save visualizations
    """
    if attribute not in outliers:
        logger.warning(f"No outliers found for attribute: {attribute}")
        return
    
    # Get dataset and class mapping
    dataset = data_loader.dataset
    idx_to_class = dataset.idx_to_class[attribute]
    
    # Create directory for outlier visualizations
    outlier_dir = Path(save_dir) / f'outliers_{attribute}'
    outlier_dir.mkdir(parents=True, exist_ok=True)
    
    # Get outlier indices and scores
    outlier_indices = outliers[attribute]['indices'][:num_outliers]
    outlier_scores = outliers[attribute]['scores'][:num_outliers]
    
    # Visualize each outlier
    for i, (idx, score) in enumerate(zip(outlier_indices, outlier_scores)):
        # Get sample
        img, labels = dataset[idx]
        
        # Convert tensor to numpy for visualization
        img_np = img.permute(1, 2, 0).numpy()
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Get true label
        true_label = idx_to_class[labels[attribute].item()]
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.imshow(img_np)
        plt.title(f"Outlier {i+1}\nAttribute: {attribute}\nTrue Label: {true_label}\nOutlier Score: {score:.4f}")
        plt.axis('off')
        
        # Save
        save_path = outlier_dir / f'outlier_{i+1}.png'
        plt.savefig(save_path)
        plt.close()
    
    logger.info(f"Outlier visualizations saved to {outlier_dir}")


def main():
    """Main function to evaluate the CNN-RNN classifier."""
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
    _, _, test_loader, _ = create_wikiart_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        attributes=args.attributes
    )
    
    # Get number of classes for each attribute
    num_classes = {}
    for attr in args.attributes:
        num_classes[attr] = len(test_loader.dataset.class_to_idx[attr])
    
    logger.info(f"Number of classes: {num_classes}")
    
    # Create model
    logger.info(f"Creating CNN-RNN classifier with {args.backbone} backbone...")
    model = CNNRNNClassifier(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=False,  # No need for pretrained weights when loading checkpoint
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    logger.info("Evaluating model...")
    eval_results = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        attributes=args.attributes
    )
    
    # Save evaluation results
    with open(output_dir / 'eval_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_json = {
            'accuracy': eval_results['accuracy'],
            'classification_reports': eval_results['classification_reports'],
            'outliers': {
                attr: {
                    'indices': outliers['indices'][:100].tolist(),  # Limit to 100 outliers
                    'scores': outliers['scores'][:100].tolist()
                }
                for attr, outliers in eval_results['outliers'].items()
            }
        }
        json.dump(results_json, f, indent=4)
    
    # Log accuracy
    logger.info("Evaluation results:")
    for attr, acc in eval_results['accuracy'].items():
        logger.info(f"  {attr.capitalize()} Accuracy: {acc:.2f}%")
    
    # Plot confusion matrices
    for attr in args.attributes:
        if attr not in eval_results['confusion_matrices']:
            continue
            
        cm = eval_results['confusion_matrices'][attr]
        class_names = [test_loader.dataset.idx_to_class[attr][i] for i in range(len(cm))]
        
        # If too many classes, use indices instead of names
        if len(class_names) > 20:
            class_names = [str(i) for i in range(len(cm))]
        
        plot_confusion_matrix(
            cm=cm,
            class_names=class_names,
            title=f'Confusion Matrix - {attr.capitalize()}',
            save_path=output_dir / f'confusion_matrix_{attr}.png'
        )
    
    # Visualize outliers
    for attr in args.attributes:
        visualize_outliers(
            data_loader=test_loader,
            outliers=eval_results['outliers'],
            attribute=attr,
            num_outliers=args.num_outliers,
            save_dir=output_dir
        )
    
    logger.info(f"Evaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
