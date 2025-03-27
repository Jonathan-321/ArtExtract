"""
Demo script for CNN-RNN model for art style/artist/genre classification.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from tqdm import tqdm
import random

# Add parent directory to path to import from project
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.style_classification.cnn_rnn_model import CNNRNNModel
from data.preprocessing.art_dataset_loader import WikiArtDataset
from data.preprocessing.data_preprocessing import ArtDatasetPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Demo for art classification model')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the WikiArt dataset')
    
    # Demo parameters
    parser.add_argument('--category', type=str, default='style',
                        choices=['style', 'artist', 'genre'],
                        help='Classification category (style, artist, or genre)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of random samples to classify')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to a specific image to classify (overrides random sampling)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./demo_outputs',
                        help='Directory to save demo results')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference (cuda or cpu)')
    
    return parser.parse_args()


def load_model(model_path, device):
    """Load the trained model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    args = checkpoint['args']
    
    # Create model with same architecture
    model = CNNRNNModel(
        num_classes=args.get('num_classes', 0),  # Will be updated below
        cnn_backbone=args.get('cnn_backbone', 'resnet50'),
        hidden_size=args.get('hidden_size', 256),
        dropout=args.get('dropout', 0.5)
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, args


def classify_image(model, image_path, preprocessor, device, top_k=3):
    """Classify a single image and return top-k predictions."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = preprocessor.get_transforms(is_training=False)
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, k=top_k)
    
    return {
        'top_indices': top_indices.cpu().numpy(),
        'top_probs': top_probs.cpu().numpy(),
        'image': image
    }


def display_prediction(result, class_names, image_path, category):
    """Display the image and prediction results."""
    # Display image
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(result['image'])
    plt.title(f"Image: {os.path.basename(image_path)}")
    plt.axis('off')
    
    # Display predictions
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(result['top_indices']))
    plt.barh(y_pos, result['top_probs'] * 100)
    plt.yticks(y_pos, [class_names[idx] for idx in result['top_indices']])
    plt.xlabel('Probability (%)')
    plt.title(f"Top {category.capitalize()} Predictions")
    
    plt.tight_layout()
    return plt.gcf()


def interactive_mode(model, dataset, preprocessor, device, category, output_dir):
    """Run the demo in interactive mode."""
    # Create mapping from index to class name
    if category == 'style':
        idx_to_class = dataset.idx_to_style
    elif category == 'artist':
        idx_to_class = dataset.idx_to_artist
    else:  # genre
        idx_to_class = dataset.idx_to_genre
    
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    while True:
        # Get user input
        print("\nInteractive Art Classification Demo")
        print("----------------------------------")
        print("Options:")
        print("1. Classify a random image from the dataset")
        print("2. Classify an image from a specific path")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            # Get random image from dataset
            df = dataset.create_dataframe(category=category)
            random_idx = random.randint(0, len(df) - 1)
            image_path = df.iloc[random_idx]['image_path']
            true_label = df.iloc[random_idx]['label']
            true_class = idx_to_class[true_label]
            
            # Classify image
            result = classify_image(model, image_path, preprocessor, device)
            
            # Display results
            print(f"\nTrue {category}: {true_class}")
            print(f"Top predictions:")
            for i, (idx, prob) in enumerate(zip(result['top_indices'], result['top_probs'])):
                print(f"{i+1}. {idx_to_class[idx]}: {prob*100:.2f}%")
            
            # Save and display figure
            fig = display_prediction(result, class_names, image_path, category)
            plt.show()
            
        elif choice == '2':
            # Get image path from user
            image_path = input("Enter the path to the image: ")
            if not os.path.exists(image_path):
                print(f"Error: File {image_path} does not exist.")
                continue
            
            # Classify image
            try:
                result = classify_image(model, image_path, preprocessor, device)
                
                # Display results
                print(f"\nTop predictions:")
                for i, (idx, prob) in enumerate(zip(result['top_indices'], result['top_probs'])):
                    print(f"{i+1}. {idx_to_class[idx]}: {prob*100:.2f}%")
                
                # Save and display figure
                fig = display_prediction(result, class_names, image_path, category)
                plt.show()
                
            except Exception as e:
                print(f"Error classifying image: {e}")
            
        elif choice == '3':
            print("Exiting demo.")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def main():
    """Main demo function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model, model_args = load_model(args.model_path, args.device)
    
    # Load dataset
    logger.info(f"Loading WikiArt dataset from {args.data_dir}")
    dataset = WikiArtDataset(args.data_dir)
    
    # Create preprocessor
    preprocessor = ArtDatasetPreprocessor(img_size=(224, 224), use_augmentation=False)
    
    # Create mapping from index to class name
    if args.category == 'style':
        idx_to_class = dataset.idx_to_style
    elif args.category == 'artist':
        idx_to_class = dataset.idx_to_artist
    else:  # genre
        idx_to_class = dataset.idx_to_genre
    
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Interactive mode
    if args.interactive:
        interactive_mode(model, dataset, preprocessor, args.device, args.category, output_dir)
        return
    
    # Classify specific image
    if args.image_path:
        logger.info(f"Classifying image: {args.image_path}")
        result = classify_image(model, args.image_path, preprocessor, args.device)
        
        # Display results
        logger.info(f"Top predictions:")
        for i, (idx, prob) in enumerate(zip(result['top_indices'], result['top_probs'])):
            logger.info(f"{i+1}. {idx_to_class[idx]}: {prob*100:.2f}%")
        
        # Save figure
        fig = display_prediction(result, class_names, args.image_path, args.category)
        fig.savefig(output_dir / f"prediction_{os.path.basename(args.image_path)}.png")
        plt.close(fig)
        
    # Classify random samples
    else:
        logger.info(f"Classifying {args.num_samples} random samples")
        df = dataset.create_dataframe(category=args.category)
        
        # Select random samples
        random_indices = random.sample(range(len(df)), min(args.num_samples, len(df)))
        
        for i, idx in enumerate(random_indices):
            image_path = df.iloc[idx]['image_path']
            true_label = df.iloc[idx]['label']
            true_class = idx_to_class[true_label]
            
            logger.info(f"Sample {i+1}/{args.num_samples}: {os.path.basename(image_path)}")
            logger.info(f"True {args.category}: {true_class}")
            
            # Classify image
            result = classify_image(model, image_path, preprocessor, args.device)
            
            # Display results
            logger.info(f"Top predictions:")
            for j, (idx, prob) in enumerate(zip(result['top_indices'], result['top_probs'])):
                logger.info(f"{j+1}. {idx_to_class[idx]}: {prob*100:.2f}%")
            
            # Save figure
            fig = display_prediction(result, class_names, image_path, args.category)
            fig.savefig(output_dir / f"prediction_sample_{i+1}.png")
            plt.close(fig)
    
    logger.info(f"Demo completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
