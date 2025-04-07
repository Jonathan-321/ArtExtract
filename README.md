# ArtExtract

<div align="center">

![ArtExtract Logo](https://img.shields.io/badge/ArtExtract-Deep%20Learning%20for%20Art%20Analysis-blue?style=for-the-badge)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

## 🎨 Project Overview

ArtExtract is an advanced deep learning framework for art analysis that combines computer vision and neural network architectures to understand and classify artistic content. The system employs state-of-the-art CNN-RNN hybrid models to perform two primary tasks:

<div align="center">
<table>
<tr>
<td width="50%">

### 🖼️ Task 1: Style/Artist/Genre Classification

- **Architecture**: CNN-RNN hybrid with attention mechanisms
- **Dataset**: Test dataset with Renaissance, Baroque, and Impressionism paintings
- **Features**:
  - Multiple CNN backbones (ResNet18, ResNet50)
  - Bidirectional RNN layers (GRU)
  - Spatial attention mechanism
  - Outlier detection using softmax uncertainty
  - Comprehensive evaluation metrics

</td>
<td width="50%">

### 🔍 Task 2: Painting Similarity Detection

- **Architecture**: Feature extraction + similarity indexing
- **Dataset**: National Gallery of Art collection
- **Features**:
  - Deep feature extraction (CNN/CLIP)
  - Multiple similarity metrics
  - Faiss indexing for efficient search
  - Interactive visualization
  - Comprehensive evaluation framework

</td>
</tr>
</table>
</div>

## 🏗️ Architecture

<div align="center">

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ArtExtract System                        │
├─────────────────────────────┬───────────────────────────────────┤
│  Style/Artist Classification│     Painting Similarity Detection │
├─────────────────────────────┼───────────────────────────────────┤
│ ┌─────────────────────────┐ │ ┌─────────────────────────────┐   │
│ │      CNN Backbone       │ │ │    Feature Extraction       │   │
│ │  ResNet/EfficientNet    │ │ │    CNN/CLIP Models          │   │
│ └───────────┬─────────────┘ │ └───────────┬─────────────────┘   │
│             ▼               │             ▼                     │
│ ┌─────────────────────────┐ │ ┌─────────────────────────────┐   │
│ │   Feature Processing    │ │ │    Similarity Computation   │   │
│ │  Attention Mechanism    │ │ │    Cosine/Faiss Index       │   │
│ └───────────┬─────────────┘ │ └───────────┬─────────────────┘   │
│             ▼               │             ▼                     │
│ ┌─────────────────────────┐ │ ┌─────────────────────────────┐   │
│ │      RNN Layers         │ │ │      Ranking Engine         │   │
│ │   LSTM/GRU/Bidirectional│ │ │      Top-K Results          │   │
│ └───────────┬─────────────┘ │ └───────────┬─────────────────┘   │
│             ▼               │             ▼                     │
│ ┌─────────────────────────┐ │ │    Interactive Results      │   │
│ │   Classification Head   │ │ │    Visualization UI         │   │
│ │   Style/Artist/Genre    │ │ │    Similarity System        │   │
│ └─────────────────────────┘ │ └─────────────────────────────┘   │
└─────────────────────────────┴───────────────────────────────────┘
```

### CNN-RNN Classification Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Style/Artist/Genre Classification System             │
├───────────┬─────────────┬────────────────┬────────────────┬─────────────┤
│           │             │                │                │             │
│  Input    │    CNN      │   Feature      │     RNN        │  Output     │
│  Image    │  Backbone   │   Processing   │    Layer       │  Classes    │
│           │             │                │                │             │
└─────┬─────┴──────┬──────┴────────┬───────┴────────┬───────┴──────┬──────┘
      │            │               │                │              │
      ▼            ▼               ▼                ▼              ▼
┌──────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐
│ 224x224  │ │• ResNet50    │ │• Spatial     │ │• GRU         │ │• Style  │
│ RGB      │ │• ResNet18    │ │• EfficientNet│ │• Attention   │ │• Artist│
│ Artwork  │ │• MobileNetV2 │ │• Attention   │ │• Weights     │ │• Genre  │
│ Image    │ │• Custom      │ │• Mechanism   │ │• L2 Distance │ │• Softmax│
└──────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └─────────┘
```

### Painting Similarity Detection System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Painting Similarity Detection System                 │
├───────────┬─────────────┬────────────────┬────────────────┬─────────────┤
│           │             │                │                │             │
│  Query    │  Feature    │   Similarity   │    Ranking     │  Retrieved  │
│  Painting │  Extraction │   Computation  │    Engine      │  Paintings  │
│           │             │                │                │             │
└─────┬─────┴──────┬──────┴────────┬───────┴────────┬───────┴──────┬──────┘
      │            │               │                │              │
      ▼            ▼               ▼                ▼              ▼
┌──────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐
│ Input    │ │• ResNet50    │ │• Cosine      │ │• Top-K       │ │• Similar │
│ Artwork  │ │• VGG16       │ │• Similarity  │ │  Results     │ │  Artwork │
│ Image    │ │• CLIP        │ │• Faiss Index │ │• Confidence  │ │• Ranked  │
│          │ │• Custom      │ │• L2 Distance │ │• Scores      │ │• Results │
└──────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └─────────┘
              │
              ▼
        ┌──────────────┐
        │  Feature     │
        │  Database    │
        └──────────────┘
```

</div>

## 🔧 Technical Implementation

### CNN-RNN Classification Model

The CNN-RNN classification model combines the spatial feature extraction capabilities of CNNs with the sequential modeling power of RNNs:

1. **CNN Backbone**: Extracts rich visual features from artwork images
   - Implemented with ResNet18 for faster training and evaluation
   - Pretrained on ImageNet and fine-tuned on art datasets
   - Outputs feature maps that capture artistic elements

2. **Feature Processing**: Transforms CNN features for RNN consumption
   - Reshapes spatial features to sequential format
   - Spatial attention mechanism to focus on discriminative regions
   - Maintains spatial relationships in feature representation

3. **RNN Layer**: Processes sequential information in the feature maps
   - Bidirectional GRU cells with dropout for regularization
   - Captures spatial relationships between features
   - Attention mechanism for highlighting important features

4. **Classification Layer**: Produces final style/artist/genre predictions
   - Fully-connected layer with softmax activation
   - Multi-class classification with confidence scores
   - Outlier detection for identifying unusual artwork

### Outlier Detection

Our system implements the softmax uncertainty method to identify paintings that don't fit well into their assigned categories:

1. **Softmax Uncertainty**: 1 - max(softmax probability)
   - Higher uncertainty values indicate potential outliers
   - Simple yet effective method for detecting ambiguous paintings
   - Works with any classification model without additional training

### Painting Similarity System

The similarity detection system finds paintings with similar visual characteristics:

1. **Feature Extraction**: Extracts deep features from paintings
   - CNN-based extraction using ResNet18
   - Produces high-dimensional feature vectors
   - Captures both low-level and high-level visual features

2. **Similarity Computation**: Efficiently computes similarity between paintings
   - Cosine similarity for measuring feature vector similarity
   - Support for multiple distance metrics
   - Efficient computation for large-scale datasets

3. **Similarity Retrieval**: Finds and ranks similar paintings
   - Retrieves top-K most similar paintings
   - Ranks results by similarity score
   - Interactive visualization of similar artwork

## 📊 Results and Experiments

We have conducted initial experiments on a test dataset containing famous paintings from three major art styles: Renaissance, Baroque, and Impressionism. This section presents our findings.

### Classification Performance

Our CNN-RNN classifier with ResNet18 backbone achieved excellent classification performance on the test dataset:

| Metric                | Value |
|-----------------------|-------|
| Overall Style Accuracy| 85.0% |
| Training Epochs       | 5     |
| Batch Size            | 4     |
| Learning Rate         | 0.0001|

Style-specific performance metrics:

| Style         | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| Renaissance   | 0.70      | 1.00   | 0.82     |
| Baroque       | 1.00      | 0.80   | 0.89     |
| Impressionism | 1.00      | 0.75   | 0.86     |

### Confusion Matrix Analysis

Analysis of the confusion matrix reveals the model's classification patterns:

<div align="center">
<pre>
┌─────────────┬───────────────┬─────────┬──────────────┐
│             │ Renaissance   │ Baroque │ Impressionism│
├─────────────┼───────────────┼─────────┼──────────────┤
│ Renaissance │      7        │    0    │      0       │
├─────────────┼───────────────┼─────────┼──────────────┤
│ Baroque     │      1        │    4    │      0       │
├─────────────┼───────────────┼─────────┼──────────────┤
│Impressionism│      2        │    0    │      6       │
└─────────────┴───────────────┴─────────┴──────────────┘
</pre>
</div>

Key insights:
- Perfect classification of Renaissance paintings
- Strong performance on Baroque and Impressionism styles
- Minor confusion between Impressionism and Renaissance (2 paintings)
- One Baroque painting misclassified as Renaissance

### Outlier Detection Results

Our outlier detection system successfully identified paintings with ambiguous style characteristics using softmax uncertainty:

| Top Outliers            | Style         | Uncertainty Score |
|-------------------------|---------------|-------------------|
| Renaissance painting #1 | Renaissance   | 0.647             |
| Impressionism painting #1| Impressionism | 0.644            |
| Renaissance painting #2 | Renaissance   | 0.639             |
| Impressionism painting #2| Impressionism | 0.629            |
| Baroque painting #1     | Baroque       | 0.624             |

Paintings with higher uncertainty scores typically exhibit characteristics that span multiple artistic styles, making them more challenging to classify definitively.

### Training Progress

The training process showed rapid learning on our test dataset:

- Initial training accuracy: 35.71%
- Final training accuracy: 85.71%
- Validation accuracy increased from 0% to 33.33%
- Test accuracy: 85.0%

The fast convergence demonstrates the effectiveness of transfer learning with pretrained CNN backbones, even when working with a small dataset.

### Future Improvements

Based on our initial experiments, we've identified several areas for improvement:

1. **Larger Dataset**: Expanding beyond our test dataset to the full WikiArt collection
2. **Architecture Refinements**: Testing different CNN backbones and RNN configurations
3. **Hyperparameter Tuning**: Optimizing learning rate, batch size, and regularization
4. **Data Augmentation**: Implementing art-specific augmentation techniques
5. **Multi-attribute Learning**: Extending the model to classify artist and genre in addition to style

### Full WikiArt Dataset Implementation Plan

For our main task, we will implement the CNN-RNN classifier on the full WikiArt dataset, which is available in the Downloads folder. This will allow us to create a comprehensive art style, artist, and genre classification system.

#### Dataset Details

The WikiArt dataset from ArtGAN contains over 80,000 paintings with annotations for:
- 27 art styles (Impressionism, Cubism, Abstract, etc.)
- 23 genres (portrait, landscape, religious, etc.)
- 195 artists (Vincent van Gogh, Pablo Picasso, etc.)

#### Preprocessing Steps

1. **Data Organization**: The dataset will be organized by style/artist/genre hierarchy
2. **Image Standardization**: All images will be resized to 224×224 pixels
3. **Data Split**: 70% training, 15% validation, 15% test
4. **Metadata Generation**: Creating JSON metadata files with annotations

#### Training Approach

1. **Multi-attribute Training**: The model will be trained to simultaneously predict style, artist, and genre
2. **ResNet50 Backbone**: Using a deeper backbone for improved feature extraction
3. **Learning Rate Scheduling**: Implementing learning rate decay for better convergence
4. **Batch Size Optimization**: Starting with batch size 32 and adjusting based on GPU memory
5. **Dropout and Regularization**: Applying appropriate regularization to prevent overfitting

#### Evaluation Metrics

We will evaluate the model using the following metrics:

1. **Classification Accuracy**: Per-attribute accuracy (style, artist, genre)
2. **Precision, Recall, F1-Score**: For each class within each attribute
3. **Confusion Matrix Analysis**: Identifying common misclassifications
4. **Top-K Accuracy**: Measuring if correct label is within top K predictions
5. **ROC Curves and AUC**: For evaluating the binary classification performance of each class

#### Outlier Detection Methods

For the full dataset, we will implement several outlier detection methods:

1. **Softmax Uncertainty**: 1 - max(softmax probability)
2. **Entropy-based**: Using entropy of the softmax distribution
3. **Distance-based**: Using feature space distance from class centroids

#### Expected Timeframe

- Data Preprocessing: 1-2 days
- Initial Model Training: 2-3 days
- Hyperparameter Tuning: 2-3 days
- Evaluation and Analysis: 1-2 days
- Outlier Detection: 1-2 days

### Task 2: Similarity Detection Implementation

For our second task, we will implement a painting similarity detection system using the National Gallery of Art open dataset. This system will allow users to find paintings with similar visual characteristics across different styles, artists, and time periods.

#### Dataset Details

The National Gallery of Art (NGA) open dataset includes:
- Over 130,000 artwork records
- High-resolution images for a significant portion of the collection
- Detailed metadata including artist, title, date, medium, etc.

#### Feature Extraction Approaches

We will implement multiple feature extraction methods:

1. **CNN-based Features**:
   - ResNet50 features from the penultimate layer (2048-dimensional)
   - Fine-tuned on art datasets for domain adaptation
   - Global average pooling for dimensionality reduction

2. **CLIP-based Features**:
   - Using OpenAI's CLIP model for multi-modal features
   - Joint visual-textual embedding space
   - Zero-shot capability for novel art types

3. **Custom Art-specific Features**:
   - Color histogram and palette analysis
   - Composition and texture features
   - Edge and structure descriptors

#### Similarity Computation

For efficient similarity computation we will use:

1. **Faiss Indexing**:
   - Facebook AI Similarity Search (Faiss) for fast retrieval
   - Support for billion-scale similarity search
   - GPU acceleration for real-time queries

2. **Distance Metrics**:
   - Cosine similarity as the primary metric
   - L2 distance for certain feature types
   - Weighted combination for multi-feature approaches

#### User Interface Features

The similarity detection system will include:

1. **Query Interface**:
   - Upload custom images for querying
   - Select from gallery examples
   - Specify similarity criteria (visual, semantic, compositional)

2. **Results Visualization**:
   - Grid display of similar artworks
   - Similarity scores and explanations
   - Filtering by style, artist, period, etc.

#### Evaluation Metrics

We will evaluate the similarity system using:

1. **Precision@K**: Fraction of relevant items among top-K results
2. **Mean Average Precision (MAP)**: Measure of precision across recall levels
3. **Normalized Discounted Cumulative Gain (NDCG)**: Quality of ranking considering relevance
4. **User Studies**: Human evaluation of similarity perception
5. **Retrieval Time**: Computational efficiency metrics

#### Expected Timeframe

- Dataset Processing: 1-2 days
- Feature Extraction Implementation: 2-3 days
- Similarity Index Construction: 1-2 days
- User Interface Development: 2-3 days
- Evaluation and Benchmarking: 1-2 days

## 📄 Project Structure

```
ArtExtract/
├── data/                             # Data storage and preprocessing
│   ├── preprocessing/                # Scripts for data loading and preprocessing
│   ├── test_dataset/                 # Small dataset for initial testing
│   │   ├── style/                    # Images organized by style
│   │   └── metadata.json             # Metadata for test images
│   └── README.md                     # Data documentation
│
├── models/                             # Model implementations
│   ├── classification/                 # CNN-RNN models for classification
│   │   ├── cnn_rnn_classifier.py       # CNN-RNN architecture implementation
│   │   └── wikiart_dataset.py          # Dataset loading and preprocessing
│   ├── utils.py                        # Utility functions for model training and evaluation
│   └── similarity/                     # Similarity models
│       ├── feature_extraction.py       # Feature extraction from paintings
│       └── similarity_model.py         # Similarity model implementations
│
├── scripts/                               # Main training and evaluation scripts
│   ├── train_cnn_rnn_classifier.py        # Script for training CNN-RNN model
│   ├── evaluate_cnn_rnn_classifier.py     # Script for evaluating CNN-RNN model
│   └── README.md                          # Documentation for the scripts
│
├── evaluation_results/                    # Results from model evaluation
│   └── test/                              # Results from test dataset
│       ├── confusion_matrix_style.png     # Confusion matrix visualization
│       ├── evaluation_metrics.json        # Detailed evaluation metrics
│       └── outliers_style/                # Outlier visualizations
│
├── model_checkpoints/                      # Saved model checkpoints
│   └── classification_test/                # Checkpoints from test runs
│       ├── best_style_model.pth            # Best model checkpoint
│       └── training_curves.png             # Training progress visualization
│
├── demo/                                   # Demo applications and visualization tools
│
├── train.py                                 # Wrapper script for training
├── evaluate.py                              # Wrapper script for evaluation
│
├── requirements.txt                        # Project dependencies
├── setup.py                                # Package installation script
└── README.md                               # Project documentation
```

## 🔄 Data Flow

<div align="center">

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Input Data  │────▶│ Preprocessing│────▶│Model Training│────▶│  Validation  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐
│Visualization │◀────│   Results    │◀────│  Inference   │◀────│Trained Models│
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

</div>

## 🚀 Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ArtExtract.git
   cd ArtExtract
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model with the test dataset:
   ```bash
   ./train.py --data_dir data/test_dataset --batch_size 4 --num_epochs 5 --pretrained --test_mode --backbone resnet18 --save_dir model_checkpoints/classification_test 
   ```

4. Evaluate the trained model:
   ```bash
   ./evaluate.py --data_dir data/test_dataset --checkpoint model_checkpoints/classification_test/best_style_model.pth --test_mode --output_dir evaluation_results/test --backbone resnet18
   ```

5. For training with the full WikiArt dataset:
   ```bash
   ./train.py --data_dir path/to/wikiart --batch_size 32 --num_epochs 30 --pretrained --backbone resnet50 --save_dir model_checkpoints/classification
   ```

## 💻 Usage

### Style/Artist/Genre Classification

Use the CNN-RNN model for art classification:

```python
from models.classification.cnn_rnn_classifier import CNNRNNModel

# Initialize model
model = CNNRNNModel(
    num_classes=10,
    cnn_backbone='resnet50',
    rnn_type='gru',
    bidirectional=True,
    use_attention=True
)

# Load pre-trained weights
model.load_weights('path/to/weights.pth')

# Predict on an image
import cv2
import numpy as np

img = cv2.imread('path/to/artwork.jpg')
img = cv2.resize(img, (224, 224))
img = img / 255.0  # normalize
img = np.expand_dims(img, axis=0)  # add batch dimension

predictions = model.predict(img)
```

### Outlier Detection

Detect outliers in your dataset:

```python
from models.classification.outlier_detection import SoftmaxUncertaintyDetector

# Initialize detector
detector = SoftmaxUncertaintyDetector()

# Get outlier scores from model predictions
outlier_scores = detector.get_uncertainty_scores(predictions)

# Identify top outliers
top_outliers = detector.get_top_outliers(outlier_scores, k=5)

# Visualize outliers
from models.utils import visualize_outliers
visualize_outliers(images, predictions, outlier_scores, class_names)
```

### Painting Similarity Detection

Extract features from paintings:

```python
from models.similarity.feature_extraction import FeatureExtractor

# Initialize feature extractor
extractor = FeatureExtractor(model_type='resnet50')

# Extract features from an image
features = extractor.extract_features_from_image(image)

# Extract features from a directory of images
features_dict = extractor.extract_features_from_directory('path/to/images/')
```

Find similar paintings using the similarity model:

```python
from models.similarity.similarity_model import SimilaritySystem

# Create painting similarity system
similarity_system = SimilaritySystem(
    feature_extractor='resnet50',
    index_type='faiss',
    feature_dim=2048
)

# Index a collection of images
similarity_system.index_images('path/to/image/collection')

# Find similar paintings
similar_paintings = similarity_system.find_similar(query_image_path, k=5)
```

### Running the Visualization Demo

We've created a simple visualization demo that generates text-based representations of all the visualizations shown in this README. To run the demo:

```bash
# Run the visualization script
bash demo/run_visualizations.sh
```

This will generate text-based visualizations and save them to the appropriate locations. You can view them with commands like:

```bash
# View confusion matrix
cat evaluation_results/test/confusion_matrix_style.txt

# View outlier data
cat evaluation_results/test/outliers_style/outlier_data.txt
```

## 📊 Evaluation Metrics

### Classification Metrics

The classification models are evaluated using:

- **Accuracy**: Overall correctness of predictions
- **Precision & Recall**: Measure of exactness and completeness
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of classification performance
- **ROC Curve and AUC**: Performance across different thresholds

### Similarity Metrics

The similarity models are evaluated using:

- **Precision@k**: Precision of the top-k retrieved results
- **Mean Average Precision (MAP)**: Overall precision across all queries
- **Normalized Discounted Cumulative Gain (NDCG)**: Ranking quality measure
- **User Studies**: Human evaluation of similarity results

## 📈 Visualization

ArtExtract includes comprehensive visualization tools:

1. **Training Visualization**: Loss and accuracy curves during training
2. **Confusion Matrix**: Visual representation of classification performance
3. **Feature Space Visualization**: t-SNE and PCA projections of learned features
4. **Outlier Visualization**: Identification of paintings that don't fit their categories
5. **Similarity Visualization**: Interactive display of similar paintings

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
