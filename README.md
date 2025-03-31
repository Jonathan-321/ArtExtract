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
- **Dataset**: ArtGAN WikiArt collection (80,000+ paintings)
- **Features**:
  - Multiple CNN backbones (ResNet, EfficientNet)
  - Bidirectional RNN layers (LSTM/GRU)
  - Attention for focusing on artistic elements
  - Comprehensive outlier detection
  - Robust evaluation metrics

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
│                        ArtExtract System                         │
├─────────────────────────────┬───────────────────────────────────┤
│  Style/Artist Classification │     Painting Similarity Detection  │
├─────────────────────────────┼───────────────────────────────────┤
│ ┌─────────────────────────┐ │ ┌─────────────────────────────┐   │
│ │      CNN Backbone       │ │ │    Feature Extraction       │   │
│ │  ResNet/EfficientNet    │ │ │    CNN/CLIP Models          │   │
│ └───────────┬─────────────┘ │ └───────────┬─────────────────┘   │
│             ▼               │             ▼                      │
│ ┌─────────────────────────┐ │ ┌─────────────────────────────┐   │
│ │   Feature Processing    │ │ │    Similarity Computation    │   │
│ │  Attention Mechanism    │ │ │    Cosine/Faiss Index        │   │
│ └───────────┬─────────────┘ │ └───────────┬─────────────────┘   │
│             ▼               │             ▼                      │
│ ┌─────────────────────────┐ │ ┌─────────────────────────────┐   │
│ │      RNN Layers         │ │ │      Ranking Engine          │   │
│ │   LSTM/GRU/Bidirectional│ │ │      Top-K Results           │   │
│ └───────────┬─────────────┘ │ └───────────┬─────────────────┘   │
│             ▼               │             ▼                      │
│ ┌─────────────────────────┐ │ ┌─────────────────────────────┐   │
│ │   Classification Head   │ │ │    Interactive Results       │   │
│ │   Style/Artist/Genre    │ │ │    Visualization UI          │   │
│ └─────────────────────────┘ │ └─────────────────────────────┘   │
└─────────────────────────────┴───────────────────────────────────┘
```

### CNN-RNN Classification Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Style/Artist/Genre Classification System              │
├───────────┬─────────────┬────────────────┬────────────────┬─────────────┤
│           │             │                │                │             │
│  Input    │    CNN      │   Feature      │     RNN        │  Output     │
│  Image    │  Backbone   │   Processing   │    Layer       │  Classes    │
│           │             │                │                │             │
└─────┬─────┴──────┬──────┴────────┬───────┴────────┬───────┴──────┬──────┘
      │            │               │                │              │
      ▼            ▼               ▼                ▼              ▼
┌──────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐
│ 224x224  │ │• ResNet50    │ │• Spatial     │ │• LSTM/GRU    │ │• Style  │
│ RGB      │ │• ResNet18    │ │  Features    │ │• Bidirectional│ │• Artist │
│ Artwork  │ │• EfficientNet│ │• Attention   │ │• Attention   │ │• Genre  │
│ Image    │ │• MobileNetV2 │ │  Mechanism   │ │  Weights     │ │• Softmax│
└──────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └─────────┘
```

### Painting Similarity Detection System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Painting Similarity Detection System                  │
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
│ Artwork  │ │• VGG16       │ │  Similarity  │ │  Results     │ │  Artwork │
│ Image    │ │• CLIP        │ │• Faiss Index │ │• Confidence  │ │• Ranked  │
│          │ │• Custom      │ │• L2 Distance │ │  Scores      │ │  Results │
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
   - Supports multiple architectures: ResNet50/18, EfficientNet-B0, MobileNetV2
   - Pretrained on ImageNet and fine-tuned on art datasets
   - Outputs feature maps that capture artistic elements

2. **Feature Processing**: Transforms CNN features for RNN consumption
   - Reshapes spatial features to sequential format
   - Optional attention mechanism to focus on discriminative regions
   - Maintains spatial relationships in feature representation

3. **RNN Layer**: Processes sequential information in the feature maps
   - LSTM or GRU cells with optional bidirectional processing
   - Captures temporal and spatial relationships between features
   - Attention mechanism for highlighting important features

4. **Classification Layer**: Produces final style/artist/genre predictions
   - Fully-connected layer with softmax activation
   - Multi-class classification with confidence scores
   - Optional outlier detection for identifying unusual artwork

### Outlier Detection

The system implements multiple outlier detection methods to identify paintings that don't fit their assigned categories:

1. **Isolation Forest**: Unsupervised algorithm that isolates observations by randomly selecting features
2. **Local Outlier Factor**: Measures the local deviation of density of a sample with respect to its neighbors
3. **Autoencoder-based Detection**: Neural network trained to reconstruct input data, where outliers have higher reconstruction error

### Painting Similarity System

The similarity detection system finds paintings with similar visual characteristics:

1. **Feature Extraction**: Extracts deep features from paintings
   - CNN-based extraction using ResNet, VGG16, or EfficientNet
   - CLIP-based extraction for semantic understanding
   - Produces high-dimensional feature vectors (2048-dim)

2. **Similarity Index**: Efficiently computes similarity between paintings
   - Faiss index for fast approximate nearest neighbor search
   - Supports different distance metrics (L2, inner product, cosine)
   - GPU acceleration for large-scale similarity computation

3. **Similarity Retrieval**: Finds and ranks similar paintings
   - Retrieves top-K most similar paintings
   - Ranks results by similarity score
   - Interactive visualization of similar artwork

## 📁 Project Structure

```
ArtExtract/
├── data/                             # Data storage and preprocessing
│   ├── preprocessing/                # Scripts for data loading and preprocessing
│   └── README.md                     # Data documentation
│
├── models/                           # Model implementations
│   ├── style_classification/         # CNN-RNN models for classification
│   │   ├── cnn_rnn_model.py          # CNN-RNN architecture implementation
│   │   ├── outlier_detection.py      # Outlier detection methods
│   │   ├── train_wikiart_refined.py  # Training script for WikiArt dataset
│   │   └── test_train.py             # Testing and evaluation script
│   │
│   ├── similarity_detection/         # Similarity models
│   │   ├── feature_extraction.py     # Feature extraction from paintings
│   │   ├── similarity_model.py       # Similarity model implementations
│   │   ├── train_similarity_model.py # Training script for similarity models
│   │   └── demo_similarity.py        # Demo script for similarity detection
│   │
│   └── utils.py                      # Shared utilities
│
├── demo/                             # Interactive demo applications
│   └── demo_app.py                   # Gradio-based demo interface
│
├── evaluation/                       # Evaluation metrics and scripts
│   ├── classification_metrics.py     # Metrics for classification task
│   ├── similarity_metrics.py         # Metrics for similarity detection task
│   └── visualization.py              # Visualization utilities
│
├── notebooks/                        # Jupyter notebooks for exploration
│   └── similarity_detection_demo.ipynb  # Demo notebook for similarity detection
│
├── ArtExtract_Project_Report.md      # Detailed project report
├── requirements.txt                  # Project dependencies
└── README.md                         # Project documentation
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

3. Download the datasets:
   - ArtGAN WikiArt dataset: https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md
   - National Gallery of Art dataset: https://github.com/NationalGalleryOfArt/opendata

4. Prepare data:
   ```bash
   python data/preprocessing/extract_wikiart.py --dataset wikiart --output_dir data/wikiart_refined
   ```

## 💻 Usage

### Style/Artist/Genre Classification

Use the CNN-RNN model for art classification:

```python
from models.style_classification.cnn_rnn_model import CNNRNNModel

# Initialize model
model = CNNRNNModel(
    num_classes=10,
    cnn_backbone='resnet50',
    rnn_type='lstm',
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
from models.style_classification.outlier_detection import IsolationForestDetector

# Initialize detector
detector = IsolationForestDetector()

# Fit detector on features
detector.fit(features)

# Get outlier indices
outlier_indices = detector.get_outlier_indices(features)

# Visualize outliers
from models.style_classification.outlier_detection import visualize_outliers
visualize_outliers(features, labels, outlier_indices, class_names)
```

### Painting Similarity Detection

Extract features from paintings:

```python
from models.similarity_detection.feature_extraction import FeatureExtractor

# Initialize feature extractor (options: 'resnet50', 'vgg16', 'clip')
extractor = FeatureExtractor(model_type='resnet50')

# Extract features from an image
features = extractor.extract_features_from_image(image)

# Extract features from a directory of images
features_dict = extractor.extract_features_from_directory('path/to/images/')
```

Find similar paintings using the similarity model:

```python
from models.similarity_detection.similarity_model import (
    create_similarity_model,
    PaintingSimilaritySystem
)

# Create similarity model (options: 'cosine', 'faiss')
similarity_model = create_similarity_model('faiss', feature_dim=2048)

# Create painting similarity system
similarity_system = PaintingSimilaritySystem(
    similarity_model=similarity_model,
    features=features,
    image_paths=image_paths
)

# Find similar paintings
result = similarity_system.find_similar_paintings(query_idx=0, k=5)
```

### Running the Demo

Run the interactive demo application:

```bash
python demo/demo_app.py --model_path path/to/model --interactive
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

## 🔮 Future Work

We are actively working on enhancing ArtExtract with:
- Transformer-based architectures (Vision Transformer)
- Multi-modal models combining image and textual descriptions
- Self-supervised learning approaches for improved feature extraction
- Style transfer capabilities
- Interactive web application for art exploration

## 📚 Citation

If you use ArtExtract in your research, please cite:

```
@software{ArtExtract2023,
  author = {Your Name},
  title = {ArtExtract: Deep Learning for Art Classification and Similarity Detection},
  year = {2023},
  url = {https://github.com/yourusername/ArtExtract}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
