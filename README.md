# ArtExtract

A deep learning project for art classification and similarity detection.

## Project Overview

This project consists of two main tasks:

### Task 1: Style/Artist/Genre Classification
- Convolutional-recurrent model for art classification using the ArtGAN WikiArt dataset
- Outlier detection in classifications
- Comprehensive evaluation metrics

### Task 2: Painting Similarity Detection
- Similarity model for paintings from the National Gallery of Art dataset
- Focus on finding similar portraits, poses, or compositions
- Feature extraction using CNN and CLIP models
- Multiple similarity metrics including cosine similarity and Faiss indexing
- Documented approach and evaluation metrics

## Project Structure

```
ArtExtract/
├── data/                      # Data storage and preprocessing
│   ├── preprocessing/         # Scripts for data loading and preprocessing
│   └── README.md              # Data documentation
├── models/                    # Model implementations
│   ├── style_classification/  # CNN-RNN models for classification
│   │   ├── cnn_rnn_model.py   # CNN-RNN architecture implementation
│   │   └── outlier_detection.py # Outlier detection methods
│   ├── similarity_detection/  # Similarity models
│   │   ├── feature_extraction.py # Feature extraction from paintings
│   │   ├── similarity_model.py # Similarity model implementations
│   │   ├── train_similarity_model.py # Training script for similarity models
│   │   └── demo_similarity.py # Demo script for similarity detection
│   └── utils.py               # Shared utilities
├── notebooks/                 # Jupyter notebooks for exploration and visualization
│   └── similarity_detection_demo.ipynb # Demo notebook for similarity detection
├── evaluation/                # Evaluation metrics and scripts
│   ├── classification_metrics.py # Metrics for classification task
│   └── similarity_metrics.py  # Metrics for similarity detection task
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the datasets:
   - ArtGAN WikiArt dataset: https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md
   - National Gallery of Art dataset: https://github.com/NationalGalleryOfArt/opendata

## Usage

### Style/Artist/Genre Classification

Use the CNN-RNN model for art classification:

```python
from models.style_classification.cnn_rnn_model import CNNRNNModel

# Initialize model
model = CNNRNNModel(num_classes=10, cnn_backbone='resnet50')

# See notebooks for training and evaluation examples
```

### Painting Similarity Detection

#### Feature Extraction

Extract features from paintings:

```python
from models.similarity_detection.feature_extraction import FeatureExtractor

# Initialize feature extractor
extractor = FeatureExtractor(model_type='resnet50')

# Extract features from an image
features = extractor.extract_features_from_image(image)
```

#### Finding Similar Paintings

Find similar paintings using the similarity model:

```python
from models.similarity_detection.similarity_model import (
    create_similarity_model,
    PaintingSimilaritySystem
)

# Create similarity model
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

#### Running the Demo

Run the similarity detection demo:

```bash
python models/similarity_detection/demo_similarity.py --model_path path/to/model --interactive
```

See the notebooks directory for more detailed examples of how to use the models and process the data.
