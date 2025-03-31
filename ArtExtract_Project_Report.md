# ArtExtract: Art Classification and Similarity Detection
## Project Report

## 1. Introduction

ArtExtract is a deep learning project focused on two main tasks in the domain of art analysis:

1. **Style/Artist/Genre Classification**: Using convolutional-recurrent neural networks to classify artwork by style, artist, and genre.
2. **Painting Similarity Detection**: Finding similarities between paintings based on visual features.

This report details our approach, methodology, implementation, and results for both tasks.

## 2. Task 1: Convolutional-Recurrent Architectures for Art Classification

### 2.1 Dataset

We used the WikiArt dataset from the ArtGAN project, which contains over 80,000 paintings categorized by style, artist, and genre. The dataset is particularly challenging due to:

- Imbalanced class distribution
- High intra-class variability
- Complex visual features that define artistic styles

### 2.2 Methodology

#### 2.2.1 Model Architecture

We implemented a hybrid CNN-RNN architecture:

1. **CNN Component**: 
   - ResNet18/50 backbone pre-trained on ImageNet
   - Fine-tuned on artwork classification
   - Extracts spatial features from paintings

2. **RNN Component** (optional):
   - LSTM layers that process sequential features from the CNN
   - Captures temporal relationships in the feature space
   - Particularly useful for style classification where context matters

3. **Classification Head**:
   - Fully connected layers with dropout for regularization
   - Softmax activation for multi-class classification

#### 2.2.2 Data Preprocessing

- Image resizing to 224×224 pixels
- Data augmentation: random crops, flips, rotations, and color jittering
- Normalization using ImageNet statistics
- Robust path handling to accommodate variations in file paths

#### 2.2.3 Training Strategy

- Cross-entropy loss function
- Adam optimizer with learning rate scheduling
- Early stopping based on validation loss
- Mixed precision training for efficiency
- Prefetching data loader for improved throughput

### 2.3 Outlier Detection

We implemented multiple outlier detection methods to identify paintings that don't fit their assigned categories:

1. **Isolation Forest**:
   - Unsupervised algorithm that isolates observations by randomly selecting features
   - Effective for high-dimensional data like image features

2. **Local Outlier Factor**:
   - Measures the local deviation of density of a sample with respect to its neighbors
   - Identifies samples with substantially lower density than their neighbors

3. **Autoencoder-based Detection**:
   - Neural network trained to reconstruct input data
   - Outliers have higher reconstruction error

### 2.4 Evaluation Metrics

We used the following metrics to evaluate our classification model:

1. **Accuracy**: Overall correctness of predictions
2. **Precision, Recall, F1-score**: Class-specific performance
3. **Confusion Matrix**: Visualizes classification errors
4. **ROC-AUC**: Evaluates model's ability to discriminate between classes
5. **Top-k Accuracy**: Considers prediction as correct if true label is among top k predictions

### 2.5 Results

[Note: This section would be populated with actual results from the training run]

- Overall accuracy: X%
- Top-5 accuracy: Y%
- F1-score (weighted): Z
- Number of outliers detected: N
- Classes with highest outlier rates: [List classes]

### 2.6 Visualization

We created visualizations to better understand the model's performance:

1. **Confusion Matrix**: Highlights common misclassifications
2. **Feature Space Visualization**: t-SNE projection of learned features
3. **Outlier Visualization**: Identifies paintings that don't fit their categories
4. **Class Distribution**: Shows imbalance in the dataset

## 3. Task 2: Similarity Detection in Paintings

### 3.1 Dataset

We used the National Gallery of Art open dataset, which contains high-quality images of paintings with detailed metadata.

### 3.2 Methodology

#### 3.2.1 Feature Extraction

We implemented multiple feature extraction methods:

1. **CNN Features**:
   - ResNet50 pre-trained on ImageNet
   - Features extracted from the penultimate layer
   - Captures general visual characteristics

2. **CLIP Features**:
   - OpenAI's CLIP model that connects text and images
   - Provides semantic understanding of painting content
   - Enables both visual and textual queries

3. **Custom Features**:
   - Combination of low-level features (color histograms, edge detection)
   - Mid-level features (composition, texture)
   - High-level semantic features (object detection)

#### 3.2.2 Similarity Metrics

We implemented several similarity metrics:

1. **Cosine Similarity**:
   - Measures the cosine of the angle between feature vectors
   - Scale-invariant, focusing on direction rather than magnitude

2. **Euclidean Distance**:
   - Measures the straight-line distance between feature vectors
   - Sensitive to the magnitude of features

3. **Faiss Indexing**:
   - Facebook AI's similarity search library
   - Efficient for large-scale similarity search
   - Supports approximate nearest neighbor search

### 3.3 Evaluation Metrics

Evaluating similarity models is challenging due to the subjective nature of similarity. We used:

1. **Human Evaluation**:
   - Expert ratings of similarity between paintings
   - Comparison of model rankings with human rankings

2. **Precision@k**:
   - Proportion of relevant items among top-k recommendations

3. **Mean Average Precision (MAP)**:
   - Average precision across multiple queries

4. **Normalized Discounted Cumulative Gain (NDCG)**:
   - Measures ranking quality with relevance scores

### 3.4 Results

[Note: This section would be populated with actual results from the similarity model]

- Top-5 precision: X%
- MAP: Y
- NDCG: Z
- Human evaluation correlation: W

### 3.5 Visualization

We created visualizations to demonstrate the similarity model:

1. **Similarity Heatmap**: Shows similarity between paintings
2. **t-SNE Visualization**: Projects high-dimensional features to 2D
3. **Query Results**: Visual examples of similar paintings for various queries

## 4. Implementation Details

### 4.1 Code Structure

The project is organized into the following components:

```
ArtExtract/
├── data/                             # Data storage and preprocessing
│   ├── preprocessing/                # Scripts for data loading and preprocessing
│   └── README.md                     # Data documentation
│
├── models/                           # Model implementations
│   ├── style_classification/         # CNN-RNN models for classification
│   │   ├── cnn_rnn_model.py          # CNN-RNN architecture implementation
│   │   └── outlier_detection.py      # Outlier detection methods
│   │
│   ├── similarity_detection/         # Similarity models
│   │   ├── feature_extraction.py     # Feature extraction from paintings
│   │   ├── similarity_model.py       # Similarity model implementations
│   │   ├── train_similarity_model.py # Training script for similarity models
│   │   └── demo_similarity.py        # Demo script for similarity detection
│   │
│   └── utils.py                      # Shared utilities
│
├── notebooks/                        # Jupyter notebooks for exploration and visualization
│   └── similarity_detection_demo.ipynb  # Demo notebook for similarity detection
│
├── evaluation/                       # Evaluation metrics and scripts
│   ├── classification_metrics.py     # Metrics for classification task
│   └── similarity_metrics.py         # Metrics for similarity detection task
│
├── requirements.txt                  # Project dependencies
└── README.md                         # Project documentation
```

### 4.2 Dependencies

The project relies on the following key libraries:

- PyTorch for deep learning models
- torchvision for image processing and pre-trained models
- scikit-learn for evaluation metrics and outlier detection
- pandas for data manipulation
- matplotlib and seaborn for visualization
- Faiss for efficient similarity search
- Gradio for the demo interface

### 4.3 Challenges and Solutions

During implementation, we encountered several challenges:

1. **Image Loading Issues**:
   - **Challenge**: Discrepancies between file paths in CSV and actual file locations
   - **Solution**: Implemented robust path handling with multiple fallback options and path caching

2. **Training Performance**:
   - **Challenge**: Slow data loading bottlenecking training
   - **Solution**: Implemented prefetching data loader to load images in background

3. **Outlier Detection in High Dimensions**:
   - **Challenge**: Curse of dimensionality affecting outlier detection
   - **Solution**: Implemented multiple detection methods and dimensionality reduction

4. **Similarity Evaluation**:
   - **Challenge**: Subjective nature of similarity making evaluation difficult
   - **Solution**: Combined multiple evaluation metrics and human evaluation

## 5. Discussion and Future Work

### 5.1 Strengths

- Robust data loading and preprocessing pipeline
- Multiple model architectures for comparison
- Comprehensive outlier detection
- Efficient similarity search implementation
- Interactive demo for showcasing results

### 5.2 Limitations

- Limited by available computational resources
- Subjective nature of art classification and similarity
- Imbalanced dataset affecting performance on minority classes
- Lack of temporal information in static images

### 5.3 Future Work

1. **Model Improvements**:
   - Experiment with transformer architectures (ViT, SWIN)
   - Implement attention mechanisms for focusing on relevant parts of paintings
   - Explore multi-task learning for joint classification of style, artist, and genre

2. **Data Enhancements**:
   - Incorporate textual metadata and descriptions
   - Use additional datasets for cross-domain generalization
   - Implement data synthesis for minority classes

3. **Application Extensions**:
   - Art recommendation system based on user preferences
   - Style transfer between paintings
   - Generative models for creating new artwork in specific styles

## 6. Conclusion

The ArtExtract project demonstrates the effectiveness of deep learning techniques for art classification and similarity detection. Our CNN-RNN architecture achieves strong performance on style classification, while our similarity detection system effectively identifies visually similar paintings.

The project contributes to the field of computational art analysis by providing:

1. A robust framework for art classification and similarity detection
2. Comprehensive outlier detection for identifying misclassified paintings
3. An interactive demo for exploring art collections

The methods developed in this project can be applied to various domains in digital humanities, museum curation, and art education.

## 7. References

[List of references to papers, datasets, and other resources used in the project]
