# Data Documentation

## Datasets

### ArtGAN WikiArt Dataset
- **Source**: https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md
- **Used for**: Style/Artist/Genre Classification (Task 1)
- **Structure**:
  - 80,000+ artworks spanning multiple centuries
  - 27 styles (e.g., Baroque, Impressionism, Cubism, Romanticism, Renaissance)
  - 139 artists (e.g., Pablo Picasso, Vincent van Gogh, Claude Monet)
  - 45 genres (e.g., portrait, landscape, religious painting, still life)
- **Format**: JPG images with varying resolutions
- **License**: For research and educational use only

### National Gallery of Art Dataset
- **Source**: https://github.com/NationalGalleryOfArt/opendata
- **Used for**: Painting Similarity Detection (Task 2)
- **Structure**:
  - Collection of over 130,000 artworks with detailed metadata
  - Focus on portraits, poses, and compositions
  - Includes artist information, creation dates, medium, dimensions
  - Contains high-resolution images suitable for feature extraction
- **Format**: JPG/TIFF images with JSON metadata
- **License**: Most images are in the public domain

## Data Preprocessing

### Loading and Initial Processing
- **WikiArtDataset** and **NationalGalleryDataset** classes handle dataset loading
- Images are verified for integrity and corrupted files are skipped
- Metadata is parsed and indexed for efficient retrieval

### Image Preprocessing Pipeline
1. **Image loading and resizing**:
   - Images resized to 224×224 pixels (standard for CNN architectures)
   - Aspect ratio preservation with center cropping

2. **Data augmentation** (for training only):
   - Random resized crops (scale: 0.8-1.0)
   - Random horizontal flips
   - Color jitter (brightness, contrast, saturation: ±0.2)
   - Random affine transformations (rotation: ±15°, translation: ±10%, scale: 0.9-1.1)

3. **Normalization**:
   - Pixel values scaled to [0,1]
   - Normalized with ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]

4. **Dataset splitting**:
   - Train: 70%, Validation: 15%, Test: 15%
   - Stratified sampling to maintain class distribution
   - Fixed random seed (42) for reproducibility

### Feature Extraction (for Similarity Detection)
- **Feature extraction** using pre-trained models:
  - CNN backbones: ResNet50, EfficientNet
  - CLIP models for multi-modal features
- Features are normalized and stored in pickle format

## Directory Structure

After downloading, organize the data as follows:

```
data/
├── raw/
│   ├── wikiart/              # ArtGAN WikiArt dataset
│   │   ├── style/            # Images organized by style
│   │   │   ├── Baroque/
│   │   │   ├── Impressionism/
│   │   │   └── ...
│   │   ├── artist/           # Images organized by artist
│   │   │   ├── Picasso/
│   │   │   ├── Monet/
│   │   │   └── ...
│   │   └── genre/            # Images organized by genre
│   │       ├── portrait/
│   │       ├── landscape/
│   │       └── ...
│   └── national_gallery/     # National Gallery dataset
│       ├── images/
│       └── metadata.json
├── processed/                # Processed data ready for model training
│   ├── wikiart/
│   │   ├── train_style.csv
│   │   ├── val_style.csv
│   │   ├── test_style.csv
│   │   └── ...
│   └── national_gallery/
│       ├── features/
│       │   ├── resnet50_features.pkl
│       │   ├── clip_features.pkl
│       │   └── ...
│       └── metadata_processed.csv
└── metadata/                 # Metadata files
    ├── style_mapping.json
    ├── artist_mapping.json
    └── genre_mapping.json
```

## Usage

### Loading the WikiArt Dataset

```python
from data.preprocessing.art_dataset_loader import WikiArtDataset

# Initialize dataset loader
wikiart = WikiArtDataset('/path/to/data/raw/wikiart')

# Get dataframe for style classification
style_df = wikiart.create_dataframe(category='style')

# Load a specific image
image = wikiart.load_image(style_df.iloc[0]['image_path'], target_size=(224, 224))
```

### Creating Data Loaders

```python
from data.preprocessing.data_preprocessing import ArtDatasetPreprocessor, create_data_loaders

# Create preprocessor
preprocessor = ArtDatasetPreprocessor(img_size=(224, 224), use_augmentation=True)

# Create data loaders
dataloaders = create_data_loaders(
    dataframe=style_df,
    preprocessor=preprocessor,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Access train, validation, and test loaders
train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']
```

### Extracting Features for Similarity Detection

```python
from models.similarity_detection.feature_extraction import create_feature_extractor

# Create feature extractor
extractor = create_feature_extractor('cnn', model_name='resnet50')

# Extract features from a dataloader
features, image_paths = extractor.extract_features_from_dataloader(dataloader)

# Save features
extractor.save_features(features, image_paths, 'data/processed/national_gallery/features/resnet50_features.pkl')
```

See the `preprocessing` directory for more detailed scripts and examples.
