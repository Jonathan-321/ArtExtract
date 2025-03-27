"""
Similarity model for painting similarity detection.
This module implements methods for finding similar paintings based on extracted features.
"""

import numpy as np
import torch
import faiss
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
from tqdm import tqdm
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimilarityModel:
    """
    Base class for similarity models.
    """
    
    def __init__(self):
        """Initialize the similarity model."""
        logger.info("Initialized similarity model")
    
    def find_similar(self, 
                    query_feature: np.ndarray, 
                    feature_database: np.ndarray, 
                    k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar items to the query.
        
        Args:
            query_feature: Query feature vector
            feature_database: Database of feature vectors
            k: Number of similar items to retrieve
            
        Returns:
            Tuple of (indices, distances)
        """
        raise NotImplementedError("Subclasses must implement find_similar")
    
    def save_model(self, output_path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            output_path: Path to save the model
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Saved similarity model to {output_path}")
    
    @staticmethod
    def load_model(input_path: str) -> 'SimilarityModel':
        """
        Load a model from disk.
        
        Args:
            input_path: Path to load the model from
            
        Returns:
            Loaded similarity model
        """
        with open(input_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loaded similarity model from {input_path}")
        
        return model


class CosineSimilarityModel(SimilarityModel):
    """
    Similarity model using cosine similarity.
    """
    
    def __init__(self, normalize_features: bool = True):
        """
        Initialize the cosine similarity model.
        
        Args:
            normalize_features: Whether to normalize feature vectors
        """
        super().__init__()
        self.normalize_features = normalize_features
        logger.info(f"Initialized cosine similarity model (normalize_features={normalize_features})")
    
    def find_similar(self, 
                    query_feature: np.ndarray, 
                    feature_database: np.ndarray, 
                    k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar items to the query using cosine similarity.
        
        Args:
            query_feature: Query feature vector
            feature_database: Database of feature vectors
            k: Number of similar items to retrieve
            
        Returns:
            Tuple of (indices, similarities)
        """
        # Ensure query_feature is 2D
        if query_feature.ndim == 1:
            query_feature = query_feature.reshape(1, -1)
        
        # Normalize features if requested
        if self.normalize_features:
            query_feature = normalize(query_feature)
            feature_database = normalize(feature_database)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_feature, feature_database)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        top_similarities = similarities[top_indices]
        
        return top_indices, top_similarities


class FaissIndexModel(SimilarityModel):
    """
    Similarity model using Faiss index for efficient similarity search.
    """
    
    def __init__(self, 
                feature_dim: int, 
                index_type: str = 'L2',
                use_gpu: bool = torch.cuda.is_available()):
        """
        Initialize the Faiss index model.
        
        Args:
            feature_dim: Dimension of feature vectors
            index_type: Type of index ('L2' for L2 distance, 'IP' for inner product, 'Cosine' for cosine similarity)
            use_gpu: Whether to use GPU for index
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.index_type = index_type
        self.use_gpu = use_gpu
        
        # Create index
        self._create_index()
        
        logger.info(f"Initialized Faiss index model with {index_type} index (feature_dim={feature_dim}, use_gpu={use_gpu})")
    
    def _create_index(self) -> None:
        """Create Faiss index."""
        if self.index_type == 'L2':
            self.index = faiss.IndexFlatL2(self.feature_dim)
        elif self.index_type == 'IP':
            self.index = faiss.IndexFlatIP(self.feature_dim)
        elif self.index_type == 'Cosine':
            self.index = faiss.IndexFlatIP(self.feature_dim)
            self.normalize_features = True
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Use GPU if requested and available
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Using GPU for Faiss index")
            except Exception as e:
                logger.warning(f"Failed to use GPU for Faiss index: {e}")
                self.use_gpu = False
    
    def add_to_index(self, features: np.ndarray) -> None:
        """
        Add features to the index.
        
        Args:
            features: Feature vectors to add
        """
        # Ensure features are float32
        features = features.astype(np.float32)
        
        # Normalize features for cosine similarity
        if self.index_type == 'Cosine':
            features = normalize(features).astype(np.float32)
        
        # Add to index
        self.index.add(features)
        logger.info(f"Added {features.shape[0]} vectors to Faiss index")
    
    def find_similar(self, 
                    query_feature: np.ndarray, 
                    feature_database: Optional[np.ndarray] = None, 
                    k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar items to the query using Faiss index.
        
        Args:
            query_feature: Query feature vector
            feature_database: Not used for Faiss index (features should be added with add_to_index)
            k: Number of similar items to retrieve
            
        Returns:
            Tuple of (indices, distances)
        """
        # Ensure query_feature is 2D and float32
        if query_feature.ndim == 1:
            query_feature = query_feature.reshape(1, -1)
        
        query_feature = query_feature.astype(np.float32)
        
        # Normalize query for cosine similarity
        if self.index_type == 'Cosine':
            query_feature = normalize(query_feature).astype(np.float32)
        
        # Search index
        distances, indices = self.index.search(query_feature, k)
        
        # Convert distances to similarities for cosine similarity
        if self.index_type == 'Cosine' or self.index_type == 'IP':
            # For inner product and cosine, higher is better
            similarities = distances[0]
            return indices[0], similarities
        else:
            # For L2 distance, lower is better, so we negate
            distances = -distances[0]
            return indices[0], distances


class PaintingSimilaritySystem:
    """
    System for finding similar paintings.
    """
    
    def __init__(self, 
                similarity_model: SimilarityModel,
                features: np.ndarray,
                image_paths: List[str],
                metadata: Optional[pd.DataFrame] = None):
        """
        Initialize the painting similarity system.
        
        Args:
            similarity_model: Similarity model
            features: Feature vectors for paintings
            image_paths: Paths to painting images
            metadata: Optional DataFrame with painting metadata
        """
        self.similarity_model = similarity_model
        self.features = features
        self.image_paths = image_paths
        self.metadata = metadata
        
        # Add features to index if using Faiss
        if isinstance(similarity_model, FaissIndexModel):
            similarity_model.add_to_index(features)
        
        logger.info(f"Initialized painting similarity system with {len(image_paths)} paintings")
    
    def find_similar_paintings(self, 
                             query_idx: int, 
                             k: int = 5) -> Dict[str, Any]:
        """
        Find paintings similar to the query painting.
        
        Args:
            query_idx: Index of the query painting
            k: Number of similar paintings to retrieve
            
        Returns:
            Dictionary with similar paintings information
        """
        # Get query feature and path
        query_feature = self.features[query_idx]
        query_path = self.image_paths[query_idx]
        
        # Find similar paintings
        indices, similarities = self.similarity_model.find_similar(
            query_feature, self.features, k + 1  # +1 because the query itself will be included
        )
        
        # Remove the query itself (should be the first result)
        if indices[0] == query_idx:
            indices = indices[1:]
            similarities = similarities[1:]
        else:
            indices = indices[:k]
            similarities = similarities[:k]
        
        # Get paths and metadata for similar paintings
        similar_paths = [self.image_paths[idx] for idx in indices]
        
        result = {
            'query_idx': query_idx,
            'query_path': query_path,
            'similar_indices': indices.tolist(),
            'similar_paths': similar_paths,
            'similarities': similarities.tolist()
        }
        
        # Add metadata if available
        if self.metadata is not None:
            query_metadata = self._get_metadata_for_path(query_path)
            similar_metadata = [self._get_metadata_for_path(path) for path in similar_paths]
            
            result['query_metadata'] = query_metadata
            result['similar_metadata'] = similar_metadata
        
        return result
    
    def find_similar_to_new_painting(self, 
                                   query_feature: np.ndarray, 
                                   query_path: str,
                                   k: int = 5) -> Dict[str, Any]:
        """
        Find paintings similar to a new painting not in the database.
        
        Args:
            query_feature: Feature vector of the query painting
            query_path: Path to the query painting
            k: Number of similar paintings to retrieve
            
        Returns:
            Dictionary with similar paintings information
        """
        # Find similar paintings
        indices, similarities = self.similarity_model.find_similar(
            query_feature, self.features, k
        )
        
        # Get paths and metadata for similar paintings
        similar_paths = [self.image_paths[idx] for idx in indices]
        
        result = {
            'query_path': query_path,
            'similar_indices': indices.tolist(),
            'similar_paths': similar_paths,
            'similarities': similarities.tolist()
        }
        
        # Add metadata if available
        if self.metadata is not None:
            similar_metadata = [self._get_metadata_for_path(path) for path in similar_paths]
            result['similar_metadata'] = similar_metadata
        
        return result
    
    def _get_metadata_for_path(self, path: str) -> Dict[str, Any]:
        """
        Get metadata for a painting path.
        
        Args:
            path: Path to the painting
            
        Returns:
            Dictionary with metadata
        """
        if self.metadata is None:
            return {}
        
        # Find the row in metadata that matches the path
        # This assumes there's a column that can be matched with the path
        # Adjust as needed based on your metadata structure
        filename = os.path.basename(path)
        
        # Try different approaches to match the path with metadata
        if 'image_path' in self.metadata.columns:
            row = self.metadata[self.metadata['image_path'] == path]
        elif 'filename' in self.metadata.columns:
            row = self.metadata[self.metadata['filename'] == filename]
        elif 'object_id' in self.metadata.columns:
            # Try to extract object ID from filename
            object_id = os.path.splitext(filename)[0]
            row = self.metadata[self.metadata['object_id'] == object_id]
        else:
            return {}
        
        if len(row) == 0:
            return {}
        
        # Convert row to dictionary
        return row.iloc[0].to_dict()
    
    def visualize_similar_paintings(self, 
                                  query_idx: int, 
                                  k: int = 5,
                                  figsize: Tuple[int, int] = (15, 10),
                                  save_path: Optional[str] = None) -> None:
        """
        Visualize paintings similar to the query painting.
        
        Args:
            query_idx: Index of the query painting
            k: Number of similar paintings to retrieve
            figsize: Figure size
            save_path: Optional path to save the visualization
        """
        # Find similar paintings
        result = self.find_similar_paintings(query_idx, k)
        
        # Load images
        query_img = Image.open(result['query_path']).convert('RGB')
        similar_imgs = [Image.open(path).convert('RGB') for path in result['similar_paths']]
        
        # Create figure
        fig, axes = plt.subplots(1, k + 1, figsize=figsize)
        
        # Plot query image
        axes[0].imshow(query_img)
        axes[0].set_title('Query')
        axes[0].axis('off')
        
        # Plot similar images
        for i, (img, sim) in enumerate(zip(similar_imgs, result['similarities'])):
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(f'Similarity: {sim:.3f}')
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def save_system(self, output_path: str) -> None:
        """
        Save the painting similarity system to disk.
        
        Args:
            output_path: Path to save the system
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # We don't pickle the whole system because it might be too large
        # Instead, we save the components separately
        
        system_data = {
            'features': self.features,
            'image_paths': self.image_paths,
            'metadata': self.metadata
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(system_data, f)
        
        # Save similarity model separately
        model_path = os.path.join(os.path.dirname(output_path), 'similarity_model.pkl')
        self.similarity_model.save_model(model_path)
        
        logger.info(f"Saved painting similarity system to {output_path}")
    
    @staticmethod
    def load_system(input_path: str, model_path: Optional[str] = None) -> 'PaintingSimilaritySystem':
        """
        Load a painting similarity system from disk.
        
        Args:
            input_path: Path to load the system data from
            model_path: Optional path to load the similarity model from (if None, use dirname(input_path)/similarity_model.pkl)
            
        Returns:
            Loaded painting similarity system
        """
        with open(input_path, 'rb') as f:
            system_data = pickle.load(f)
        
        # Load similarity model
        if model_path is None:
            model_path = os.path.join(os.path.dirname(input_path), 'similarity_model.pkl')
        
        similarity_model = SimilarityModel.load_model(model_path)
        
        # Create system
        system = PaintingSimilaritySystem(
            similarity_model=similarity_model,
            features=system_data['features'],
            image_paths=system_data['image_paths'],
            metadata=system_data['metadata']
        )
        
        logger.info(f"Loaded painting similarity system from {input_path}")
        
        return system


def create_similarity_model(model_type: str, **kwargs) -> SimilarityModel:
    """
    Create a similarity model.
    
    Args:
        model_type: Type of similarity model ('cosine' or 'faiss')
        **kwargs: Additional arguments for the similarity model
        
    Returns:
        Similarity model instance
    """
    if model_type == 'cosine':
        return CosineSimilarityModel(**kwargs)
    elif model_type == 'faiss':
        return FaissIndexModel(**kwargs)
    else:
        raise ValueError(f"Unsupported similarity model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    print("Similarity Model module")
    print("Use this module to find similar paintings based on extracted features.")
