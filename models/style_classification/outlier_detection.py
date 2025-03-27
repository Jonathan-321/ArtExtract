"""
Outlier detection module for art style classification.
This module implements methods for detecting outliers in art style classifications.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Outlier detection using Isolation Forest algorithm.
    
    Isolation Forest is an unsupervised learning algorithm that isolates observations
    by randomly selecting a feature and then randomly selecting a split value
    between the maximum and minimum values of the selected feature.
    """
    
    def __init__(self, 
                n_estimators: int = 100, 
                contamination: float = 0.1, 
                random_state: int = 42):
        """
        Initialize the Isolation Forest detector.
        
        Args:
            n_estimators: Number of base estimators in the ensemble
            contamination: Expected proportion of outliers in the data
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        
        logger.info(f"Initialized Isolation Forest detector with {n_estimators} estimators")
    
    def fit(self, features: np.ndarray) -> 'IsolationForestDetector':
        """
        Fit the detector to the data.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            Self
        """
        logger.info(f"Fitting Isolation Forest detector to {features.shape[0]} samples")
        self.model.fit(features)
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict if samples are outliers.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples,) where 1 is inlier and -1 is outlier
        """
        return self.model.predict(features)
    
    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """
        Compute the anomaly score of each sample.
        
        The lower, the more abnormal. Negative scores represent outliers,
        positive scores represent inliers.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples,) with anomaly scores
        """
        return self.model.decision_function(features)
    
    def get_outlier_indices(self, features: np.ndarray) -> np.ndarray:
        """
        Get indices of outlier samples.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            Array of indices of outlier samples
        """
        predictions = self.predict(features)
        return np.where(predictions == -1)[0]


class LocalOutlierFactorDetector:
    """
    Outlier detection using Local Outlier Factor (LOF) algorithm.
    
    LOF measures the local deviation of the density of a given sample with
    respect to its neighbors.
    """
    
    def __init__(self, 
                n_neighbors: int = 20, 
                contamination: float = 0.1):
        """
        Initialize the LOF detector.
        
        Args:
            n_neighbors: Number of neighbors to consider
            contamination: Expected proportion of outliers in the data
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            n_jobs=-1
        )
        
        logger.info(f"Initialized Local Outlier Factor detector with {n_neighbors} neighbors")
    
    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Fit the detector to the data and predict if samples are outliers.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples,) where 1 is inlier and -1 is outlier
        """
        logger.info(f"Fitting LOF detector to {features.shape[0]} samples and predicting")
        return self.model.fit_predict(features)
    
    def get_outlier_indices(self, features: np.ndarray) -> np.ndarray:
        """
        Get indices of outlier samples.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            Array of indices of outlier samples
        """
        predictions = self.fit_predict(features)
        return np.where(predictions == -1)[0]


class AutoencoderDetector:
    """
    Outlier detection using an autoencoder.
    
    The autoencoder learns to reconstruct normal samples. Outliers will have
    higher reconstruction error.
    """
    
    def __init__(self, 
                input_dim: int,
                hidden_dims: List[int] = [128, 64, 32],
                latent_dim: int = 16,
                dropout: float = 0.2,
                learning_rate: float = 1e-3,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the autoencoder detector.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions
            latent_dim: Latent dimension
            dropout: Dropout probability
            learning_rate: Learning rate for optimization
            device: Device to run the model on
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device
        
        # Build encoder and decoder
        self._build_model()
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Define loss function
        self.criterion = nn.MSELoss()
        
        logger.info(f"Initialized Autoencoder detector with latent dimension {latent_dim}")
    
    def _build_model(self) -> None:
        """Build the autoencoder model."""
        # Define encoder layers
        encoder_layers = []
        
        # Input layer
        encoder_layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Dropout(self.dropout))
        
        # Hidden layers
        for i in range(len(self.hidden_dims) - 1):
            encoder_layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(self.dropout))
        
        # Latent layer
        encoder_layers.append(nn.Linear(self.hidden_dims[-1], self.latent_dim))
        
        # Define decoder layers
        decoder_layers = []
        
        # Latent to hidden
        decoder_layers.append(nn.Linear(self.latent_dim, self.hidden_dims[-1]))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Dropout(self.dropout))
        
        # Hidden layers (reversed)
        for i in range(len(self.hidden_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i-1]))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(self.dropout))
        
        # Output layer
        decoder_layers.append(nn.Linear(self.hidden_dims[0], self.input_dim))
        
        # Create encoder and decoder
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Create full model
        self.model = nn.Sequential(
            self.encoder,
            self.decoder
        )
    
    def fit(self, 
           features: np.ndarray, 
           epochs: int = 100, 
           batch_size: int = 64, 
           validation_split: float = 0.1) -> Dict[str, List[float]]:
        """
        Fit the autoencoder to the data.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training history
        """
        # Convert to torch tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Split into train and validation
        val_size = int(validation_split * len(features))
        train_size = len(features) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            features_tensor, [train_size, val_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute loss
                loss = self.criterion(outputs, batch)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * batch.size(0)
            
            train_loss /= train_size
            history['train_loss'].append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    # Move batch to device
                    batch = batch.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch)
                    
                    # Compute loss
                    loss = self.criterion(outputs, batch)
                    
                    # Update statistics
                    val_loss += loss.item() * batch.size(0)
            
            val_loss /= val_size
            history['val_loss'].append(val_loss)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return history
    
    def compute_reconstruction_error(self, features: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for each sample.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples,) with reconstruction errors
        """
        # Convert to torch tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Compute reconstructions
        with torch.no_grad():
            reconstructions = self.model(features_tensor)
        
        # Compute reconstruction errors
        errors = torch.mean(torch.square(reconstructions - features_tensor), dim=1).cpu().numpy()
        
        return errors
    
    def get_outlier_indices(self, 
                          features: np.ndarray, 
                          threshold: Optional[float] = None, 
                          contamination: float = 0.1) -> np.ndarray:
        """
        Get indices of outlier samples based on reconstruction error.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            threshold: Threshold for outlier detection (if None, use contamination)
            contamination: Expected proportion of outliers in the data
            
        Returns:
            Array of indices of outlier samples
        """
        # Compute reconstruction errors
        errors = self.compute_reconstruction_error(features)
        
        # Determine threshold
        if threshold is None:
            # Use contamination parameter
            threshold = np.percentile(errors, 100 * (1 - contamination))
        
        # Find outliers
        outlier_indices = np.where(errors > threshold)[0]
        
        return outlier_indices


def visualize_outliers(features: np.ndarray, 
                      labels: np.ndarray, 
                      outlier_indices: np.ndarray,
                      class_names: List[str],
                      method: str = 'tsne',
                      save_path: Optional[str] = None) -> None:
    """
    Visualize outliers in a 2D plot.
    
    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Label array of shape (n_samples,)
        outlier_indices: Array of indices of outlier samples
        class_names: List of class names
        method: Dimensionality reduction method ('tsne' or 'pca')
        save_path: Optional path to save the plot
    """
    # Create mask for outliers
    outlier_mask = np.zeros(len(features), dtype=bool)
    outlier_mask[outlier_indices] = True
    
    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(features)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(features)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot inliers
    for i, class_name in enumerate(class_names):
        mask = (~outlier_mask) & (labels == i)
        plt.scatter(
            reduced_features[mask, 0],
            reduced_features[mask, 1],
            alpha=0.7,
            label=f"{class_name} (inlier)"
        )
    
    # Plot outliers
    for i, class_name in enumerate(class_names):
        mask = outlier_mask & (labels == i)
        if np.any(mask):
            plt.scatter(
                reduced_features[mask, 0],
                reduced_features[mask, 1],
                marker='x',
                s=100,
                linewidths=2,
                label=f"{class_name} (outlier)"
            )
    
    plt.title(f"Outlier Detection Visualization ({method.upper()})")
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Outlier visualization saved to {save_path}")
    
    plt.show()


def analyze_outliers(features: np.ndarray, 
                    labels: np.ndarray, 
                    outlier_indices: np.ndarray,
                    class_names: List[str]) -> Dict[str, Any]:
    """
    Analyze outliers by class.
    
    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Label array of shape (n_samples,)
        outlier_indices: Array of indices of outlier samples
        class_names: List of class names
        
    Returns:
        Dictionary with outlier analysis
    """
    # Create mask for outliers
    outlier_mask = np.zeros(len(features), dtype=bool)
    outlier_mask[outlier_indices] = True
    
    # Count outliers by class
    outliers_by_class = {}
    for i, class_name in enumerate(class_names):
        class_mask = labels == i
        class_outliers = np.sum(outlier_mask & class_mask)
        class_total = np.sum(class_mask)
        class_outlier_rate = class_outliers / class_total if class_total > 0 else 0
        
        outliers_by_class[class_name] = {
            'total_samples': int(class_total),
            'outliers': int(class_outliers),
            'outlier_rate': float(class_outlier_rate)
        }
    
    # Overall statistics
    total_samples = len(features)
    total_outliers = len(outlier_indices)
    overall_outlier_rate = total_outliers / total_samples
    
    # Prepare result
    result = {
        'total_samples': int(total_samples),
        'total_outliers': int(total_outliers),
        'overall_outlier_rate': float(overall_outlier_rate),
        'outliers_by_class': outliers_by_class
    }
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Outlier Detection module")
    print("Use this module to detect outliers in art style classifications.")
