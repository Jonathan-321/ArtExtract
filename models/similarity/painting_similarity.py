"""
Painting Similarity Detection module for the ArtExtract project.
This module implements methods to find similarities between paintings based on various features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import faiss
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PaintingSimilarityDetector:
    """
    Model for detecting similarities between paintings.
    Uses a pre-trained CNN to extract features and computes similarity metrics.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        feature_layer: str = 'avgpool',
        device: Optional[torch.device] = None,
        use_face_detection: bool = True,
        use_pose_estimation: bool = True
    ):
        """
        Initialize the PaintingSimilarityDetector.
        
        Args:
            backbone: CNN backbone architecture ('resnet18', 'resnet50', 'vgg16', etc.)
            pretrained: Whether to use pretrained weights for the backbone
            feature_layer: Layer to extract features from
            device: Device to use
            use_face_detection: Whether to use face detection for portrait similarity
            use_pose_estimation: Whether to use pose estimation for pose similarity
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone_name = backbone
        self.feature_layer = feature_layer
        self.use_face_detection = use_face_detection
        self.use_pose_estimation = use_pose_estimation
        
        # Initialize CNN backbone
        if backbone.startswith('resnet'):
            if backbone == 'resnet18':
                base_model = models.resnet18(pretrained=pretrained)
            elif backbone == 'resnet34':
                base_model = models.resnet34(pretrained=pretrained)
            elif backbone == 'resnet50':
                base_model = models.resnet50(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported ResNet variant: {backbone}")
                
        elif backbone.startswith('vgg'):
            if backbone == 'vgg16':
                base_model = models.vgg16(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported VGG variant: {backbone}")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Create feature extractor
        self.model, self.feature_dim = self._create_feature_extractor(base_model)
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Initialize face detector if needed
        self.face_detector = None
        if use_face_detection:
            self._init_face_detector()
            
        # Initialize pose estimator if needed
        self.pose_estimator = None
        if use_pose_estimation:
            self._init_pose_estimator()
            
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize FAISS index for fast similarity search
        self.faiss_index = None
        self.painting_ids = []
        
    def _create_feature_extractor(self, base_model):
        """
        Create a feature extractor from the base model.
        
        Args:
            base_model: Base CNN model
            
        Returns:
            tuple: (feature_extractor, feature_dimension)
        """
        if self.backbone_name.startswith('resnet'):
            if self.feature_layer == 'avgpool':
                # Remove the final fully connected layer
                feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
                feature_dim = base_model.fc.in_features
            else:
                raise ValueError(f"Unsupported feature layer for ResNet: {self.feature_layer}")
                
        elif self.backbone_name.startswith('vgg'):
            if self.feature_layer == 'features':
                feature_extractor = base_model.features
                feature_dim = 512 * 7 * 7  # For 224x224 input
            else:
                raise ValueError(f"Unsupported feature layer for VGG: {self.feature_layer}")
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
            
        return feature_extractor, feature_dim
    
    def _init_face_detector(self):
        """Initialize face detector."""
        try:
            # Use OpenCV's face detector
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(face_cascade_path)
            logger.info("Face detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            self.use_face_detection = False
    
    def _init_pose_estimator(self):
        """Initialize pose estimator."""
        try:
            # For simplicity, we'll use a pre-trained model from OpenCV
            # In a real implementation, you might want to use a more sophisticated model
            # like OpenPose, HRNet, or MediaPipe
            self.pose_estimator = cv2.dnn.readNetFromTensorflow(
                'models/similarity/pose_estimation_model.pb'
            )
            logger.info("Pose estimator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pose estimator: {e}")
            self.use_pose_estimation = False
    
    def extract_features(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract features from an image tensor.
        
        Args:
            img_tensor: Image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Feature tensor
        """
        with torch.no_grad():
            features = self.model(img_tensor.to(self.device))
            # Flatten if needed
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
            return features
    
    def extract_face_features(self, img_np: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face features from an image.
        
        Args:
            img_np: Image as numpy array (BGR format for OpenCV)
            
        Returns:
            Face features or None if no face detected
        """
        if not self.use_face_detection or self.face_detector is None:
            return None
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
            
        # Extract the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Extract face ROI
        face_roi = img_np[y:y+h, x:x+w]
        
        # Resize to a standard size
        face_roi = cv2.resize(face_roi, (128, 128))
        
        # Convert to RGB and normalize
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_roi = face_roi.astype(np.float32) / 255.0
        
        return face_roi
    
    def extract_pose_features(self, img_np: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose features from an image.
        
        Args:
            img_np: Image as numpy array (BGR format for OpenCV)
            
        Returns:
            Pose features or None if pose estimation failed
        """
        if not self.use_pose_estimation or self.pose_estimator is None:
            return None
            
        try:
            # Prepare input for the pose estimator
            height, width = img_np.shape[:2]
            input_blob = cv2.dnn.blobFromImage(
                img_np, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False
            )
            
            # Run pose estimation
            self.pose_estimator.setInput(input_blob)
            output = self.pose_estimator.forward()
            
            # Process output to get keypoints
            keypoints = []
            for i in range(output.shape[1]):
                # Get heatmap for this keypoint
                heatmap = output[0, i, :, :]
                
                # Find the position of the maximum value
                _, confidence, _, point = cv2.minMaxLoc(heatmap)
                
                # Scale the point to the original image size
                x = int(point[0] * width / heatmap.shape[1])
                y = int(point[1] * height / heatmap.shape[0])
                
                # Add keypoint if confidence is high enough
                if confidence > 0.1:
                    keypoints.append((x, y, confidence))
                else:
                    keypoints.append((0, 0, 0))  # Placeholder for missing keypoint
            
            # Convert to numpy array
            keypoints = np.array(keypoints)
            
            return keypoints
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            return None
    
    def compute_similarity(
        self, 
        features1: torch.Tensor, 
        features2: torch.Tensor,
        method: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            method: Similarity method ('cosine', 'euclidean', 'dot')
            
        Returns:
            Similarity score
        """
        # Convert to numpy if needed
        if isinstance(features1, torch.Tensor):
            features1 = features1.cpu().numpy()
        if isinstance(features2, torch.Tensor):
            features2 = features2.cpu().numpy()
            
        # Ensure features are 2D
        if len(features1.shape) == 1:
            features1 = features1.reshape(1, -1)
        if len(features2.shape) == 1:
            features2 = features2.reshape(1, -1)
            
        if method == 'cosine':
            return cosine_similarity(features1, features2)[0, 0]
        elif method == 'euclidean':
            return -np.linalg.norm(features1 - features2)  # Negative so higher is more similar
        elif method == 'dot':
            return np.dot(features1, features2.T)[0, 0]
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
    
    def build_index(self, features: np.ndarray, ids: List[str]):
        """
        Build a FAISS index for fast similarity search.
        
        Args:
            features: Feature matrix of shape (n_samples, feature_dim)
            ids: List of painting IDs corresponding to the features
        """
        # Normalize features for cosine similarity
        features = features.astype(np.float32)
        faiss.normalize_L2(features)
        
        # Build index
        self.faiss_index = faiss.IndexFlatIP(features.shape[1])  # Inner product for cosine similarity
        self.faiss_index.add(features)
        self.painting_ids = ids
        
        logger.info(f"Built FAISS index with {len(ids)} paintings")
    
    def find_similar(
        self, 
        query_features: np.ndarray, 
        k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """
        Find similar paintings using the FAISS index.
        
        Args:
            query_features: Query feature vector
            k: Number of similar paintings to return
            
        Returns:
            Tuple of (painting_ids, similarity_scores)
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_index first.")
            
        # Normalize query features
        query_features = query_features.astype(np.float32)
        faiss.normalize_L2(query_features)
        
        # Search index
        k = min(k, len(self.painting_ids))
        scores, indices = self.faiss_index.search(query_features, k)
        
        # Get painting IDs and scores
        similar_ids = [self.painting_ids[idx] for idx in indices[0]]
        similarity_scores = scores[0].tolist()
        
        return similar_ids, similarity_scores
    
    def visualize_similarity(
        self,
        query_img: np.ndarray,
        similar_imgs: List[np.ndarray],
        similarity_scores: List[float],
        save_path: Optional[str] = None
    ):
        """
        Visualize query image and similar images with their similarity scores.
        
        Args:
            query_img: Query image
            similar_imgs: List of similar images
            similarity_scores: List of similarity scores
            save_path: Optional path to save the visualization
        """
        n_similar = len(similar_imgs)
        fig, axes = plt.subplots(1, n_similar + 1, figsize=(4 * (n_similar + 1), 4))
        
        # Plot query image
        axes[0].imshow(query_img)
        axes[0].set_title('Query Image')
        axes[0].axis('off')
        
        # Plot similar images
        for i, (img, score) in enumerate(zip(similar_imgs, similarity_scores)):
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(f'Similarity: {score:.3f}')
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Visualization saved to {save_path}")
            
        plt.close()
    
    def analyze_similarity_distribution(
        self,
        all_features: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Analyze the distribution of similarities in the dataset.
        
        Args:
            all_features: Feature matrix of shape (n_samples, feature_dim)
            save_path: Optional path to save the visualization
        """
        # Normalize features
        features = all_features.astype(np.float32)
        faiss.normalize_L2(features)
        
        # Compute pairwise similarities
        n_samples = min(1000, len(features))  # Limit to 1000 samples for efficiency
        indices = np.random.choice(len(features), n_samples, replace=False)
        sample_features = features[indices]
        
        similarities = sample_features @ sample_features.T
        
        # Plot histogram of similarities
        plt.figure(figsize=(10, 6))
        plt.hist(similarities.flatten(), bins=50, alpha=0.7)
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Similarity Scores')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Similarity distribution saved to {save_path}")
            
        plt.close()
        
        # Return statistics
        return {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'median': float(np.median(similarities))
        }
    
    def visualize_feature_space(
        self,
        features: np.ndarray,
        labels: Optional[List[str]] = None,
        method: str = 'pca',
        save_path: Optional[str] = None
    ):
        """
        Visualize the feature space using dimensionality reduction.
        
        Args:
            features: Feature matrix of shape (n_samples, feature_dim)
            labels: Optional list of labels for coloring
            method: Dimensionality reduction method ('pca', 'tsne')
            save_path: Optional path to save the visualization
        """
        # Normalize features
        features = features.astype(np.float32)
        
        # Apply dimensionality reduction
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            reduced_features = reducer.fit_transform(features)
            
            # Calculate explained variance
            explained_variance = reducer.explained_variance_ratio_
            explained_variance_sum = sum(explained_variance)
            
            title = f'PCA Visualization (Explained Variance: {explained_variance_sum:.2f})'
            
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            reduced_features = reducer.fit_transform(features)
            
            title = 't-SNE Visualization'
            
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            # Convert labels to numeric for coloring
            unique_labels = list(set(labels))
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            label_ids = [label_to_id[label] for label in labels]
            
            scatter = plt.scatter(
                reduced_features[:, 0],
                reduced_features[:, 1],
                c=label_ids,
                cmap='viridis',
                alpha=0.7
            )
            
            # Add legend
            if len(unique_labels) <= 20:  # Only show legend if not too many labels
                plt.legend(
                    handles=scatter.legend_elements()[0],
                    labels=unique_labels,
                    title='Labels',
                    loc='best'
                )
        else:
            plt.scatter(
                reduced_features[:, 0],
                reduced_features[:, 1],
                alpha=0.7
            )
        
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Feature space visualization saved to {save_path}")
            
        plt.close()
