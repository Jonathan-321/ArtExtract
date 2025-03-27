"""
Similarity metrics module for the ArtExtract project.
This module implements metrics for evaluating painting similarity models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os
import pandas as pd
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimilarityEvaluator:
    """
    Evaluator for similarity models.
    """
    
    def __init__(self):
        """Initialize the similarity evaluator."""
        logger.info("Initialized SimilarityEvaluator")
    
    def precision_at_k(self, 
                      relevant_items: List[int], 
                      recommended_items: List[int], 
                      k: int) -> float:
        """
        Calculate precision@k.
        
        Args:
            relevant_items: List of relevant item indices
            recommended_items: List of recommended item indices
            k: Number of items to consider
            
        Returns:
            Precision@k value
        """
        # Ensure k is not larger than the number of recommended items
        k = min(k, len(recommended_items))
        
        # Get the first k recommended items
        recommended_k = recommended_items[:k]
        
        # Count relevant items in the recommendations
        relevant_count = sum(1 for item in recommended_k if item in relevant_items)
        
        # Calculate precision@k
        precision = relevant_count / k if k > 0 else 0.0
        
        return precision
    
    def average_precision(self, 
                         relevant_items: List[int], 
                         recommended_items: List[int]) -> float:
        """
        Calculate average precision.
        
        Args:
            relevant_items: List of relevant item indices
            recommended_items: List of recommended item indices
            
        Returns:
            Average precision value
        """
        # Initialize variables
        relevant_count = 0
        sum_precision = 0.0
        
        # Calculate precision at each position where a relevant item is found
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                sum_precision += precision_at_i
        
        # Calculate average precision
        average_precision = sum_precision / len(relevant_items) if relevant_items else 0.0
        
        return average_precision
    
    def mean_average_precision(self, 
                              all_relevant_items: List[List[int]], 
                              all_recommended_items: List[List[int]]) -> float:
        """
        Calculate mean average precision (MAP).
        
        Args:
            all_relevant_items: List of lists of relevant item indices for each query
            all_recommended_items: List of lists of recommended item indices for each query
            
        Returns:
            MAP value
        """
        # Calculate average precision for each query
        aps = [
            self.average_precision(relevant, recommended)
            for relevant, recommended in zip(all_relevant_items, all_recommended_items)
        ]
        
        # Calculate mean average precision
        map_value = sum(aps) / len(aps) if aps else 0.0
        
        return map_value
    
    def ndcg_at_k(self, 
                 relevant_items: List[int], 
                 recommended_items: List[int], 
                 relevance_scores: Dict[int, float], 
                 k: int) -> float:
        """
        Calculate normalized discounted cumulative gain (NDCG) at k.
        
        Args:
            relevant_items: List of relevant item indices
            recommended_items: List of recommended item indices
            relevance_scores: Dictionary mapping item indices to relevance scores
            k: Number of items to consider
            
        Returns:
            NDCG@k value
        """
        # Ensure k is not larger than the number of recommended items
        k = min(k, len(recommended_items))
        
        # Get the first k recommended items
        recommended_k = recommended_items[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            if item in relevant_items:
                # Get relevance score (default to 1.0 if not provided)
                rel = relevance_scores.get(item, 1.0)
                # Calculate DCG contribution (log base 2)
                dcg += rel / np.log2(i + 2)  # +2 because i is 0-indexed
        
        # Calculate ideal DCG (IDCG)
        # Sort relevant items by relevance score
        sorted_relevant = sorted(
            [item for item in relevant_items if item in relevance_scores],
            key=lambda x: relevance_scores.get(x, 1.0),
            reverse=True
        )
        
        # If no relevance scores are provided, assume all relevant items have score 1.0
        if not sorted_relevant:
            sorted_relevant = relevant_items
        
        # Calculate IDCG
        idcg = 0.0
        for i, item in enumerate(sorted_relevant[:k]):
            # Get relevance score (default to 1.0 if not provided)
            rel = relevance_scores.get(item, 1.0)
            # Calculate IDCG contribution
            idcg += rel / np.log2(i + 2)  # +2 because i is 0-indexed
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return ndcg
    
    def mean_reciprocal_rank(self, 
                            all_relevant_items: List[List[int]], 
                            all_recommended_items: List[List[int]]) -> float:
        """
        Calculate mean reciprocal rank (MRR).
        
        Args:
            all_relevant_items: List of lists of relevant item indices for each query
            all_recommended_items: List of lists of recommended item indices for each query
            
        Returns:
            MRR value
        """
        # Calculate reciprocal rank for each query
        rrs = []
        for relevant, recommended in zip(all_relevant_items, all_recommended_items):
            # Find the rank of the first relevant item
            for i, item in enumerate(recommended):
                if item in relevant:
                    # Reciprocal rank is 1 / (rank)
                    rrs.append(1.0 / (i + 1))
                    break
            else:
                # No relevant item found
                rrs.append(0.0)
        
        # Calculate mean reciprocal rank
        mrr = sum(rrs) / len(rrs) if rrs else 0.0
        
        return mrr
    
    def evaluate_similarity_model(self, 
                                 all_relevant_items: List[List[int]], 
                                 all_recommended_items: List[List[int]], 
                                 relevance_scores: Optional[List[Dict[int, float]]] = None, 
                                 k_values: List[int] = [5, 10, 20]) -> Dict[str, Any]:
        """
        Evaluate a similarity model using multiple metrics.
        
        Args:
            all_relevant_items: List of lists of relevant item indices for each query
            all_recommended_items: List of lists of recommended item indices for each query
            relevance_scores: Optional list of dictionaries mapping item indices to relevance scores for each query
            k_values: List of k values for precision@k and NDCG@k
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Initialize results
        results = {}
        
        # Calculate MAP
        map_value = self.mean_average_precision(all_relevant_items, all_recommended_items)
        results['map'] = map_value
        
        # Calculate MRR
        mrr = self.mean_reciprocal_rank(all_relevant_items, all_recommended_items)
        results['mrr'] = mrr
        
        # Calculate precision@k for each k
        precision_at_k_values = {}
        for k in k_values:
            precision_values = [
                self.precision_at_k(relevant, recommended, k)
                for relevant, recommended in zip(all_relevant_items, all_recommended_items)
            ]
            precision_at_k_values[f'precision@{k}'] = sum(precision_values) / len(precision_values) if precision_values else 0.0
        
        results['precision_at_k'] = precision_at_k_values
        
        # Calculate NDCG@k for each k if relevance scores are provided
        if relevance_scores:
            ndcg_at_k_values = {}
            for k in k_values:
                ndcg_values = [
                    self.ndcg_at_k(relevant, recommended, scores, k)
                    for relevant, recommended, scores in zip(all_relevant_items, all_recommended_items, relevance_scores)
                ]
                ndcg_at_k_values[f'ndcg@{k}'] = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0
            
            results['ndcg_at_k'] = ndcg_at_k_values
        
        return results
    
    def plot_precision_at_k(self, 
                           results: Dict[str, Any], 
                           figsize: Tuple[int, int] = (10, 6),
                           save_path: Optional[str] = None) -> None:
        """
        Plot precision@k values.
        
        Args:
            results: Dictionary with evaluation metrics
            figsize: Figure size
            save_path: Optional path to save the plot
        """
        # Extract precision@k values
        precision_at_k = results.get('precision_at_k', {})
        
        if not precision_at_k:
            logger.warning("No precision@k values found in results")
            return
        
        # Extract k values and precision values
        k_values = [int(k.split('@')[1]) for k in precision_at_k.keys()]
        precision_values = list(precision_at_k.values())
        
        # Sort by k
        sorted_indices = np.argsort(k_values)
        k_values = [k_values[i] for i in sorted_indices]
        precision_values = [precision_values[i] for i in sorted_indices]
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.plot(k_values, precision_values, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('k')
        plt.ylabel('Precision@k')
        plt.title('Precision@k for Different k Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Precision@k plot saved to {save_path}")
        
        plt.show()
    
    def plot_ndcg_at_k(self, 
                      results: Dict[str, Any], 
                      figsize: Tuple[int, int] = (10, 6),
                      save_path: Optional[str] = None) -> None:
        """
        Plot NDCG@k values.
        
        Args:
            results: Dictionary with evaluation metrics
            figsize: Figure size
            save_path: Optional path to save the plot
        """
        # Extract NDCG@k values
        ndcg_at_k = results.get('ndcg_at_k', {})
        
        if not ndcg_at_k:
            logger.warning("No NDCG@k values found in results")
            return
        
        # Extract k values and NDCG values
        k_values = [int(k.split('@')[1]) for k in ndcg_at_k.keys()]
        ndcg_values = list(ndcg_at_k.values())
        
        # Sort by k
        sorted_indices = np.argsort(k_values)
        k_values = [k_values[i] for i in sorted_indices]
        ndcg_values = [ndcg_values[i] for i in sorted_indices]
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.plot(k_values, ndcg_values, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('k')
        plt.ylabel('NDCG@k')
        plt.title('NDCG@k for Different k Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"NDCG@k plot saved to {save_path}")
        
        plt.show()
    
    def save_results_to_json(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Dictionary with evaluation metrics
            output_path: Path to save the results
        """
        # Convert numpy types to Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"Evaluation results saved to {output_path}")


class UserEvaluationTool:
    """
    Tool for collecting and analyzing user evaluations of similarity results.
    """
    
    def __init__(self):
        """Initialize the user evaluation tool."""
        self.evaluations = []
        logger.info("Initialized UserEvaluationTool")
    
    def add_evaluation(self, 
                      query_id: str, 
                      recommended_ids: List[str], 
                      relevance_scores: List[int],
                      user_id: Optional[str] = None,
                      comments: Optional[str] = None) -> None:
        """
        Add a user evaluation.
        
        Args:
            query_id: ID of the query painting
            recommended_ids: List of IDs of recommended paintings
            relevance_scores: List of relevance scores (typically 0-5) for each recommended painting
            user_id: Optional ID of the user who provided the evaluation
            comments: Optional comments from the user
        """
        evaluation = {
            'query_id': query_id,
            'recommended_ids': recommended_ids,
            'relevance_scores': relevance_scores,
            'user_id': user_id,
            'comments': comments,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.evaluations.append(evaluation)
        logger.info(f"Added user evaluation for query {query_id}")
    
    def get_average_relevance(self) -> Dict[str, float]:
        """
        Calculate average relevance scores across all evaluations.
        
        Returns:
            Dictionary mapping query IDs to average relevance scores
        """
        # Group evaluations by query ID
        query_groups = {}
        for eval in self.evaluations:
            query_id = eval['query_id']
            if query_id not in query_groups:
                query_groups[query_id] = []
            
            # Calculate average relevance for this evaluation
            avg_relevance = sum(eval['relevance_scores']) / len(eval['relevance_scores']) if eval['relevance_scores'] else 0.0
            query_groups[query_id].append(avg_relevance)
        
        # Calculate average relevance for each query
        avg_relevance_by_query = {
            query_id: sum(scores) / len(scores) if scores else 0.0
            for query_id, scores in query_groups.items()
        }
        
        return avg_relevance_by_query
    
    def plot_average_relevance(self, 
                              figsize: Tuple[int, int] = (12, 6),
                              save_path: Optional[str] = None) -> None:
        """
        Plot average relevance scores by query.
        
        Args:
            figsize: Figure size
            save_path: Optional path to save the plot
        """
        # Get average relevance by query
        avg_relevance = self.get_average_relevance()
        
        if not avg_relevance:
            logger.warning("No evaluations found")
            return
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Query': list(avg_relevance.keys()),
            'Average Relevance': list(avg_relevance.values())
        })
        
        # Sort by average relevance
        df = df.sort_values('Average Relevance', ascending=False)
        
        # Create plot
        plt.figure(figsize=figsize)
        ax = sns.barplot(x='Query', y='Average Relevance', data=df)
        
        # Add value labels
        for i, p in enumerate(ax.patches):
            ax.annotate(
                f'{p.get_height():.2f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='bottom',
                fontsize=10,
                rotation=0
            )
        
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Query ID')
        plt.ylabel('Average Relevance Score')
        plt.title('Average Relevance Scores by Query')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Average relevance plot saved to {save_path}")
        
        plt.show()
    
    def export_evaluations(self, output_path: str) -> None:
        """
        Export evaluations to a CSV file.
        
        Args:
            output_path: Path to save the evaluations
        """
        # Convert evaluations to DataFrame
        df = pd.DataFrame(self.evaluations)
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(self.evaluations)} evaluations to {output_path}")
    
    def import_evaluations(self, input_path: str) -> None:
        """
        Import evaluations from a CSV file.
        
        Args:
            input_path: Path to load the evaluations from
        """
        # Load from CSV
        df = pd.read_csv(input_path)
        
        # Convert to list of dictionaries
        evaluations = df.to_dict('records')
        
        # Add to existing evaluations
        self.evaluations.extend(evaluations)
        
        logger.info(f"Imported {len(evaluations)} evaluations from {input_path}")


if __name__ == "__main__":
    # Example usage
    print("Similarity Metrics module")
    print("Use this module to evaluate painting similarity models.")
