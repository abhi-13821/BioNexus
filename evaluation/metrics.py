"""
Evaluation Metrics for BioNexus
Measures: Accuracy, Precision, Recall, F1 Score, Latency
"""

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Metrics:
    """Calculate evaluation metrics for agents."""
    
    @staticmethod
    def accuracy(predictions: List[Any], ground_truth: List[Any]) -> float:
        """Calculate accuracy."""
        if not predictions or len(predictions) != len(ground_truth):
            return 0.0
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        return correct / len(predictions) if predictions else 0.0
    
    @staticmethod
    def precision(retrieved: List[Any], relevant: List[Any]) -> float:
        """Calculate precision."""
        if not retrieved:
            return 0.0
        relevant_retrieved = [r for r in retrieved if r in relevant]
        return len(relevant_retrieved) / len(retrieved)
    
    @staticmethod
    def recall(retrieved: List[Any], relevant: List[Any]) -> float:
        """Calculate recall."""
        if not relevant:
            return 0.0
        relevant_retrieved = [r for r in retrieved if r in relevant]
        return len(relevant_retrieved) / len(relevant)
    
    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """Calculate F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mean_average_precision(results: List[Dict]) -> float:
        """Calculate mean average precision for ranked results."""
        if not results:
            return 0.0
        
        total_ap = 0.0
        for result in results:
            retrieved = result.get("retrieved", [])
            relevant = result.get("relevant", [])
            
            if not relevant:
                continue
            
            ap = 0.0
            relevant_count = 0
            for i, item in enumerate(retrieved, 1):
                if item in relevant:
                    relevant_count += 1
                    ap += relevant_count / i
            
            total_ap += ap / len(relevant) if relevant else 0
        
        return total_ap / len(results) if results else 0.0
    
    @staticmethod
    def measure_latency(func, *args, **kwargs) -> Dict[str, Any]:
        """Measure execution latency."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        return {
            "result": result,
            "latency_ms": (end_time - start_time) * 1000,
            "latency_seconds": end_time - start_time
        }
    
    @staticmethod
    def ndcg(retrieved: List[Any], relevant: List[Any], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not retrieved or not relevant:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(retrieved[:k], 1):
            if item in relevant:
                dcg += 1 / (i + 1)
        
        # Calculate IDCG (ideal DCG)
        ideal_retrieved = relevant[:k]
        idcg = 0.0
        for i in range(1, min(len(ideal_retrieved), k) + 1):
            idcg += 1 / (i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0