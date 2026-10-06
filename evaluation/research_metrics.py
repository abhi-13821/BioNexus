"""
Research-grade Evaluation Metrics for BioNexus
For publication-quality results
"""

import time
import logging
import statistics
from typing import List, Dict, Any, Tuple
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class ResearchMetrics:
    """Comprehensive metrics for research paper quality evaluation."""
    
    @staticmethod
    def precision(retrieved: List[str], relevant: List[str]) -> float:
        """Precision: Retrieved documents that are relevant."""
        if not retrieved:
            return 0.0
        relevant_retrieved = [r for r in retrieved if r in relevant]
        return len(relevant_retrieved) / len(retrieved)
    
    @staticmethod
    def recall(retrieved: List[str], relevant: List[str]) -> float:
        """Recall: Relevant documents that were retrieved."""
        if not relevant:
            return 0.0
        relevant_retrieved = [r for r in retrieved if r in relevant]
        return len(relevant_retrieved) / len(relevant)
    
    @staticmethod
    def f1(precision: float, recall: float) -> float:
        """F1 Score: Harmonic mean of precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mean_average_precision(results: List[Dict]) -> float:
        """
        Mean Average Precision (MAP) for ranked retrieval.
        """
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
            
            total_ap += ap / len(relevant)
        
        return total_ap / len(results) if results else 0.0
    
    @staticmethod
    def ndcg(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain.
        """
        if not retrieved or not relevant:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(retrieved[:k], 1):
            # Binary relevance (1 if relevant, 0 otherwise)
            relevance = 1 if item in relevant else 0
            dcg += relevance / math.log2(i + 1)
        
        # Calculate IDCG (ideal DCG)
        ideal = [1] * min(len(relevant), k) + [0] * (k - min(len(relevant), k))
        idcg = 0.0
        for i, rel in enumerate(ideal, 1):
            idcg += rel / math.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(results: List[Dict]) -> float:
        """
        Mean Reciprocal Rank (MRR) - measures first relevant result position.
        """
        total_rr = 0.0
        for result in results:
            retrieved = result.get("retrieved", [])
            relevant = result.get("relevant", [])
            
            for i, item in enumerate(retrieved, 1):
                if item in relevant:
                    total_rr += 1.0 / i
                    break
        
        return total_rr / len(results) if results else 0.0
    
    @staticmethod
    def latency_stats(times: List[float]) -> Dict[str, float]:
        """
        Calculate latency statistics: mean, median, p95, p99.
        """
        if not times:
            return {"mean": 0, "median": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
        
        sorted_times = sorted(times)
        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "p95": sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 1 else times[0],
            "p99": sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 1 else times[0],
            "min": min(times),
            "max": max(times),
        }
    
    @staticmethod
    def error_analysis(results: List[Dict]) -> Dict[str, float]:
        """
        Analyze failure types and rates.
        """
        total = len(results)
        if total == 0:
            return {"success_rate": 0, "failure_rate": 0}
        
        failures = [r for r in results if not r.get("success", False)]
        failure_types = {}
        for failure in failures:
            error_type = failure.get("error_type", "unknown")
            failure_types[error_type] = failure_types.get(error_type, 0) + 1
        
        return {
            "success_rate": (total - len(failures)) / total,
            "failure_rate": len(failures) / total,
            "failure_types": failure_types,
        }