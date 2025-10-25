"""
Evaluation Framework for Academic Research
"""

from .academic_metrics import StandardizedMetrics, NetworkPerformanceMetrics, ConvergenceMetrics, ScalabilityMetrics
from .statistical_analysis import AcademicStatisticalAnalyzer, ComparisonResult, MultipleComparisonResult, DistributionAnalysis

__all__ = [
    'StandardizedMetrics',
    'NetworkPerformanceMetrics',
    'ConvergenceMetrics',
    'ScalabilityMetrics',
    'AcademicStatisticalAnalyzer',
    'ComparisonResult',
    'MultipleComparisonResult',
    'DistributionAnalysis'
]