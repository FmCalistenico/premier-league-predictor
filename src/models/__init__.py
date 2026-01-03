"""
Model training and evaluation module.

This module provides functionality for:
- Poisson regression models for goals prediction
- Model training and evaluation
- Probability calculations
- Model persistence
- Comprehensive evaluation metrics
- Time series cross-validation
"""

from .poisson_model import PoissonGoalsModel
from .evaluator import ModelEvaluator, TimeSeriesValidator

__all__ = [
    'PoissonGoalsModel',
    'ModelEvaluator',
    'TimeSeriesValidator',
]
