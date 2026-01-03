"""
Model training and evaluation module.

This module provides functionality for:
- Poisson regression models for goals prediction
- Model training and evaluation
- Probability calculations
- Model persistence
"""

from .poisson_model import PoissonGoalsModel

__all__ = [
    'PoissonGoalsModel',
]
