"""
Feature engineering module.

This module provides functionality for:
- Creating rolling statistics features
- Match-level comparative features
- Rest days and fixture congestion features
- Head-to-head statistics
- Complete feature engineering pipeline
"""

from .engineering import FeatureEngineer

__all__ = [
    'FeatureEngineer',
]
