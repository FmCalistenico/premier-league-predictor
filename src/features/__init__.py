"""
Feature engineering module.

This module provides functionality for:
- Creating rolling statistics features
- Match-level comparative features
- Rest days and fixture congestion features
- Head-to-head statistics
- Complete feature engineering pipeline

V2 enhancements:
- Bias reduction (removed circular features)
- Ratio-based features (relative to league average)
- Momentum and volatility features
- Context-aware features (derbies, top clashes)
- Feature validation (VIF and correlation checks)
"""

from .engineering import FeatureEngineer
from .engineering_v2 import FeatureEngineerV2, run_feature_engineering_v2

__all__ = [
    'FeatureEngineer',
    'FeatureEngineerV2',
    'run_feature_engineering_v2',
]
