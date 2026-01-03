"""
Data module for loading and processing data.

This module provides functionality for:
- API clients for external data sources
- Data extraction and persistence
- Raw data management
- Data transformation and validation
"""

from .api_client import APIFootballClient, FootballDataCSVClient
from .extractor import DataExtractor
from .transformer import DataTransformer

__all__ = [
    'APIFootballClient',
    'FootballDataCSVClient',
    'DataExtractor',
    'DataTransformer',
]
