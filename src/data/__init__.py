"""
Data module for loading and processing data.

This module provides functionality for:
- API clients for external data sources
- Data extraction and persistence
- Raw data management
"""

from .api_client import APIFootballClient, FootballDataCSVClient
from .extractor import DataExtractor

__all__ = [
    'APIFootballClient',
    'FootballDataCSVClient',
    'DataExtractor',
]
