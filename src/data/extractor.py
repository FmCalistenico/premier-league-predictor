"""
Data extraction orchestration module.
Coordinates extraction from multiple sources and handles data persistence.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
import pandas as pd

from ..utils import LoggerMixin, Config
from .api_client import APIFootballClient, FootballDataCSVClient


class DataExtractor(LoggerMixin):
    """
    Orchestrates data extraction from multiple sources.

    Features:
    - Multi-source extraction (API + CSV)
    - Automatic versioning by date
    - Metadata tracking
    - Error recovery
    - Data validation
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize data extractor.

        Args:
            config: Configuration instance. If None, creates new one.
        """
        self.config = config or Config()

        # Initialize clients
        self.api_client = APIFootballClient()
        self.csv_client = FootballDataCSVClient()

        # Setup paths
        self.raw_data_path = self.config.data_raw_path
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("DataExtractor initialized")

    def extract_from_api(
        self,
        league_id: Optional[int] = None,
        season: Optional[str] = None,
        include_statistics: bool = False
    ) -> Dict[str, Any]:
        """
        Extract data from API-Football.

        Args:
            league_id: League ID (default from config)
            season: Season year (default from config)
            include_statistics: Whether to fetch team statistics

        Returns:
            Dictionary containing extracted data

        Raises:
            Exception: If extraction fails
        """
        league_id = league_id or self.config.league_id
        season = season or self.config.current_season.split('-')[0]  # Get year

        self.logger.info(f"Starting API extraction for league {league_id}, season {season}")

        extracted_data = {
            'metadata': {
                'source': 'api-football',
                'league_id': league_id,
                'season': season,
                'extraction_date': datetime.now().isoformat(),
                'extractor_version': '1.0.0'
            },
            'fixtures': [],
            'standings': [],
            'team_statistics': []
        }

        try:
            # Extract fixtures
            self.logger.info("Extracting fixtures...")
            fixtures = self.api_client.get_fixtures(
                league_id=league_id,
                season=season
            )
            extracted_data['fixtures'] = fixtures
            self.logger.info(f"Extracted {len(fixtures)} fixtures")

            # Extract standings
            self.logger.info("Extracting standings...")
            standings = self.api_client.get_standings(
                league_id=league_id,
                season=season
            )
            extracted_data['standings'] = standings
            self.logger.info(f"Extracted standings")

            # Extract team statistics if requested
            if include_statistics and fixtures:
                self.logger.info("Extracting team statistics...")

                # Get unique team IDs from fixtures
                team_ids = set()
                for fixture in fixtures:
                    team_ids.add(fixture['teams']['home']['id'])
                    team_ids.add(fixture['teams']['away']['id'])

                team_stats = []
                for team_id in team_ids:
                    try:
                        stats = self.api_client.get_team_statistics(
                            team_id=team_id,
                            league_id=league_id,
                            season=season
                        )
                        team_stats.append(stats)
                        self.logger.debug(f"Extracted statistics for team {team_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to extract stats for team {team_id}: {str(e)}")

                extracted_data['team_statistics'] = team_stats
                self.logger.info(f"Extracted statistics for {len(team_stats)} teams")

            # Update metadata with counts
            extracted_data['metadata']['fixtures_count'] = len(fixtures)
            extracted_data['metadata']['teams_count'] = len(extracted_data['team_statistics'])
            extracted_data['metadata']['extraction_status'] = 'success'

            self.logger.info("API extraction completed successfully")
            return extracted_data

        except Exception as e:
            self.logger.error(f"API extraction failed: {str(e)}", exc_info=True)
            extracted_data['metadata']['extraction_status'] = 'failed'
            extracted_data['metadata']['error'] = str(e)
            raise

    def extract_from_csv(
        self,
        seasons: Optional[List[str]] = None,
        division: str = "E0"
    ) -> Dict[str, Any]:
        """
        Extract data from football-data.co.uk CSV files.

        Args:
            seasons: List of season codes. If None, gets current season.
            division: Division code (E0 = Premier League)

        Returns:
            Dictionary containing extracted data

        Raises:
            Exception: If extraction fails
        """
        if seasons is None:
            # Default to last 3 seasons
            seasons = ['2223', '2324', '2425']

        self.logger.info(f"Starting CSV extraction for seasons {seasons}")

        extracted_data = {
            'metadata': {
                'source': 'football-data.co.uk',
                'seasons': seasons,
                'division': division,
                'extraction_date': datetime.now().isoformat(),
                'extractor_version': '1.0.0'
            },
            'data': None
        }

        try:
            # Extract data
            if len(seasons) == 1:
                df = self.csv_client.get_season_data(season=seasons[0], division=division)
            else:
                df = self.csv_client.get_multiple_seasons(seasons=seasons, division=division)

            extracted_data['data'] = df

            # Update metadata
            extracted_data['metadata']['matches_count'] = len(df)
            extracted_data['metadata']['columns'] = df.columns.tolist()
            extracted_data['metadata']['date_range'] = {
                'start': df['Date'].min() if 'Date' in df.columns else None,
                'end': df['Date'].max() if 'Date' in df.columns else None
            }
            extracted_data['metadata']['extraction_status'] = 'success'

            self.logger.info(f"CSV extraction completed. Extracted {len(df)} matches")
            return extracted_data

        except Exception as e:
            self.logger.error(f"CSV extraction failed: {str(e)}", exc_info=True)
            extracted_data['metadata']['extraction_status'] = 'failed'
            extracted_data['metadata']['error'] = str(e)
            raise

    def save_raw_data(
        self,
        data: Any,
        source: Literal['api', 'csv'],
        metadata: Dict[str, Any],
        custom_name: Optional[str] = None
    ) -> Path:
        """
        Save raw data with versioning and metadata.

        Args:
            data: Data to save (DataFrame, dict, or list)
            source: Data source ('api' or 'csv')
            metadata: Metadata dictionary
            custom_name: Custom filename. If None, auto-generates with timestamp.

        Returns:
            Path to saved file
        """
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

        if custom_name:
            base_name = custom_name
        else:
            base_name = f"{source}_data_{timestamp}"

        # Create source-specific directory
        source_dir = self.raw_data_path / source
        source_dir.mkdir(parents=True, exist_ok=True)

        # Save data
        if isinstance(data, pd.DataFrame):
            # Save as CSV
            data_file = source_dir / f"{base_name}.csv"
            data.to_csv(data_file, index=False)
            self.logger.info(f"Saved CSV data to {data_file}")

        elif isinstance(data, (dict, list)):
            # Save as JSON
            data_file = source_dir / f"{base_name}.json"
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved JSON data to {data_file}")

        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Save metadata
        metadata_file = source_dir / f"{base_name}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved metadata to {metadata_file}")

        return data_file

    def run_extraction(
        self,
        sources: List[Literal['api', 'csv']] = ['csv'],
        league_id: Optional[int] = None,
        season: Optional[str] = None,
        csv_seasons: Optional[List[str]] = None,
        save_data: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete extraction process from specified sources.

        Args:
            sources: List of sources to extract from ('api', 'csv', or both)
            league_id: League ID for API extraction
            season: Season for API extraction
            csv_seasons: Seasons for CSV extraction
            save_data: Whether to save extracted data

        Returns:
            Dictionary with extraction results

        Example:
            >>> extractor = DataExtractor()
            >>> results = extractor.run_extraction(
            ...     sources=['csv', 'api'],
            ...     csv_seasons=['2223', '2324', '2425']
            ... )
        """
        self.logger.info(f"Starting extraction from sources: {sources}")

        results = {
            'extraction_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'sources': sources,
            'results': {},
            'errors': {},
            'files_saved': {}
        }

        # Extract from CSV
        if 'csv' in sources:
            try:
                self.logger.info("Starting CSV extraction...")
                csv_data = self.extract_from_csv(seasons=csv_seasons)
                results['results']['csv'] = csv_data['metadata']

                if save_data:
                    data_file = self.save_raw_data(
                        data=csv_data['data'],
                        source='csv',
                        metadata=csv_data['metadata']
                    )
                    results['files_saved']['csv'] = str(data_file)

            except Exception as e:
                self.logger.error(f"CSV extraction failed: {str(e)}")
                results['errors']['csv'] = str(e)

        # Extract from API
        if 'api' in sources:
            try:
                self.logger.info("Starting API extraction...")
                api_data = self.extract_from_api(
                    league_id=league_id,
                    season=season,
                    include_statistics=True
                )
                results['results']['api'] = api_data['metadata']

                if save_data:
                    data_file = self.save_raw_data(
                        data=api_data,
                        source='api',
                        metadata=api_data['metadata']
                    )
                    results['files_saved']['api'] = str(data_file)

            except Exception as e:
                self.logger.error(f"API extraction failed: {str(e)}")
                results['errors']['api'] = str(e)

        # Summary
        success_count = len(results['results'])
        error_count = len(results['errors'])

        self.logger.info(
            f"Extraction completed. "
            f"Successful: {success_count}, Failed: {error_count}"
        )

        # Save extraction summary
        if save_data:
            summary_file = self.raw_data_path / f"extraction_summary_{results['extraction_id']}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved extraction summary to {summary_file}")

        return results
