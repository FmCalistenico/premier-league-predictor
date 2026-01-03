"""
Data transformation module.
Handles parsing, cleaning, and standardization of raw data from multiple sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Literal
from datetime import datetime

from ..utils import LoggerMixin


class DataTransformer(LoggerMixin):
    """
    Transforms raw data from various sources into standardized format.

    Features:
    - Parse API and CSV data formats
    - Create target variables (over/under goals)
    - Validate and clean data
    - Standardize team names
    - Complete transformation pipeline
    """

    # Team name standardization mapping
    TEAM_NAME_MAPPING = {
        # Manchester teams
        'Man City': 'Manchester City',
        'Man Utd': 'Manchester United',
        'Manchester Utd': 'Manchester United',

        # London teams
        'Spurs': 'Tottenham',
        'Tottenham Hotspur': 'Tottenham',

        # Other teams
        'Newcastle': 'Newcastle United',
        'Brighton': 'Brighton and Hove Albion',
        'West Ham': 'West Ham United',
        'Wolves': 'Wolverhampton Wanderers',
        'Nott\'m Forest': 'Nottingham Forest',
        'Nottm Forest': 'Nottingham Forest',
        'Leicester': 'Leicester City',
        'Leeds': 'Leeds United',
        'Sheffield Utd': 'Sheffield United',
        'West Brom': 'West Bromwich Albion',
    }

    # Standard column names
    STANDARD_COLUMNS = [
        'fixture_id',
        'date',
        'season',
        'home_team_name',
        'away_team_name',
        'home_goals',
        'away_goals',
        'total_goals',
    ]

    def __init__(self):
        """Initialize data transformer."""
        self.logger.info("DataTransformer initialized")

    def parse_api_fixtures(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert API-Football JSON response to DataFrame.

        Args:
            raw_data: Raw API response dictionary with 'fixtures' key

        Returns:
            DataFrame with standardized columns

        Example:
            >>> transformer = DataTransformer()
            >>> df = transformer.parse_api_fixtures(api_response)
        """
        self.logger.info("Parsing API fixtures data")

        fixtures = raw_data.get('fixtures', [])

        if not fixtures:
            self.logger.warning("No fixtures found in API data")
            return pd.DataFrame(columns=self.STANDARD_COLUMNS)

        parsed_data = []

        for fixture in fixtures:
            try:
                # Extract match info
                fixture_info = fixture.get('fixture', {})
                teams = fixture.get('teams', {})
                goals = fixture.get('goals', {})
                league = fixture.get('league', {})

                # Parse fixture data
                parsed_fixture = {
                    'fixture_id': fixture_info.get('id'),
                    'date': fixture_info.get('date'),
                    'season': league.get('season'),
                    'home_team_name': teams.get('home', {}).get('name'),
                    'away_team_name': teams.get('away', {}).get('name'),
                    'home_goals': goals.get('home'),
                    'away_goals': goals.get('away'),
                }

                # Calculate total goals
                if parsed_fixture['home_goals'] is not None and parsed_fixture['away_goals'] is not None:
                    parsed_fixture['total_goals'] = parsed_fixture['home_goals'] + parsed_fixture['away_goals']
                else:
                    parsed_fixture['total_goals'] = None

                parsed_data.append(parsed_fixture)

            except Exception as e:
                self.logger.warning(f"Failed to parse fixture: {str(e)}")
                continue

        df = pd.DataFrame(parsed_data)

        # Convert date to datetime
        if 'date' in df.columns and not df.empty:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        self.logger.info(f"Parsed {len(df)} fixtures from API data")

        return df

    def parse_csv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize CSV data from football-data.co.uk.

        Args:
            df: Raw CSV DataFrame

        Returns:
            DataFrame with standardized columns

        CSV Column Mapping:
            - Date → date
            - HomeTeam → home_team_name
            - AwayTeam → away_team_name
            - FTHG (Full Time Home Goals) → home_goals
            - FTAG (Full Time Away Goals) → away_goals
        """
        self.logger.info(f"Parsing CSV data with {len(df)} rows")

        # Create copy to avoid modifying original
        df_parsed = df.copy()

        # Column mapping
        column_mapping = {
            'Date': 'date',
            'HomeTeam': 'home_team_name',
            'AwayTeam': 'away_team_name',
            'FTHG': 'home_goals',
            'FTAG': 'away_goals',
        }

        # Rename columns
        df_parsed = df_parsed.rename(columns=column_mapping)

        # Ensure required columns exist
        required_cols = ['date', 'home_team_name', 'away_team_name', 'home_goals', 'away_goals']
        missing_cols = [col for col in required_cols if col not in df_parsed.columns]

        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert date to datetime
        df_parsed['date'] = pd.to_datetime(df_parsed['date'], format='%d/%m/%Y', errors='coerce')

        # Convert goals to numeric
        df_parsed['home_goals'] = pd.to_numeric(df_parsed['home_goals'], errors='coerce')
        df_parsed['away_goals'] = pd.to_numeric(df_parsed['away_goals'], errors='coerce')

        # Calculate total goals
        df_parsed['total_goals'] = df_parsed['home_goals'] + df_parsed['away_goals']

        # Create fixture_id from date and teams (for CSV data)
        df_parsed['fixture_id'] = (
            df_parsed['date'].dt.strftime('%Y%m%d') + '_' +
            df_parsed['home_team_name'].str.replace(' ', '') + '_' +
            df_parsed['away_team_name'].str.replace(' ', '')
        )

        # Add season if exists, otherwise derive from date
        if 'season' not in df_parsed.columns:
            df_parsed['season'] = df_parsed['date'].apply(self._derive_season)

        # Select standard columns
        available_cols = [col for col in self.STANDARD_COLUMNS if col in df_parsed.columns]
        df_parsed = df_parsed[available_cols]

        self.logger.info(f"Parsed {len(df_parsed)} rows from CSV data")

        return df_parsed

    def _derive_season(self, date: pd.Timestamp) -> str:
        """
        Derive season from date (e.g., 2024-08-01 → '2024').

        Args:
            date: Match date

        Returns:
            Season string (start year)
        """
        if pd.isna(date):
            return None

        # Season starts in August
        if date.month >= 8:
            return str(date.year)
        else:
            return str(date.year - 1)

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create over/under target variables based on total goals.

        Creates binary columns:
        - over_0.5: 1 if total_goals > 0.5, else 0
        - over_1.5: 1 if total_goals > 1.5, else 0
        - over_2.5: 1 if total_goals > 2.5, else 0 (main target)
        - over_3.5: 1 if total_goals > 3.5, else 0
        - over_4.5: 1 if total_goals > 4.5, else 0

        Args:
            df: DataFrame with 'total_goals' column

        Returns:
            DataFrame with target variable columns added
        """
        self.logger.info("Creating target variables")

        df_with_targets = df.copy()

        thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]

        for threshold in thresholds:
            col_name = f'over_{threshold}'
            df_with_targets[col_name] = (df_with_targets['total_goals'] > threshold).astype(int)

        # Set NaN for matches without goals data
        mask_no_goals = df_with_targets['total_goals'].isna()
        for threshold in thresholds:
            col_name = f'over_{threshold}'
            df_with_targets.loc[mask_no_goals, col_name] = np.nan

        # Count target distribution
        target_counts = df_with_targets['over_2.5'].value_counts()
        self.logger.info(f"Target distribution (over_2.5): {target_counts.to_dict()}")

        return df_with_targets

    def validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate and clean data.

        Validations:
        1. Check for nulls in critical columns
        2. Validate goals >= 0
        3. Remove duplicates by fixture_id
        4. Validate dates in reasonable range
        5. Check for invalid team names

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (cleaned_df, list_of_issues)
        """
        self.logger.info(f"Validating data: {len(df)} rows")

        issues = []
        df_clean = df.copy()
        initial_rows = len(df_clean)

        # 1. Check nulls in critical columns
        critical_columns = ['date', 'home_team_name', 'away_team_name']
        for col in critical_columns:
            if col in df_clean.columns:
                null_count = df_clean[col].isna().sum()
                if null_count > 0:
                    issue = f"Found {null_count} null values in {col}"
                    issues.append(issue)
                    self.logger.warning(issue)
                    df_clean = df_clean.dropna(subset=[col])

        # 2. Validate goals >= 0
        for col in ['home_goals', 'away_goals', 'total_goals']:
            if col in df_clean.columns:
                invalid_goals = df_clean[df_clean[col] < 0]
                if len(invalid_goals) > 0:
                    issue = f"Found {len(invalid_goals)} rows with negative {col}"
                    issues.append(issue)
                    self.logger.warning(issue)
                    df_clean = df_clean[df_clean[col] >= 0]

        # 3. Remove duplicates by fixture_id
        if 'fixture_id' in df_clean.columns:
            duplicates = df_clean.duplicated(subset=['fixture_id'], keep='first')
            dup_count = duplicates.sum()
            if dup_count > 0:
                issue = f"Found {dup_count} duplicate fixtures"
                issues.append(issue)
                self.logger.warning(issue)
                df_clean = df_clean[~duplicates]

        # 4. Validate dates in reasonable range (1990-2030)
        if 'date' in df_clean.columns:
            min_date = pd.Timestamp('1990-01-01')
            max_date = pd.Timestamp('2030-12-31')

            invalid_dates = df_clean[
                (df_clean['date'] < min_date) | (df_clean['date'] > max_date)
            ]

            if len(invalid_dates) > 0:
                issue = f"Found {len(invalid_dates)} rows with invalid dates"
                issues.append(issue)
                self.logger.warning(issue)
                df_clean = df_clean[
                    (df_clean['date'] >= min_date) & (df_clean['date'] <= max_date)
                ]

        # 5. Check for empty team names
        for col in ['home_team_name', 'away_team_name']:
            if col in df_clean.columns:
                empty_teams = df_clean[df_clean[col].str.strip() == '']
                if len(empty_teams) > 0:
                    issue = f"Found {len(empty_teams)} rows with empty {col}"
                    issues.append(issue)
                    self.logger.warning(issue)
                    df_clean = df_clean[df_clean[col].str.strip() != '']

        # 6. Check for inconsistent total_goals
        if all(col in df_clean.columns for col in ['home_goals', 'away_goals', 'total_goals']):
            calculated_total = df_clean['home_goals'] + df_clean['away_goals']
            inconsistent = df_clean[df_clean['total_goals'] != calculated_total]

            if len(inconsistent) > 0:
                issue = f"Found {len(inconsistent)} rows with inconsistent total_goals"
                issues.append(issue)
                self.logger.warning(issue)
                # Fix inconsistent total_goals
                df_clean['total_goals'] = df_clean['home_goals'] + df_clean['away_goals']

        rows_removed = initial_rows - len(df_clean)

        if rows_removed > 0:
            self.logger.info(f"Removed {rows_removed} invalid rows during validation")

        if not issues:
            self.logger.info("Data validation passed with no issues")
        else:
            self.logger.warning(f"Data validation found {len(issues)} issues")

        return df_clean, issues

    def standardize_team_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize team names to standard format.

        Args:
            df: DataFrame with team name columns

        Returns:
            DataFrame with standardized team names

        Example:
            "Man City" → "Manchester City"
            "Spurs" → "Tottenham"
        """
        self.logger.info("Standardizing team names")

        df_std = df.copy()
        replacements_made = 0

        for col in ['home_team_name', 'away_team_name']:
            if col in df_std.columns:
                # Apply mapping
                original_values = df_std[col].copy()
                df_std[col] = df_std[col].replace(self.TEAM_NAME_MAPPING)

                # Count replacements
                changed = (original_values != df_std[col]).sum()
                replacements_made += changed

        if replacements_made > 0:
            self.logger.info(f"Standardized {replacements_made} team names")
        else:
            self.logger.info("No team names needed standardization")

        # Log unique teams
        if 'home_team_name' in df_std.columns and 'away_team_name' in df_std.columns:
            unique_teams = set(df_std['home_team_name'].unique()) | set(df_std['away_team_name'].unique())
            self.logger.debug(f"Found {len(unique_teams)} unique teams: {sorted(unique_teams)}")

        return df_std

    def transform(
        self,
        raw_data: Any,
        source: Literal['api', 'csv']
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete transformation pipeline.

        Pipeline steps:
        1. Parse data based on source
        2. Standardize team names
        3. Create target variables
        4. Validate and clean data

        Args:
            raw_data: Raw data (DataFrame for CSV, Dict for API)
            source: Data source type ('api' or 'csv')

        Returns:
            Tuple of (transformed_df, transformation_metadata)

        Example:
            >>> transformer = DataTransformer()
            >>> df, metadata = transformer.transform(raw_csv_data, source='csv')
        """
        self.logger.info(f"Starting transformation pipeline for {source} data")

        metadata = {
            'source': source,
            'transformation_date': datetime.now().isoformat(),
            'pipeline_steps': [],
            'issues': [],
            'initial_rows': 0,
            'final_rows': 0,
        }

        try:
            # Step 1: Parse data
            self.logger.info("Step 1: Parsing data")
            if source == 'api':
                df = self.parse_api_fixtures(raw_data)
            elif source == 'csv':
                df = self.parse_csv_data(raw_data)
            else:
                raise ValueError(f"Unsupported source: {source}")

            metadata['initial_rows'] = len(df)
            metadata['pipeline_steps'].append('parse')
            self.logger.info(f"Parsed {len(df)} rows")

            if df.empty:
                self.logger.warning("No data after parsing")
                metadata['final_rows'] = 0
                return df, metadata

            # Step 2: Standardize team names
            self.logger.info("Step 2: Standardizing team names")
            df = self.standardize_team_names(df)
            metadata['pipeline_steps'].append('standardize_teams')

            # Step 3: Create target variables
            self.logger.info("Step 3: Creating target variables")
            df = self.create_target_variables(df)
            metadata['pipeline_steps'].append('create_targets')

            # Step 4: Validate and clean
            self.logger.info("Step 4: Validating data")
            df, issues = self.validate_data(df)
            metadata['pipeline_steps'].append('validate')
            metadata['issues'] = issues

            metadata['final_rows'] = len(df)

            # Add summary statistics
            metadata['summary'] = {
                'date_range': {
                    'start': df['date'].min().isoformat() if 'date' in df.columns and not df.empty else None,
                    'end': df['date'].max().isoformat() if 'date' in df.columns and not df.empty else None,
                },
                'total_matches': len(df),
                'unique_teams': len(set(df['home_team_name'].unique()) | set(df['away_team_name'].unique())) if 'home_team_name' in df.columns else 0,
                'avg_goals_per_match': df['total_goals'].mean() if 'total_goals' in df.columns else None,
                'over_2.5_pct': (df['over_2.5'].sum() / len(df) * 100) if 'over_2.5' in df.columns and len(df) > 0 else None,
            }

            self.logger.info(
                f"Transformation complete: {metadata['initial_rows']} → {metadata['final_rows']} rows"
            )
            self.logger.info(f"Summary: {metadata['summary']}")

            return df, metadata

        except Exception as e:
            self.logger.error(f"Transformation failed: {str(e)}", exc_info=True)
            metadata['error'] = str(e)
            raise
