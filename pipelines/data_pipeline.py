"""
Complete data pipeline for Premier League prediction.
Orchestrates extraction, transformation, and feature engineering.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from datetime import datetime

from src.utils import LoggerMixin, Config
from src.data import DataExtractor, DataTransformer
from src.features import FeatureEngineer


class DataPipeline(LoggerMixin):
    """
    Complete data pipeline orchestration.

    Pipeline stages:
    1. Extraction: Fetch data from API or CSV sources
    2. Transformation: Clean and standardize data
    3. Feature Engineering: Create predictive features
    4. Save: Store final dataset for modeling

    Example:
        >>> pipeline = DataPipeline()
        >>> df_final = pipeline.run_full_pipeline(source='csv')
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Config] = None):
        """
        Initialize data pipeline.

        Args:
            api_key: API key for data extraction (optional)
            config: Configuration instance (optional)
        """
        self.config = config or Config()

        # Initialize components
        self.extractor = DataExtractor(self.config)
        self.transformer = DataTransformer()
        self.feature_engineer = FeatureEngineer()

        # Setup output directories
        self.raw_dir = self.config.data_raw_path
        self.processed_dir = self.config.data_processed_path
        self.final_dir = self.config.data_final_path

        for directory in [self.raw_dir, self.processed_dir, self.final_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.logger.info("DataPipeline initialized")
        self.logger.info(f"Raw data: {self.raw_dir}")
        self.logger.info(f"Processed data: {self.processed_dir}")
        self.logger.info(f"Final data: {self.final_dir}")

    def run_extraction(
        self,
        source: Literal['csv', 'api'] = 'csv',
        **kwargs
    ) -> Any:
        """
        Run data extraction stage.

        Args:
            source: Data source ('csv' or 'api')
            **kwargs: Additional arguments for extraction
                - csv_seasons: List of season codes for CSV
                - league_id: League ID for API
                - season: Season for API

        Returns:
            Raw data (DataFrame for CSV, dict for API)

        Example:
            >>> raw_data = pipeline.run_extraction(
            ...     source='csv',
            ...     csv_seasons=['2223', '2324', '2425']
            ... )
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: DATA EXTRACTION")
        self.logger.info("=" * 60)

        try:
            if source == 'csv':
                csv_seasons = kwargs.get('csv_seasons', ['2223', '2324', '2425'])
                self.logger.info(f"Extracting CSV data for seasons: {csv_seasons}")

                extracted_data = self.extractor.extract_from_csv(seasons=csv_seasons)
                raw_data = extracted_data['data']

                self.logger.info(f"Extracted {len(raw_data)} matches from CSV")

            elif source == 'api':
                league_id = kwargs.get('league_id', self.config.league_id)
                season = kwargs.get('season', self.config.current_season.split('-')[0])

                self.logger.info(f"Extracting API data for league {league_id}, season {season}")

                extracted_data = self.extractor.extract_from_api(
                    league_id=league_id,
                    season=season,
                    include_statistics=True
                )
                raw_data = extracted_data

                self.logger.info(f"Extracted {extracted_data['metadata']['fixtures_count']} fixtures from API")

            else:
                raise ValueError(f"Invalid source: {source}")

            self.logger.info("✓ Extraction stage completed successfully")
            return raw_data

        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}", exc_info=True)
            raise

    def run_transformation(
        self,
        raw_data: Any,
        source: Literal['csv', 'api']
    ) -> pd.DataFrame:
        """
        Run data transformation stage.

        Args:
            raw_data: Raw data from extraction
            source: Data source type

        Returns:
            Cleaned and standardized DataFrame

        Example:
            >>> df_clean = pipeline.run_transformation(raw_data, source='csv')
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: DATA TRANSFORMATION")
        self.logger.info("=" * 60)

        try:
            self.logger.info(f"Transforming {source} data")

            df_transformed, metadata = self.transformer.transform(raw_data, source=source)

            # Log transformation summary
            self.logger.info(f"Initial rows: {metadata['initial_rows']}")
            self.logger.info(f"Final rows: {metadata['final_rows']}")
            self.logger.info(f"Rows removed: {metadata['initial_rows'] - metadata['final_rows']}")

            if metadata['issues']:
                self.logger.warning(f"Issues found: {len(metadata['issues'])}")
                for issue in metadata['issues']:
                    self.logger.warning(f"  - {issue}")

            # Save to processed directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            processed_file = self.processed_dir / f'transformed_{source}_{timestamp}.csv'

            df_transformed.to_csv(processed_file, index=False)
            self.logger.info(f"Saved transformed data to {processed_file}")

            self.logger.info("✓ Transformation stage completed successfully")
            return df_transformed

        except Exception as e:
            self.logger.error(f"Transformation failed: {str(e)}", exc_info=True)
            raise

    def run_feature_engineering(
        self,
        df: pd.DataFrame,
        rolling_windows: Optional[list] = None,
        h2h_window: int = 5
    ) -> pd.DataFrame:
        """
        Run feature engineering stage.

        Args:
            df: Transformed DataFrame
            rolling_windows: Window sizes for rolling features
            h2h_window: Window for head-to-head features

        Returns:
            DataFrame with engineered features

        Example:
            >>> df_features = pipeline.run_feature_engineering(df_transformed)
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3: FEATURE ENGINEERING")
        self.logger.info("=" * 60)

        try:
            if rolling_windows is None:
                rolling_windows = [3, 5, 10]

            self.logger.info(f"Engineering features with windows: {rolling_windows}")

            df_features = self.feature_engineer.engineer_features(
                df,
                rolling_windows=rolling_windows,
                h2h_window=h2h_window,
                min_matches_required=5
            )

            feature_names = self.feature_engineer.get_feature_names()

            self.logger.info(f"Created {len(feature_names)} features")
            self.logger.info(f"Final dataset: {len(df_features)} matches")

            self.logger.info("✓ Feature engineering stage completed successfully")
            return df_features

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}", exc_info=True)
            raise

    def save_final_dataset(
        self,
        df: pd.DataFrame,
        version: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Save final dataset with versioning.

        Saves three versions:
        1. Timestamped version: training_data_YYYYMMDD_HHMMSS.parquet
        2. Latest version: training_data_latest.parquet
        3. CSV for inspection: training_data_latest.csv

        Args:
            df: Final DataFrame with features
            version: Custom version string (optional)

        Returns:
            Dictionary with paths to saved files

        Example:
            >>> files = pipeline.save_final_dataset(df_features)
            >>> print(files['parquet'])
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 4: SAVING FINAL DATASET")
        self.logger.info("=" * 60)

        try:
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if version is None:
                version = timestamp

            # File paths
            parquet_versioned = self.final_dir / f'training_data_{version}.parquet'
            parquet_latest = self.final_dir / 'training_data_latest.parquet'
            csv_latest = self.final_dir / 'training_data_latest.csv'

            # Save versioned parquet
            self.logger.info(f"Saving versioned dataset: {parquet_versioned}")
            df.to_parquet(parquet_versioned, index=False)

            # Save latest parquet
            self.logger.info(f"Saving latest dataset: {parquet_latest}")
            df.to_parquet(parquet_latest, index=False)

            # Save CSV for inspection
            self.logger.info(f"Saving CSV for inspection: {csv_latest}")
            df.to_csv(csv_latest, index=False)

            # Calculate file sizes
            files = {
                'parquet': parquet_versioned,
                'parquet_latest': parquet_latest,
                'csv': csv_latest
            }

            self.logger.info("\nFiles saved:")
            for file_type, file_path in files.items():
                size_kb = file_path.stat().st_size / 1024
                self.logger.info(f"  {file_type:15s}: {file_path} ({size_kb:.1f} KB)")

            # Log dataset summary
            self.log_dataset_summary(df)

            self.logger.info("✓ Dataset saved successfully")
            return files

        except Exception as e:
            self.logger.error(f"Failed to save dataset: {str(e)}", exc_info=True)
            raise

    def log_dataset_summary(self, df: pd.DataFrame) -> None:
        """
        Log comprehensive dataset summary.

        Logs:
        - Total matches
        - Date range
        - Unique teams
        - Target distribution
        - Available features
        - Missing data report

        Args:
            df: Final DataFrame

        Example:
            >>> pipeline.log_dataset_summary(df_final)
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("DATASET SUMMARY")
        self.logger.info("=" * 60)

        # Basic statistics
        self.logger.info(f"\nDataset Size:")
        self.logger.info(f"  Total matches: {len(df)}")
        self.logger.info(f"  Total columns: {len(df.columns)}")

        # Date range
        if 'date' in df.columns:
            self.logger.info(f"\nDate Range:")
            self.logger.info(f"  From: {df['date'].min()}")
            self.logger.info(f"  To:   {df['date'].max()}")
            self.logger.info(f"  Span: {(df['date'].max() - df['date'].min()).days} days")

        # Teams
        if 'home_team_name' in df.columns and 'away_team_name' in df.columns:
            unique_teams = set(df['home_team_name'].unique()) | set(df['away_team_name'].unique())
            self.logger.info(f"\nTeams:")
            self.logger.info(f"  Unique teams: {len(unique_teams)}")

        # Target distribution
        if 'over_2.5' in df.columns:
            target_dist = df['over_2.5'].value_counts()
            target_pct = df['over_2.5'].mean() * 100

            self.logger.info(f"\nTarget Distribution (over_2.5):")
            self.logger.info(f"  Under 2.5: {target_dist.get(0, 0)} ({100-target_pct:.1f}%)")
            self.logger.info(f"  Over 2.5:  {target_dist.get(1, 0)} ({target_pct:.1f}%)")

        # Features
        feature_cols = self.feature_engineer.get_feature_names()
        if feature_cols:
            self.logger.info(f"\nFeatures:")
            self.logger.info(f"  Total features: {len(feature_cols)}")

            # Categorize features
            rolling_features = [f for f in feature_cols if 'L' in f and ('home_' in f or 'away_' in f)]
            match_features = [f for f in feature_cols if any(x in f for x in ['expected', 'form', 'strength', 'combined'])]
            rest_features = [f for f in feature_cols if 'rest' in f]
            h2h_features = [f for f in feature_cols if 'h2h' in f]

            self.logger.info(f"    - Rolling: {len(rolling_features)}")
            self.logger.info(f"    - Match: {len(match_features)}")
            self.logger.info(f"    - Rest: {len(rest_features)}")
            self.logger.info(f"    - H2H: {len(h2h_features)}")

        # Missing data report
        missing_counts = df.isnull().sum()
        missing_features = missing_counts[missing_counts > 0]

        if len(missing_features) > 0:
            self.logger.warning(f"\nMissing Data:")
            self.logger.warning(f"  Columns with missing values: {len(missing_features)}")

            # Show top 5 columns with most missing values
            top_missing = missing_features.nlargest(5)
            for col, count in top_missing.items():
                pct = (count / len(df)) * 100
                self.logger.warning(f"    {col}: {count} ({pct:.1f}%)")
        else:
            self.logger.info(f"\n✓ No missing data")

        self.logger.info("\n" + "=" * 60)

    def run_full_pipeline(
        self,
        source: Literal['csv', 'api'] = 'csv',
        save: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Run complete end-to-end data pipeline.

        Pipeline stages:
        1. Extraction
        2. Transformation
        3. Feature Engineering
        4. Save (optional)

        Args:
            source: Data source ('csv' or 'api')
            save: Whether to save final dataset
            **kwargs: Additional arguments for extraction

        Returns:
            Final DataFrame with features

        Example:
            >>> pipeline = DataPipeline()
            >>> df_final = pipeline.run_full_pipeline(
            ...     source='csv',
            ...     csv_seasons=['2223', '2324', '2425'],
            ...     save=True
            ... )
        """
        pipeline_start = datetime.now()

        self.logger.info("\n" + "#" * 60)
        self.logger.info("# STARTING COMPLETE DATA PIPELINE")
        self.logger.info("#" * 60)
        self.logger.info(f"Source: {source}")
        self.logger.info(f"Save: {save}")
        self.logger.info(f"Start time: {pipeline_start}")
        self.logger.info("#" * 60 + "\n")

        try:
            # Stage 1: Extraction
            raw_data = self.run_extraction(source=source, **kwargs)

            # Stage 2: Transformation
            df_transformed = self.run_transformation(raw_data, source=source)

            # Stage 3: Feature Engineering
            df_features = self.run_feature_engineering(df_transformed)

            # Stage 4: Save
            if save:
                self.save_final_dataset(df_features)

            # Pipeline summary
            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()

            self.logger.info("\n" + "#" * 60)
            self.logger.info("# PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("#" * 60)
            self.logger.info(f"Start time: {pipeline_start}")
            self.logger.info(f"End time:   {pipeline_end}")
            self.logger.info(f"Duration:   {duration:.1f} seconds")
            self.logger.info(f"Final dataset: {len(df_features)} matches, {len(df_features.columns)} columns")
            self.logger.info("#" * 60 + "\n")

            return df_features

        except Exception as e:
            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()

            self.logger.error("\n" + "#" * 60)
            self.logger.error("# PIPELINE FAILED")
            self.logger.error("#" * 60)
            self.logger.error(f"Error: {str(e)}")
            self.logger.error(f"Duration before failure: {duration:.1f} seconds")
            self.logger.error("#" * 60 + "\n")

            raise


def run_quick_pipeline(source: Literal['csv', 'api'] = 'csv') -> pd.DataFrame:
    """
    Quick pipeline execution for testing.

    Args:
        source: Data source ('csv' or 'api')

    Returns:
        Final DataFrame with features

    Example:
        >>> df = run_quick_pipeline(source='csv')
    """
    from src.utils import setup_logging

    # Setup logging
    setup_logging()

    # Run pipeline
    pipeline = DataPipeline()

    if source == 'csv':
        df_final = pipeline.run_full_pipeline(
            source='csv',
            csv_seasons=['2324', '2425'],  # Just 2 seasons for speed
            save=True
        )
    else:
        df_final = pipeline.run_full_pipeline(
            source='api',
            league_id=39,
            season='2024',
            save=True
        )

    return df_final


def main():
    """Run complete data pipeline."""
    from src.utils import setup_logging

    # Setup logging
    setup_logging()

    # Initialize pipeline
    pipeline = DataPipeline()

    # Run full pipeline with multiple seasons
    df_final = pipeline.run_full_pipeline(
        source='csv',
        csv_seasons=['2223', '2324', '2425'],
        save=True
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED!")
    print("=" * 60)
    print(f"Final dataset: {len(df_final)} matches")
    print(f"Features: {len(pipeline.feature_engineer.get_feature_names())}")
    print(f"Data saved to: {pipeline.final_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
