"""
Complete ETL (Extract, Transform, Load) pipeline for Premier League data.
Orchestrates extraction, transformation, and storage of football data.
"""

from pathlib import Path
from typing import List, Optional
from datetime import datetime
import json

from src.utils import LoggerMixin, Config
from src.data import DataExtractor, DataTransformer


class ETLPipeline(LoggerMixin):
    """
    Complete ETL pipeline for Premier League data.

    Pipeline stages:
    1. Extract: Fetch data from API and/or CSV sources
    2. Transform: Parse, clean, and standardize data
    3. Load: Save processed data to final location
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize ETL pipeline.

        Args:
            config: Configuration instance
        """
        self.config = config or Config()
        self.extractor = DataExtractor(self.config)
        self.transformer = DataTransformer()

        self.logger.info("ETL Pipeline initialized")

    def run(
        self,
        sources: List[str] = ['csv'],
        csv_seasons: Optional[List[str]] = None,
        league_id: Optional[int] = None,
        season: Optional[str] = None,
        save_raw: bool = True,
        save_transformed: bool = True
    ) -> dict:
        """
        Run complete ETL pipeline.

        Args:
            sources: Data sources to extract from
            csv_seasons: Seasons for CSV extraction
            league_id: League ID for API extraction
            season: Season for API extraction
            save_raw: Whether to save raw extracted data
            save_transformed: Whether to save transformed data

        Returns:
            Dictionary with pipeline results

        Example:
            >>> pipeline = ETLPipeline()
            >>> results = pipeline.run(
            ...     sources=['csv'],
            ...     csv_seasons=['2223', '2324', '2425']
            ... )
        """
        pipeline_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.logger.info("=" * 60)
        self.logger.info(f"Starting ETL Pipeline (ID: {pipeline_id})")
        self.logger.info("=" * 60)

        results = {
            'pipeline_id': pipeline_id,
            'stages': {
                'extract': {},
                'transform': {},
                'load': {}
            },
            'errors': [],
            'files_created': []
        }

        try:
            # Stage 1: Extract
            self.logger.info("\n" + "=" * 60)
            self.logger.info("STAGE 1: EXTRACT")
            self.logger.info("=" * 60)

            extraction_results = self.extractor.run_extraction(
                sources=sources,
                league_id=league_id,
                season=season,
                csv_seasons=csv_seasons,
                save_data=save_raw
            )

            results['stages']['extract'] = extraction_results

            if extraction_results['errors']:
                self.logger.error(f"Extraction had {len(extraction_results['errors'])} errors")

            if save_raw and extraction_results['files_saved']:
                results['files_created'].extend(extraction_results['files_saved'].values())

            # Stage 2: Transform
            self.logger.info("\n" + "=" * 60)
            self.logger.info("STAGE 2: TRANSFORM")
            self.logger.info("=" * 60)

            transformed_data = {}

            # Transform CSV data
            if 'csv' in sources and 'csv' in extraction_results['results']:
                self.logger.info("Transforming CSV data...")

                # Load the saved CSV file
                if 'csv' in extraction_results['files_saved']:
                    import pandas as pd
                    csv_file = extraction_results['files_saved']['csv']
                    raw_csv = pd.read_csv(csv_file)

                    df_transformed, metadata = self.transformer.transform(
                        raw_csv,
                        source='csv'
                    )

                    transformed_data['csv'] = {
                        'data': df_transformed,
                        'metadata': metadata
                    }

                    results['stages']['transform']['csv'] = metadata

            # Transform API data
            if 'api' in sources and 'api' in extraction_results['results']:
                self.logger.info("Transforming API data...")

                if 'api' in extraction_results['files_saved']:
                    api_file = extraction_results['files_saved']['api']
                    with open(api_file, 'r') as f:
                        raw_api = json.load(f)

                    df_transformed, metadata = self.transformer.transform(
                        raw_api,
                        source='api'
                    )

                    transformed_data['api'] = {
                        'data': df_transformed,
                        'metadata': metadata
                    }

                    results['stages']['transform']['api'] = metadata

            # Stage 3: Load
            self.logger.info("\n" + "=" * 60)
            self.logger.info("STAGE 3: LOAD")
            self.logger.info("=" * 60)

            if save_transformed and transformed_data:
                final_dir = self.config.data_final_path
                final_dir.mkdir(parents=True, exist_ok=True)

                for source, data_dict in transformed_data.items():
                    df = data_dict['data']
                    metadata = data_dict['metadata']

                    # Save final data
                    output_file = final_dir / f'{source}_final_{pipeline_id}.csv'
                    df.to_csv(output_file, index=False)

                    # Save metadata
                    metadata_file = final_dir / f'{source}_final_{pipeline_id}_metadata.json'
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)

                    self.logger.info(f"Saved {source} data: {output_file}")
                    results['files_created'].extend([str(output_file), str(metadata_file)])

                results['stages']['load']['status'] = 'success'
                results['stages']['load']['files_count'] = len(transformed_data) * 2

            # Pipeline summary
            self.logger.info("\n" + "=" * 60)
            self.logger.info("ETL PIPELINE SUMMARY")
            self.logger.info("=" * 60)

            self.logger.info(f"\nPipeline ID: {pipeline_id}")
            self.logger.info(f"Sources processed: {sources}")
            self.logger.info(f"Files created: {len(results['files_created'])}")

            if results['errors']:
                self.logger.warning(f"Errors encountered: {len(results['errors'])}")
            else:
                self.logger.info("✓ Pipeline completed successfully with no errors")

            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            results['errors'].append(str(e))
            raise


def main():
    """Run ETL pipeline from command line."""
    from src.utils import setup_logging

    setup_logging()

    pipeline = ETLPipeline()

    # Run pipeline with default settings
    results = pipeline.run(
        sources=['csv'],
        csv_seasons=['2223', '2324', '2425'],
        save_raw=True,
        save_transformed=True
    )

    print("\n" + "=" * 60)
    print("ETL Pipeline Complete!")
    print("=" * 60)
    print(f"\nPipeline ID: {results['pipeline_id']}")
    print(f"Files created: {len(results['files_created'])}")
    print("\nFiles:")
    for file_path in results['files_created']:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()
