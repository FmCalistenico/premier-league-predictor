"""
Script to transform raw Premier League data into standardized format.
Can be run from command line to process extracted data.
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logging, get_logger, Config
from src.data import DataTransformer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Transform raw Premier League data into standardized format'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file path (CSV or JSON)'
    )

    parser.add_argument(
        '--source',
        type=str,
        choices=['csv', 'api'],
        required=True,
        help='Data source type'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: data/processed/transformed_TIMESTAMP.csv)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save transformed data (dry run)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def load_raw_data(input_path: Path, source: str):
    """Load raw data from file."""
    logger = get_logger(__name__)

    logger.info(f"Loading raw data from {input_path}")

    if source == 'csv':
        df = pd.read_csv(input_path)
        logger.info(f"Loaded CSV with {len(df)} rows")
        return df

    elif source == 'api':
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded API JSON data")
        return data

    else:
        raise ValueError(f"Unsupported source: {source}")


def main():
    """Main transformation function."""
    args = parse_arguments()

    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    if args.verbose:
        logger.setLevel('DEBUG')

    logger.info("=" * 60)
    logger.info("Premier League Data Transformation")
    logger.info("=" * 60)

    logger.info(f"Input file: {args.input}")
    logger.info(f"Source type: {args.source}")
    logger.info(f"Save output: {not args.no_save}")

    try:
        # Load raw data
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)

        raw_data = load_raw_data(input_path, args.source)

        # Transform data
        logger.info("\nStarting transformation...")

        transformer = DataTransformer()
        df_transformed, metadata = transformer.transform(raw_data, source=args.source)

        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("TRANSFORMATION RESULTS")
        logger.info("=" * 60)

        logger.info(f"\nInput: {metadata['initial_rows']} rows")
        logger.info(f"Output: {metadata['final_rows']} rows")
        logger.info(f"Removed: {metadata['initial_rows'] - metadata['final_rows']} rows")

        logger.info(f"\nPipeline steps:")
        for step in metadata['pipeline_steps']:
            logger.info(f"  ✓ {step}")

        if metadata['issues']:
            logger.warning(f"\nIssues encountered: {len(metadata['issues'])}")
            for issue in metadata['issues']:
                logger.warning(f"  - {issue}")
        else:
            logger.info("\n✓ No issues found")

        # Summary
        summary = metadata['summary']
        logger.info("\n" + "=" * 60)
        logger.info("DATA SUMMARY")
        logger.info("=" * 60)

        logger.info(f"\nDate range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        logger.info(f"Total matches: {summary['total_matches']}")
        logger.info(f"Unique teams: {summary['unique_teams']}")
        logger.info(f"Avg goals/match: {summary['avg_goals_per_match']:.2f}")
        logger.info(f"Over 2.5 goals: {summary['over_2.5_pct']:.1f}%")

        # Save transformed data
        if not args.no_save:
            config = Config()
            processed_dir = config.data_processed_path
            processed_dir.mkdir(parents=True, exist_ok=True)

            # Determine output path
            if args.output:
                output_file = Path(args.output)
            else:
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = processed_dir / f'transformed_{timestamp}.csv'

            # Save CSV
            df_transformed.to_csv(output_file, index=False)
            logger.info(f"\n✓ Saved transformed data: {output_file}")
            logger.info(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")

            # Save metadata
            metadata_file = output_file.parent / f"{output_file.stem}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved metadata: {metadata_file}")

            logger.info(f"\nOutput files:")
            logger.info(f"  - {output_file}")
            logger.info(f"  - {metadata_file}")

        logger.info("\n" + "=" * 60)
        logger.info("Transformation completed successfully!")
        logger.info("=" * 60)

        sys.exit(0)

    except Exception as e:
        logger.error(f"\nFatal error during transformation: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
