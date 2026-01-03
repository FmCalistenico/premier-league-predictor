"""
Script to extract Premier League data from configured sources.
Can be run from command line with various options.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logging, get_logger, Config
from src.data import DataExtractor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract Premier League data from various sources'
    )

    parser.add_argument(
        '--sources',
        nargs='+',
        choices=['csv', 'api', 'all'],
        default=['csv'],
        help='Data sources to extract from (default: csv)'
    )

    parser.add_argument(
        '--seasons',
        nargs='+',
        default=['2223', '2324', '2425'],
        help='Season codes for CSV extraction (default: 2223 2324 2425)'
    )

    parser.add_argument(
        '--league-id',
        type=int,
        default=None,
        help='League ID for API extraction (default: from config)'
    )

    parser.add_argument(
        '--season',
        type=str,
        default=None,
        help='Season year for API extraction (default: from config)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save extracted data (dry run)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main extraction function."""
    args = parse_arguments()

    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    if args.verbose:
        logger.setLevel('DEBUG')

    logger.info("=" * 60)
    logger.info("Premier League Data Extraction")
    logger.info("=" * 60)

    # Determine sources
    if 'all' in args.sources:
        sources = ['csv', 'api']
    else:
        sources = args.sources

    logger.info(f"Sources: {sources}")
    logger.info(f"CSV Seasons: {args.seasons}")
    logger.info(f"Save data: {not args.no_save}")

    # Initialize extractor
    try:
        config = Config()
        extractor = DataExtractor(config)

        # Run extraction
        logger.info("\nStarting extraction...")

        results = extractor.run_extraction(
            sources=sources,
            league_id=args.league_id,
            season=args.season,
            csv_seasons=args.seasons,
            save_data=not args.no_save
        )

        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTION RESULTS")
        logger.info("=" * 60)

        logger.info(f"\nExtraction ID: {results['extraction_id']}")
        logger.info(f"Sources processed: {', '.join(results['sources'])}")

        # Success summary
        if results['results']:
            logger.info(f"\n✓ Successful extractions: {len(results['results'])}")
            for source, metadata in results['results'].items():
                logger.info(f"\n  {source.upper()}:")
                if source == 'csv':
                    logger.info(f"    - Matches: {metadata.get('matches_count', 'N/A')}")
                    logger.info(f"    - Seasons: {metadata.get('seasons', 'N/A')}")
                elif source == 'api':
                    logger.info(f"    - Fixtures: {metadata.get('fixtures_count', 'N/A')}")
                    logger.info(f"    - Teams: {metadata.get('teams_count', 'N/A')}")
                logger.info(f"    - Status: {metadata.get('extraction_status', 'N/A')}")

        # Error summary
        if results['errors']:
            logger.error(f"\n✗ Failed extractions: {len(results['errors'])}")
            for source, error in results['errors'].items():
                logger.error(f"  {source.upper()}: {error}")

        # Files saved
        if results['files_saved']:
            logger.info(f"\n📁 Files saved:")
            for source, path in results['files_saved'].items():
                logger.info(f"  {source}: {path}")

        logger.info("\n" + "=" * 60)
        logger.info("Extraction completed!")
        logger.info("=" * 60)

        # Exit code
        exit_code = 0 if not results['errors'] else 1
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"\nFatal error during extraction: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
