"""
Example usage of data extraction modules.
Demonstrates how to use API clients and data extractor.
"""

from src.utils import setup_logging, get_logger
from src.data import APIFootballClient, FootballDataCSVClient, DataExtractor


def example_csv_client():
    """Example of using FootballDataCSVClient."""
    print("=" * 60)
    print("CSV Client Example")
    print("=" * 60)

    client = FootballDataCSVClient()

    # Get single season data
    print("\n1. Fetching single season (2024-2025)...")
    df_single = client.get_season_data(season='2425')
    print(f"   Retrieved {len(df_single)} matches")
    print(f"   Columns: {df_single.columns.tolist()[:5]}...")  # Show first 5 columns

    # Get multiple seasons
    print("\n2. Fetching multiple seasons...")
    df_multiple = client.get_multiple_seasons(seasons=['2223', '2324', '2425'])
    print(f"   Retrieved {len(df_multiple)} matches across 3 seasons")
    print(f"   Season distribution:")
    print(df_multiple['season'].value_counts().to_dict())

    print()


def example_api_client():
    """Example of using APIFootballClient."""
    print("=" * 60)
    print("API Client Example")
    print("=" * 60)

    # Note: Requires valid API key in .env file
    client = APIFootballClient()

    try:
        # Get fixtures
        print("\n1. Fetching fixtures for Premier League 2024...")
        fixtures = client.get_fixtures(
            league_id=39,  # Premier League
            season='2024'
        )
        print(f"   Retrieved {len(fixtures)} fixtures")

        if fixtures:
            sample = fixtures[0]
            print(f"   Sample fixture: {sample['teams']['home']['name']} vs "
                  f"{sample['teams']['away']['name']}")

        # Get standings
        print("\n2. Fetching standings...")
        standings = client.get_standings(league_id=39, season='2024')
        print(f"   Retrieved standings data")

        # Get team statistics
        if fixtures:
            team_id = fixtures[0]['teams']['home']['id']
            print(f"\n3. Fetching statistics for team {team_id}...")
            stats = client.get_team_statistics(
                team_id=team_id,
                league_id=39,
                season='2024'
            )
            print(f"   Retrieved team statistics")

    except Exception as e:
        print(f"   Error: {str(e)}")
        print("   Note: Make sure you have a valid API key in your .env file")

    print()


def example_data_extractor_csv():
    """Example of using DataExtractor for CSV extraction."""
    print("=" * 60)
    print("Data Extractor - CSV Example")
    print("=" * 60)

    extractor = DataExtractor()

    # Extract from CSV only
    print("\nExtracting data from CSV sources...")
    results = extractor.run_extraction(
        sources=['csv'],
        csv_seasons=['2223', '2324', '2425'],
        save_data=True
    )

    print("\nExtraction Results:")
    print(f"  Extraction ID: {results['extraction_id']}")
    print(f"  Sources processed: {results['sources']}")

    if 'csv' in results['results']:
        csv_info = results['results']['csv']
        print(f"\n  CSV Data:")
        print(f"    - Matches: {csv_info['matches_count']}")
        print(f"    - Seasons: {csv_info['seasons']}")
        print(f"    - Status: {csv_info['extraction_status']}")

    if results['files_saved']:
        print(f"\n  Files saved:")
        for source, path in results['files_saved'].items():
            print(f"    - {source}: {path}")

    if results['errors']:
        print(f"\n  Errors:")
        for source, error in results['errors'].items():
            print(f"    - {source}: {error}")

    print()


def example_data_extractor_api():
    """Example of using DataExtractor for API extraction."""
    print("=" * 60)
    print("Data Extractor - API Example")
    print("=" * 60)

    extractor = DataExtractor()

    try:
        # Extract from API
        print("\nExtracting data from API...")
        results = extractor.run_extraction(
            sources=['api'],
            league_id=39,
            season='2024',
            save_data=True
        )

        print("\nExtraction Results:")
        print(f"  Extraction ID: {results['extraction_id']}")

        if 'api' in results['results']:
            api_info = results['results']['api']
            print(f"\n  API Data:")
            print(f"    - Fixtures: {api_info['fixtures_count']}")
            print(f"    - Teams: {api_info.get('teams_count', 0)}")
            print(f"    - Status: {api_info['extraction_status']}")

        if results['files_saved']:
            print(f"\n  Files saved:")
            for source, path in results['files_saved'].items():
                print(f"    - {source}: {path}")

    except Exception as e:
        print(f"  Error: {str(e)}")
        print("  Note: Make sure you have a valid API key in your .env file")

    print()


def example_data_extractor_combined():
    """Example of using DataExtractor for both sources."""
    print("=" * 60)
    print("Data Extractor - Combined Example")
    print("=" * 60)

    extractor = DataExtractor()

    try:
        # Extract from both sources
        print("\nExtracting data from all sources...")
        results = extractor.run_extraction(
            sources=['csv', 'api'],
            league_id=39,
            season='2024',
            csv_seasons=['2324', '2425'],
            save_data=True
        )

        print("\nExtraction Summary:")
        print(f"  Extraction ID: {results['extraction_id']}")
        print(f"  Successful extractions: {len(results['results'])}")
        print(f"  Failed extractions: {len(results['errors'])}")

        for source in ['csv', 'api']:
            if source in results['results']:
                print(f"\n  ✓ {source.upper()} extraction successful")
            elif source in results['errors']:
                print(f"\n  ✗ {source.upper()} extraction failed: {results['errors'][source]}")

        if results['files_saved']:
            print(f"\n  Files saved: {len(results['files_saved'])}")

    except Exception as e:
        print(f"  Error: {str(e)}")

    print()


def main():
    """Run all examples."""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    logger.info("Starting data extraction examples")

    print("\n" + "=" * 60)
    print("PREMIER LEAGUE DATA EXTRACTION EXAMPLES")
    print("=" * 60 + "\n")

    # Run CSV examples (no API key needed)
    try:
        example_csv_client()
        example_data_extractor_csv()
    except Exception as e:
        print(f"CSV examples failed: {str(e)}\n")

    # Run API examples (requires API key)
    print("\nAPI Examples (requires API key in .env):")
    print("-" * 60)
    try:
        example_api_client()
        example_data_extractor_api()
    except Exception as e:
        print(f"API examples failed: {str(e)}\n")

    # Run combined example
    try:
        example_data_extractor_combined()
    except Exception as e:
        print(f"Combined example failed: {str(e)}\n")

    print("=" * 60)
    print("Examples completed!")
    print("Check data/raw/ directory for extracted files")
    print("Check logs/ directory for detailed logs")
    print("=" * 60)


if __name__ == "__main__":
    main()
