"""
CLI tool for running Premier League predictions with mode selection.

This script provides a user-friendly command-line interface for:
- Running predictions in SIMULATION or REAL_FIXTURES mode
- Validating configuration
- Checking current mode
- Testing API connection

Usage:
    # Run with default mode (from config)
    python scripts/run_predictions.py

    # Run with specific mode
    python scripts/run_predictions.py --mode SIMULATION
    python scripts/run_predictions.py --mode REAL_FIXTURES

    # Validate configuration
    python scripts/run_predictions.py --validate

    # Check current mode
    python scripts/run_predictions.py --check-mode

    # Test API connection (REAL_FIXTURES mode only)
    python scripts/run_predictions.py --test-api
"""

import sys
import argparse
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.sources.factory import MatchDataSourceFactory


def validate_config():
    """Validate configuration and report issues."""
    print("=" * 60)
    print("CONFIGURATION VALIDATION")
    print("=" * 60)

    results = MatchDataSourceFactory.validate_config()

    print(f"\nMode: {results['mode']}")
    print(f"Valid: {results['valid']}")

    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  [X] {error}")

    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"  [!] {warning}")

    if results['valid']:
        print("\n[OK] Configuration is valid")
    else:
        print("\n[X] Configuration has errors - fix before running")
        sys.exit(1)


def check_mode():
    """Check current prediction mode."""
    print("=" * 60)
    print("CURRENT MODE CHECK")
    print("=" * 60)

    try:
        mode = MatchDataSourceFactory.get_mode()
        print(f"\nCurrent mode: {mode}")

        if mode == "SIMULATION":
            print("\n[!] SIMULATION MODE")
            print("  - Fixtures are SYNTHETIC")
            print("  - Generated from historical data")
            print("  - NO API calls")
            print("  - Use for: experimentation, backtesting")
        else:
            print("\n[OK] REAL_FIXTURES MODE")
            print("  - Fixtures are OFFICIAL")
            print("  - Fetched from API-Football")
            print("  - Requires: API key in .env")
            print("  - Use for: production predictions")

        print("\nTo change mode:")
        print("  1. Set PREDICTION_MODE in .env file")
        print("  2. Or edit config/prediction.yaml")
        print("  3. Run: python scripts/run_predictions.py --mode <MODE>")

    except Exception as e:
        print(f"\n[X] Error checking mode: {e}")
        sys.exit(1)


def test_api_connection():
    """Test API-Football connection."""
    print("=" * 60)
    print("API CONNECTION TEST")
    print("=" * 60)

    # Load environment
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("API_FOOTBALL_KEY")

    if not api_key or api_key == "your_rapidapi_key_here":
        print("\n[X] API key not configured")
        print("\nSetup instructions:")
        print("  1. Copy .env.example to .env")
        print("  2. Get key from: https://rapidapi.com/api-sports/api/api-football")
        print("  3. Set API_FOOTBALL_KEY in .env")
        sys.exit(1)

    print(f"\nAPI Key: {api_key[:10]}..." + "*" * 10)

    # Try to create API source
    try:
        from src.data.sources.fixture_api import FixtureAPISource

        # Create temporary source with dummy teams
        print("\nCreating test API source...")
        source = FixtureAPISource(
            historical_teams=["Arsenal", "Liverpool", "Chelsea"],
            api_key=api_key,
            league_id=39,
            season=2024,  # Current season 2024-2025
            cache_enabled=False  # Disable cache for test
        )

        print("[OK] API source created successfully")

        # Try to fetch 1 fixture
        print("\nFetching 1 upcoming fixture...")
        fixtures = source.get_upcoming_fixtures(num_gameweeks=1, future_only=True)

        if len(fixtures) > 0:
            print(f"\n[OK] API connection successful!")
            print(f"  Retrieved {len(fixtures)} fixtures")
            print(f"\nSample fixture:")
            first = fixtures.iloc[0]
            print(f"  {first['home_team']} vs {first['away_team']}")
            print(f"  Date: {first['date']}")
            print(f"  Gameweek: {first['gameweek']}")
        else:
            print("\n[!] No upcoming fixtures found")
            print("  This may be normal during off-season")

    except Exception as e:
        print(f"\n[X] API connection failed: {e}")
        sys.exit(1)


def run_predictions(mode=None):
    """Run predictions pipeline."""
    print("=" * 60)
    print("RUNNING PREDICTIONS PIPELINE")
    print("=" * 60)

    # Set mode override if provided
    if mode:
        print(f"\nMode override: {mode}")
        os.environ["PREDICTION_MODE"] = mode

    # Import and run main pipeline
    from build_prediction_dataset import main
    main()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Premier League Predictions CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default mode
  python scripts/run_predictions.py

  # Run with specific mode
  python scripts/run_predictions.py --mode SIMULATION
  python scripts/run_predictions.py --mode REAL_FIXTURES

  # Validate configuration
  python scripts/run_predictions.py --validate

  # Check current mode
  python scripts/run_predictions.py --check-mode

  # Test API connection
  python scripts/run_predictions.py --test-api
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["SIMULATION", "REAL_FIXTURES"],
        help="Prediction mode (overrides config)"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration and exit"
    )

    parser.add_argument(
        "--check-mode",
        action="store_true",
        help="Check current mode and exit"
    )

    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Test API connection and exit"
    )

    args = parser.parse_args()

    # Execute appropriate action
    if args.validate:
        validate_config()
    elif args.check_mode:
        check_mode()
    elif args.test_api:
        test_api_connection()
    else:
        # Run predictions (with optional mode override)
        run_predictions(mode=args.mode)


if __name__ == "__main__":
    main()
