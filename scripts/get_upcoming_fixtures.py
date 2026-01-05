"""
Get Upcoming Fixtures from API-Football or CSV
===============================================

Downloads upcoming fixtures for prediction.

Usage:
    python scripts/get_upcoming_fixtures.py --gameweek 23
    python scripts/get_upcoming_fixtures.py --from-api --gameweek 23
    python scripts/get_upcoming_fixtures.py --manual
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, setup_logging

logger = get_logger(__name__)


class FixtureFetcher:
    """Fetches upcoming fixtures from various sources."""

    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: API-Football API key (from .env or environment)
        """
        self.api_key = api_key or os.getenv('FOOTBALL_DATA_API_KEY')
        self.api_base_url = "https://v3.football.api-sports.io"
        self.league_id = 39  # Premier League

    def get_from_api(self, gameweek: int = None, season: int = None) -> pd.DataFrame:
        """
        Get upcoming fixtures from API-Football.

        Args:
            gameweek: Specific gameweek (optional)
            season: Season year (e.g., 2025 for 2025/2026)

        Returns:
            DataFrame with upcoming fixtures
        """
        if not self.api_key:
            logger.error("ERROR: API key not found")
            logger.error("Set FOOTBALL_DATA_API_KEY in .env or environment")
            raise ValueError("API key required")

        # Default to current season
        if season is None:
            now = datetime.now()
            season = now.year if now.month >= 8 else now.year - 1

        logger.info(f"Fetching fixtures from API-Football")
        logger.info(f"  League: Premier League (ID: {self.league_id})")
        logger.info(f"  Season: {season}")

        # Build API request
        headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }

        params = {
            'league': self.league_id,
            'season': season,
            'status': 'NS'  # Not Started
        }

        if gameweek:
            params['round'] = f"Regular Season - {gameweek}"
            logger.info(f"  Gameweek: {gameweek}")

        url = f"{self.api_base_url}/fixtures"

        logger.info(f"API Request: {url}")
        response = requests.get(url, headers=headers, params=params, timeout=30)

        if response.status_code != 200:
            logger.error(f"API Error: {response.status_code}")
            logger.error(response.text)
            raise Exception(f"API request failed: {response.status_code}")

        data = response.json()

        if data.get('errors'):
            logger.error(f"API returned errors: {data['errors']}")
            raise Exception(f"API errors: {data['errors']}")

        fixtures_raw = data.get('response', [])
        logger.info(f"✓ Received {len(fixtures_raw)} fixtures from API")

        # Parse into DataFrame
        fixtures = []

        for fixture in fixtures_raw:
            fixtures.append({
                'fixture_id': f"API_{fixture['fixture']['id']}",
                'date': fixture['fixture']['date'][:10],  # YYYY-MM-DD
                'time': fixture['fixture']['date'][11:16],  # HH:MM
                'home_team': fixture['teams']['home']['name'],
                'away_team': fixture['teams']['away']['name'],
                'gameweek': self._extract_gameweek(fixture.get('league', {}).get('round', '')),
                'venue': fixture['fixture']['venue']['name'],
                'referee': fixture['fixture']['referee']
            })

        df = pd.DataFrame(fixtures)

        logger.info(f"✓ Parsed {len(df)} fixtures")

        return df

    def get_manual_input(self) -> pd.DataFrame:
        """
        Get fixtures via manual console input.

        Returns:
            DataFrame with manually entered fixtures
        """
        logger.info("=" * 60)
        logger.info("MANUAL FIXTURE INPUT")
        logger.info("=" * 60)
        logger.info("Enter upcoming fixtures (one per line)")
        logger.info("Format: date,home_team,away_team,gameweek")
        logger.info("Example: 2026-01-11,Arsenal,Man City,23")
        logger.info("Press Enter twice when done")
        logger.info("=" * 60)

        fixtures = []
        while True:
            line = input("> ").strip()

            if not line:
                break

            try:
                parts = [p.strip() for p in line.split(',')]

                if len(parts) < 3:
                    logger.warning(f"Invalid format (need at least: date,home,away): {line}")
                    continue

                fixture = {
                    'date': parts[0],
                    'home_team': parts[1],
                    'away_team': parts[2],
                    'gameweek': int(parts[3]) if len(parts) > 3 else None
                }

                fixtures.append(fixture)
                logger.info(f"  ✓ Added: {fixture['home_team']} vs {fixture['away_team']}")

            except Exception as e:
                logger.warning(f"Error parsing line: {e}")
                continue

        if not fixtures:
            logger.warning("No fixtures entered")
            return pd.DataFrame()

        df = pd.DataFrame(fixtures)
        logger.info(f"\n✓ Total fixtures entered: {len(df)}")

        return df

    def get_from_csv_template(self, csv_path: str) -> pd.DataFrame:
        """
        Load fixtures from a pre-filled CSV template.

        Args:
            csv_path: Path to CSV file

        Returns:
            DataFrame with fixtures
        """
        logger.info(f"Loading fixtures from CSV: {csv_path}")

        df = pd.read_csv(csv_path)

        required_cols = ['date', 'home_team', 'away_team']
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        logger.info(f"✓ Loaded {len(df)} fixtures from CSV")

        return df

    def create_csv_template(self, output_path: str, num_fixtures: int = 10):
        """
        Create a blank CSV template for manual filling.

        Args:
            output_path: Where to save template
            num_fixtures: Number of blank rows
        """
        template = pd.DataFrame({
            'date': ['YYYY-MM-DD'] * num_fixtures,
            'home_team': [''] * num_fixtures,
            'away_team': [''] * num_fixtures,
            'gameweek': [0] * num_fixtures
        })

        template.to_csv(output_path, index=False)

        logger.info(f"✓ Created CSV template: {output_path}")
        logger.info(f"  Fill in the template and run:")
        logger.info(f"  python scripts/get_upcoming_fixtures.py --from-csv {output_path}")

    def _extract_gameweek(self, round_str: str) -> int:
        """Extract gameweek number from API round string."""
        # e.g., "Regular Season - 23" → 23
        try:
            return int(round_str.split('-')[-1].strip())
        except:
            return None

    def save_fixtures(self, df: pd.DataFrame, output_path: str):
        """Save fixtures to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        logger.info(f"✓ Saved fixtures: {output_path}")
        logger.info(f"  Rows: {len(df)}")

        return output_path


def main():
    parser = argparse.ArgumentParser(description="Get upcoming fixtures for prediction")
    parser.add_argument('--from-api', action='store_true', help="Fetch from API-Football")
    parser.add_argument('--from-csv', type=str, help="Load from CSV file")
    parser.add_argument('--manual', action='store_true', help="Manual console input")
    parser.add_argument('--create-template', type=str, help="Create blank CSV template")
    parser.add_argument('--gameweek', type=int, help="Specific gameweek to fetch")
    parser.add_argument('--season', type=int, help="Season year (e.g., 2025)")
    parser.add_argument('--output', type=str, help="Output path (default: auto-generated)")

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger.info("=" * 80)
    logger.info("GET UPCOMING FIXTURES")
    logger.info("=" * 80)

    # Create fetcher
    fetcher = FixtureFetcher()

    # Handle create template mode
    if args.create_template:
        fetcher.create_csv_template(args.create_template)
        return

    # Get fixtures based on source
    if args.from_api:
        fixtures = fetcher.get_from_api(gameweek=args.gameweek, season=args.season)
    elif args.from_csv:
        fixtures = fetcher.get_from_csv_template(args.from_csv)
    elif args.manual:
        fixtures = fetcher.get_manual_input()
    else:
        logger.error("ERROR: Must specify source: --from-api, --from-csv, or --manual")
        logger.info("Usage examples:")
        logger.info("  python scripts/get_upcoming_fixtures.py --from-api --gameweek 23")
        logger.info("  python scripts/get_upcoming_fixtures.py --manual")
        logger.info("  python scripts/get_upcoming_fixtures.py --create-template data/raw/template.csv")
        sys.exit(1)

    if fixtures.empty:
        logger.warning("No fixtures obtained")
        sys.exit(0)

    # Generate output path
    if args.output:
        output_path = args.output
    else:
        gw_suffix = f"_GW{args.gameweek}" if args.gameweek else ""
        timestamp = datetime.now().strftime("%Y%m%d")
        output_path = f"data/raw/upcoming_fixtures{gw_suffix}_{timestamp}.csv"

    # Save
    fetcher.save_fixtures(fixtures, output_path)

    # Display summary
    logger.info("=" * 80)
    logger.info("FIXTURES SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\nUpcoming fixtures:")
    for _, row in fixtures.head(10).iterrows():
        logger.info(f"  {row['date']} | {row['home_team']:20s} vs {row['away_team']:20s}")

    if len(fixtures) > 10:
        logger.info(f"  ... and {len(fixtures) - 10} more")

    logger.info(f"\n✓ Next step:")
    logger.info(f"  python scripts/build_prediction_dataset.py --input {output_path}")

    return fixtures


if __name__ == "__main__":
    main()
