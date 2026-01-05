"""
Build prediction dataset using dual-mode architecture.

This script generates upcoming fixtures using Factory Pattern to select
between SIMULATION and REAL_FIXTURES modes, then engineers features.

Modes:
- SIMULATION: Synthetic fixtures from historical data (experimentation)
- REAL_FIXTURES: Official fixtures from API-Football (production)

Configuration: config/prediction.yaml
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.sources.factory import MatchDataSourceFactory
from src.features.engineering_v2 import FeatureEngineerV2


def load_historical_data(data_path: str = "data/final/training_data_v2.parquet") -> pd.DataFrame:
    """
    Load historical training data.

    Args:
        data_path: Path to training data parquet file

    Returns:
        DataFrame with historical matches (base columns only)
    """
    print(f"Loading historical data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Keep only base columns needed for feature engineering
    base_cols = ['fixture_id', 'date', 'season', 'home_team', 'away_team',
                 'home_goals', 'away_goals', 'total_goals', 'gameweek']

    # Filter to available columns
    available_cols = [c for c in base_cols if c in df.columns]
    df = df[available_cols].copy()

    print(f"Loaded {len(df)} historical matches")
    return df


def extract_metadata(historical: pd.DataFrame) -> dict:
    """
    Extract metadata from historical data.

    Args:
        historical: Historical matches DataFrame

    Returns:
        Dictionary with teams, last_gameweek, last_date, season
    """
    print("Extracting metadata...")

    # Extract unique teams
    home_teams = set(historical['home_team'].unique())
    away_teams = set(historical['away_team'].unique())
    teams = sorted(list(home_teams.union(away_teams)))

    # Last gameweek
    last_gameweek = int(historical['gameweek'].max())

    # Last match date
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(historical['date']):
        historical['date'] = pd.to_datetime(historical['date'])

    last_match_date = historical['date'].max()

    # Season (assume most recent)
    season = historical['season'].iloc[-1] if 'season' in historical.columns else "2025/26"

    metadata = {
        'teams': teams,
        'num_teams': len(teams),
        'last_gameweek': last_gameweek,
        'last_match_date': last_match_date,
        'season': season
    }

    print(f"Found {metadata['num_teams']} teams")
    print(f"Last gameweek: {metadata['last_gameweek']}")
    print(f"Last match date: {metadata['last_match_date']}")
    print(f"Season: {metadata['season']}")

    return metadata


def generate_fixtures(
    teams: list,
    last_gameweek: int,
    last_match_date: datetime,
    season: str,
    num_gameweeks: int = 3
) -> tuple:
    """
    Generate fixtures using Factory Pattern (mode-agnostic).

    Mode is determined by:
    1. Environment variable: PREDICTION_MODE
    2. Config file: config/prediction.yaml (prediction.mode)

    Args:
        teams: List of team names
        last_gameweek: Last known gameweek
        last_match_date: Date of last match
        season: Season string
        num_gameweeks: Number of gameweeks to fetch/generate

    Returns:
        Tuple of (fixtures DataFrame, source_metadata)
    """
    print(f"\n{'='*60}")
    print("GENERATING FIXTURES")
    print(f"{'='*60}")

    # Create data source using Factory Pattern
    source = MatchDataSourceFactory.create_source(
        historical_teams=teams,
        last_gameweek=last_gameweek,
        last_match_date=last_match_date,
        season=season
    )

    # Get fixtures (works for both SIMULATION and REAL_FIXTURES)
    fixtures = source.get_upcoming_fixtures(
        num_gameweeks=num_gameweeks,
        future_only=False  # Allow all fixtures
    )

    # Get metadata
    source_metadata = source.get_source_metadata()

    print(f"\n[OK] Retrieved {len(fixtures)} fixtures")
    print(f"  Source Type: {source_metadata.source_type}")
    print(f"  Provider: {source_metadata.provider}")
    print(f"  Confidence: {source_metadata.confidence_score:.0%}")
    print(f"{'='*60}\n")

    return fixtures, source_metadata


def engineer_features(
    fixtures: pd.DataFrame,
    historical: pd.DataFrame
) -> pd.DataFrame:
    """
    Engineer features for fixtures.

    Args:
        fixtures: Upcoming fixtures DataFrame
        historical: Historical matches DataFrame

    Returns:
        DataFrame with engineered features
    """
    print("\nEngineering features...")

    # Initialize feature engineer
    feature_engineer = FeatureEngineerV2()

    # Mark fixtures as future for later extraction
    fixtures = fixtures.copy()
    fixtures['_is_future'] = True

    # Add placeholder result columns for future fixtures (will be NaN)
    if 'home_goals' not in fixtures.columns:
        fixtures['home_goals'] = np.nan
    if 'away_goals' not in fixtures.columns:
        fixtures['away_goals'] = np.nan
    if 'total_goals' not in fixtures.columns:
        fixtures['total_goals'] = np.nan

    historical = historical.copy()
    historical['_is_future'] = False

    # Combine historical + fixtures
    combined = pd.concat([historical, fixtures], ignore_index=True).sort_values('date').reset_index(drop=True)

    # Engineer features on combined dataset
    print(f"  Combined {len(historical)} historical + {len(fixtures)} future fixtures")
    combined_with_features = feature_engineer.engineer_features(combined)

    # Extract only future fixtures
    features = combined_with_features[combined_with_features['_is_future'] == True].copy()
    features = features.drop(columns=['_is_future'])

    print(f"Engineered {features.shape[1]} features for {len(features)} fixtures")

    return features


def save_predictions(
    features: pd.DataFrame,
    metadata: object,
    output_dir: str = "data/predictions"
) -> None:
    """
    Save predictions and metadata.

    Args:
        features: Features DataFrame
        metadata: Source metadata
        output_dir: Output directory
    """
    print(f"\nSaving predictions to {output_dir}/...")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save features
    features_path = f"{output_dir}/upcoming_fixtures_features.parquet"
    features.to_parquet(features_path, index=False)
    print(f"Saved features: {features_path}")

    # Save metadata
    metadata_path = f"{output_dir}/source_metadata.json"

    # Convert datetime to ISO format for JSON serialization
    metadata_serializable = {
        'source_type': metadata.source_type,
        'provider': metadata.provider,
        'last_updated': metadata.last_updated.isoformat(),
        'confidence_score': metadata.confidence_score,
        'total_fixtures': metadata.total_fixtures,
        'api_version': metadata.api_version,
        'api_rate_limit': metadata.api_rate_limit
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata_serializable, f, indent=2)

    print(f"Saved metadata: {metadata_path}")

    # Also save raw fixtures for inspection
    if len(features) > 0:
        fixtures_csv_path = f"{output_dir}/upcoming_fixtures_raw.csv"
        fixtures_cols = [
            'fixture_id', 'date', 'gameweek', 'season',
            'home_team', 'away_team'
        ]
        # Add optional columns if they exist
        optional_cols = ['source_type', 'is_official', 'confidence_score', 'validation_status']
        for col in optional_cols:
            if col in features.columns:
                fixtures_cols.append(col)

        features[fixtures_cols].to_csv(fixtures_csv_path, index=False)
        print(f"Saved raw fixtures: {fixtures_csv_path}")
    else:
        print("No fixtures to save to CSV (0 fixtures generated)")


def main():
    """
    Main execution flow.

    Dual-Mode Architecture:
    - Mode selection via Factory Pattern
    - Feature engineering is mode-agnostic
    - Output includes source metadata for transparency
    """
    print("=" * 60)
    print("Premier League Prediction Dataset Builder")
    print("Dual-Mode Architecture (SIMULATION | REAL_FIXTURES)")
    print("=" * 60)

    # Configuration (can be overridden by config/prediction.yaml)
    NUM_GAMEWEEKS = 3

    # 1. Load historical data
    historical = load_historical_data()

    # 2. Extract metadata
    metadata = extract_metadata(historical)

    # 3. Generate fixtures (Factory Pattern - mode determined by config)
    fixtures, source_metadata = generate_fixtures(
        teams=metadata['teams'],
        last_gameweek=metadata['last_gameweek'],
        last_match_date=metadata['last_match_date'],
        season=metadata['season'],
        num_gameweeks=NUM_GAMEWEEKS
    )

    # 4. Engineer features (mode-agnostic)
    features = engineer_features(fixtures, historical)

    # 5. Save predictions
    save_predictions(features, source_metadata)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully")
    print("=" * 60)

    # Mode-specific messaging
    if source_metadata.source_type == "SIMULATION":
        print("\n[!] SIMULATION MODE")
        print("   Fixtures are SYNTHETIC, not official")
        print("   Useful for: experimentation, backtesting, development")
    else:
        print("\n[OK] REAL_FIXTURES MODE")
        print("   Fixtures are OFFICIAL from API-Football")
        print("   Confidence: 100%")

    print("\n   View predictions in data/predictions/")
    print("\nNext steps:")
    print("  1. Run dashboard: streamlit run dashboards/bias_monitor.py")
    print("  2. Check source banner (SIMULATION warning or OFFICIAL badge)")


if __name__ == '__main__':
    main()
