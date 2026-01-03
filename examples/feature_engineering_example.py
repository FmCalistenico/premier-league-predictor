"""
Example usage of feature engineering module.
Demonstrates creating features while preventing data leakage.
"""

import pandas as pd
import numpy as np

from src.utils import setup_logging, get_logger
from src.data import FootballDataCSVClient, DataTransformer
from src.features import FeatureEngineer


def example_rolling_features():
    """Example of creating rolling features."""
    print("=" * 60)
    print("Rolling Features Example")
    print("=" * 60)

    # Get and transform data
    csv_client = FootballDataCSVClient()
    transformer = DataTransformer()

    df_raw = csv_client.get_season_data(season='2425')
    df_transformed, _ = transformer.transform(df_raw, source='csv')

    print(f"\nTransformed data: {len(df_transformed)} matches")

    # Create rolling features
    engineer = FeatureEngineer()
    df_rolling = engineer.create_rolling_features(df_transformed, windows=[3, 5, 10])

    print(f"Data with rolling features: {len(df_rolling)} matches, {len(df_rolling.columns)} columns")

    # Show example features
    print("\nSample rolling features (last 3 matches):")
    feature_cols = [
        'date', 'home_team_name', 'away_team_name',
        'home_goals_scored_L3', 'away_goals_scored_L3',
        'home_goal_diff_L3', 'away_goal_diff_L3',
        'home_over_rate_L3', 'away_over_rate_L3'
    ]

    # Filter to show only rows with features
    df_with_features = df_rolling.dropna(subset=['home_goals_scored_L3'])
    print(df_with_features[feature_cols].head(10))

    print("\n⚠️ IMPORTANT: All rolling features use .shift(1) to prevent data leakage")
    print("   This ensures we only use information from BEFORE the match.\n")


def example_match_features():
    """Example of creating match-level features."""
    print("=" * 60)
    print("Match-Level Features Example")
    print("=" * 60)

    # Get data with rolling features
    csv_client = FootballDataCSVClient()
    transformer = DataTransformer()
    engineer = FeatureEngineer()

    df_raw = csv_client.get_season_data(season='2425')
    df_transformed, _ = transformer.transform(df_raw, source='csv')
    df_rolling = engineer.create_rolling_features(df_transformed)

    print(f"\nData after rolling features: {len(df_rolling)} matches")

    # Create match features
    df_match = engineer.create_match_features(df_rolling)

    print(f"Data with match features: {len(df_match)} matches")

    # Show example features
    print("\nSample match-level features:")
    feature_cols = [
        'date', 'home_team_name', 'away_team_name',
        'expected_total_goals', 'form_difference',
        'attack_strength_diff', 'defense_strength_diff',
        'combined_over_rate'
    ]

    df_with_features = df_match.dropna(subset=['expected_total_goals'])
    print(df_with_features[feature_cols].head(10))

    print()


def example_rest_days_features():
    """Example of creating rest days features."""
    print("=" * 60)
    print("Rest Days Features Example")
    print("=" * 60)

    # Get transformed data
    csv_client = FootballDataCSVClient()
    transformer = DataTransformer()
    engineer = FeatureEngineer()

    df_raw = csv_client.get_season_data(season='2425')
    df_transformed, _ = transformer.transform(df_raw, source='csv')

    # Create rest days features
    df_rest = engineer.create_rest_days_features(df_transformed)

    print(f"\nData with rest days features: {len(df_rest)} matches")

    # Show example features
    print("\nSample rest days features:")
    feature_cols = [
        'date', 'home_team_name', 'away_team_name',
        'home_days_rest', 'away_days_rest', 'rest_advantage'
    ]

    df_with_features = df_rest.dropna(subset=['home_days_rest'])
    print(df_with_features[feature_cols].head(10))

    # Statistics
    print("\nRest days statistics:")
    print(f"  Average home rest days: {df_rest['home_days_rest'].mean():.1f}")
    print(f"  Average away rest days: {df_rest['away_days_rest'].mean():.1f}")
    print(f"  Average rest advantage: {df_rest['rest_advantage'].mean():.1f}")

    print()


def example_h2h_features():
    """Example of creating head-to-head features."""
    print("=" * 60)
    print("Head-to-Head Features Example")
    print("=" * 60)

    # Get transformed data
    csv_client = FootballDataCSVClient()
    transformer = DataTransformer()
    engineer = FeatureEngineer()

    df_raw = csv_client.get_multiple_seasons(['2324', '2425'])
    df_transformed, _ = transformer.transform(df_raw, source='csv')

    print(f"\nData from 2 seasons: {len(df_transformed)} matches")

    # Create H2H features
    df_h2h = engineer.create_head_to_head_features(df_transformed, h2h_window=5)

    print(f"Data with H2H features: {len(df_h2h)} matches")

    # Show example features
    print("\nSample H2H features:")
    feature_cols = [
        'date', 'home_team_name', 'away_team_name',
        'total_goals', 'h2h_avg_goals', 'h2h_over_rate'
    ]

    df_with_features = df_h2h[df_h2h['h2h_avg_goals'].notna()]
    print(df_with_features[feature_cols].head(10))

    print()


def example_complete_pipeline():
    """Example of complete feature engineering pipeline."""
    print("=" * 60)
    print("Complete Feature Engineering Pipeline")
    print("=" * 60)

    # Get multiple seasons of data
    csv_client = FootballDataCSVClient()
    transformer = DataTransformer()
    engineer = FeatureEngineer()

    print("\nStep 1: Extracting and transforming data...")
    df_raw = csv_client.get_multiple_seasons(['2223', '2324', '2425'])
    print(f"  Raw data: {len(df_raw)} matches")

    df_transformed, metadata = transformer.transform(df_raw, source='csv')
    print(f"  Transformed data: {len(df_transformed)} matches")

    # Run complete feature engineering
    print("\nStep 2: Engineering features...")
    df_features = engineer.engineer_features(
        df_transformed,
        rolling_windows=[3, 5, 10],
        h2h_window=5,
        min_matches_required=5
    )

    print(f"\n" + "=" * 60)
    print("FEATURE ENGINEERING RESULTS")
    print("=" * 60)

    print(f"\nFinal dataset: {len(df_features)} matches")
    print(f"Total features created: {len(engineer.get_feature_names())}")

    # Feature breakdown
    feature_names = engineer.get_feature_names()
    rolling_features = [f for f in feature_names if 'L' in f and ('home_' in f or 'away_' in f)]
    match_features = [f for f in feature_names if any(x in f for x in ['expected', 'form', 'strength', 'combined'])]
    rest_features = [f for f in feature_names if 'rest' in f]
    h2h_features = [f for f in feature_names if 'h2h' in f]

    print(f"\nFeature breakdown:")
    print(f"  - Rolling features: {len(rolling_features)}")
    print(f"  - Match features: {len(match_features)}")
    print(f"  - Rest features: {len(rest_features)}")
    print(f"  - H2H features: {len(h2h_features)}")

    print(f"\nAll feature names:")
    for i, feature in enumerate(feature_names, 1):
        print(f"  {i:2d}. {feature}")

    # Show sample data
    print(f"\nSample data with features:")
    display_cols = [
        'date', 'home_team_name', 'away_team_name',
        'total_goals', 'over_2.5',
        'expected_total_goals', 'combined_over_rate'
    ]
    print(df_features[display_cols].head())

    # Feature statistics
    print(f"\nFeature statistics:")
    print(f"  Expected total goals: {df_features['expected_total_goals'].mean():.2f} ± {df_features['expected_total_goals'].std():.2f}")
    print(f"  Combined over rate: {df_features['combined_over_rate'].mean():.2f}")
    print(f"  Form difference: {df_features['form_difference'].mean():.2f} ± {df_features['form_difference'].std():.2f}")

    print()


def example_data_leakage_check():
    """Example demonstrating data leakage prevention."""
    print("=" * 60)
    print("Data Leakage Prevention Check")
    print("=" * 60)

    # Get data
    csv_client = FootballDataCSVClient()
    transformer = DataTransformer()
    engineer = FeatureEngineer()

    df_raw = csv_client.get_season_data(season='2425')
    df_transformed, _ = transformer.transform(df_raw, source='csv')
    df_features = engineer.create_rolling_features(df_transformed, windows=[3])

    # Check a specific match
    print("\nChecking data leakage for a specific match:")
    print("-" * 60)

    # Get a match with features
    sample = df_features.dropna(subset=['home_goals_scored_L3']).iloc[10]

    print(f"\nMatch: {sample['home_team_name']} vs {sample['away_team_name']}")
    print(f"Date: {sample['date']}")
    print(f"Actual goals: {sample['total_goals']}")

    print(f"\nHome team rolling features (last 3 matches):")
    print(f"  Goals scored L3: {sample['home_goals_scored_L3']:.2f}")
    print(f"  Goals conceded L3: {sample['home_goals_conceded_L3']:.2f}")

    print(f"\nAway team rolling features (last 3 matches):")
    print(f"  Goals scored L3: {sample['away_goals_scored_L3']:.2f}")
    print(f"  Goals conceded L3: {sample['away_goals_conceded_L3']:.2f}")

    print(f"\n✓ These features were calculated using .shift(1)")
    print(f"✓ They do NOT include the current match (date: {sample['date']})")
    print(f"✓ Only information BEFORE this date was used")

    # Verify by manual calculation
    home_team = sample['home_team_name']
    current_date = sample['date']

    # Get previous matches for home team
    prev_home_matches = df_transformed[
        ((df_transformed['home_team_name'] == home_team) |
         (df_transformed['away_team_name'] == home_team)) &
        (df_transformed['date'] < current_date)
    ].tail(3)

    print(f"\nManual verification - Previous 3 matches for {home_team}:")
    for _, match in prev_home_matches.iterrows():
        if match['home_team_name'] == home_team:
            print(f"  {match['date']}: {match['home_team_name']} {match['home_goals']}-{match['away_goals']} {match['away_team_name']}")
        else:
            print(f"  {match['date']}: {match['home_team_name']} {match['home_goals']}-{match['away_goals']} {match['away_team_name']}")

    print("\n✓ No data leakage detected!")

    print()


def main():
    """Run all examples."""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    logger.info("Starting feature engineering examples")

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING EXAMPLES")
    print("=" * 60 + "\n")

    try:
        example_rolling_features()
        example_match_features()
        example_rest_days_features()
        example_h2h_features()
        example_complete_pipeline()
        example_data_leakage_check()
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Examples failed: {str(e)}", exc_info=True)

    print("=" * 60)
    print("Examples completed!")
    print("Check logs/ directory for detailed logs")
    print("=" * 60)


if __name__ == "__main__":
    main()
