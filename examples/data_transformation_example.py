"""
Example usage of data transformation module.
Demonstrates parsing, cleaning, and standardization of football data.
"""

import pandas as pd
from pathlib import Path

from src.utils import setup_logging, get_logger
from src.data import FootballDataCSVClient, DataTransformer


def example_parse_csv():
    """Example of parsing CSV data."""
    print("=" * 60)
    print("Parse CSV Data Example")
    print("=" * 60)

    # Get CSV data
    csv_client = FootballDataCSVClient()
    df_raw = csv_client.get_season_data(season='2425')

    print(f"\nRaw CSV data shape: {df_raw.shape}")
    print(f"Raw columns: {df_raw.columns.tolist()[:10]}...")  # First 10 columns

    # Parse CSV
    transformer = DataTransformer()
    df_parsed = transformer.parse_csv_data(df_raw)

    print(f"\nParsed data shape: {df_parsed.shape}")
    print(f"Parsed columns: {df_parsed.columns.tolist()}")
    print("\nFirst few rows:")
    print(df_parsed.head())

    print()


def example_create_targets():
    """Example of creating target variables."""
    print("=" * 60)
    print("Create Target Variables Example")
    print("=" * 60)

    # Get and parse data
    csv_client = FootballDataCSVClient()
    transformer = DataTransformer()

    df_raw = csv_client.get_season_data(season='2425')
    df_parsed = transformer.parse_csv_data(df_raw)

    # Create targets
    df_with_targets = transformer.create_target_variables(df_parsed)

    print(f"\nData shape: {df_with_targets.shape}")
    print(f"\nTarget columns added:")
    target_cols = [col for col in df_with_targets.columns if col.startswith('over_')]
    print(target_cols)

    print("\nTarget distribution (over_2.5):")
    print(df_with_targets['over_2.5'].value_counts())
    print(f"\nPercentage of matches with over 2.5 goals: "
          f"{df_with_targets['over_2.5'].mean() * 100:.1f}%")

    print("\nSample rows with targets:")
    display_cols = ['date', 'home_team_name', 'away_team_name', 'total_goals'] + target_cols
    print(df_with_targets[display_cols].head(10))

    print()


def example_validate_data():
    """Example of data validation."""
    print("=" * 60)
    print("Data Validation Example")
    print("=" * 60)

    # Get and parse data
    csv_client = FootballDataCSVClient()
    transformer = DataTransformer()

    df_raw = csv_client.get_season_data(season='2425')
    df_parsed = transformer.parse_csv_data(df_raw)

    print(f"\nData before validation: {len(df_parsed)} rows")

    # Validate
    df_clean, issues = transformer.validate_data(df_parsed)

    print(f"Data after validation: {len(df_clean)} rows")
    print(f"Rows removed: {len(df_parsed) - len(df_clean)}")

    if issues:
        print(f"\nValidation issues found: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ No validation issues found!")

    print()


def example_standardize_teams():
    """Example of team name standardization."""
    print("=" * 60)
    print("Team Name Standardization Example")
    print("=" * 60)

    # Create sample data with non-standard names
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-08-01', periods=5),
        'home_team_name': ['Man City', 'Man Utd', 'Spurs', 'Newcastle', 'Brighton'],
        'away_team_name': ['Liverpool', 'Chelsea', 'Arsenal', 'Wolves', 'West Ham'],
        'home_goals': [2, 1, 3, 0, 2],
        'away_goals': [1, 1, 2, 1, 0],
        'total_goals': [3, 2, 5, 1, 2],
        'fixture_id': ['f1', 'f2', 'f3', 'f4', 'f5'],
    })

    print("\nBefore standardization:")
    print(sample_data[['home_team_name', 'away_team_name']])

    # Standardize
    transformer = DataTransformer()
    df_std = transformer.standardize_team_names(sample_data)

    print("\nAfter standardization:")
    print(df_std[['home_team_name', 'away_team_name']])

    print("\nTeam mapping applied:")
    for old_name, new_name in transformer.TEAM_NAME_MAPPING.items():
        print(f"  {old_name} → {new_name}")

    print()


def example_complete_pipeline():
    """Example of complete transformation pipeline."""
    print("=" * 60)
    print("Complete Transformation Pipeline Example")
    print("=" * 60)

    # Get raw data
    csv_client = FootballDataCSVClient()
    df_raw = csv_client.get_season_data(season='2425')

    print(f"\nRaw data: {len(df_raw)} rows, {len(df_raw.columns)} columns")

    # Transform
    transformer = DataTransformer()
    df_transformed, metadata = transformer.transform(df_raw, source='csv')

    print(f"\nTransformed data: {len(df_transformed)} rows, {len(df_transformed.columns)} columns")

    print("\n" + "=" * 60)
    print("Transformation Metadata")
    print("=" * 60)

    print(f"\nSource: {metadata['source']}")
    print(f"Initial rows: {metadata['initial_rows']}")
    print(f"Final rows: {metadata['final_rows']}")
    print(f"Rows removed: {metadata['initial_rows'] - metadata['final_rows']}")

    print(f"\nPipeline steps executed:")
    for step in metadata['pipeline_steps']:
        print(f"  ✓ {step}")

    if metadata['issues']:
        print(f"\nIssues encountered: {len(metadata['issues'])}")
        for issue in metadata['issues']:
            print(f"  - {issue}")

    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)

    summary = metadata['summary']
    print(f"\nDate range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Total matches: {summary['total_matches']}")
    print(f"Unique teams: {summary['unique_teams']}")
    print(f"Avg goals per match: {summary['avg_goals_per_match']:.2f}")
    print(f"Over 2.5 goals: {summary['over_2.5_pct']:.1f}%")

    print("\nTransformed data columns:")
    print(df_transformed.columns.tolist())

    print("\nSample transformed data:")
    display_cols = [
        'date', 'home_team_name', 'away_team_name',
        'home_goals', 'away_goals', 'total_goals',
        'over_2.5'
    ]
    print(df_transformed[display_cols].head(10))

    print()


def example_save_transformed_data():
    """Example of transforming and saving data."""
    print("=" * 60)
    print("Transform and Save Data Example")
    print("=" * 60)

    # Get multiple seasons
    csv_client = FootballDataCSVClient()
    df_raw = csv_client.get_multiple_seasons(['2223', '2324', '2425'])

    print(f"\nRaw data: {len(df_raw)} matches across 3 seasons")

    # Transform
    transformer = DataTransformer()
    df_transformed, metadata = transformer.transform(df_raw, source='csv')

    print(f"Transformed data: {len(df_transformed)} matches")

    # Save transformed data
    from pathlib import Path
    import json

    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    output_file = output_dir / 'premier_league_transformed.csv'
    df_transformed.to_csv(output_file, index=False)
    print(f"\n✓ Saved transformed data to: {output_file}")

    # Save metadata
    metadata_file = output_dir / 'premier_league_transformed_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved metadata to: {metadata_file}")

    print(f"\nFiles created:")
    print(f"  - {output_file} ({output_file.stat().st_size / 1024:.1f} KB)")
    print(f"  - {metadata_file} ({metadata_file.stat().st_size / 1024:.1f} KB)")

    print()


def main():
    """Run all examples."""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    logger.info("Starting data transformation examples")

    print("\n" + "=" * 60)
    print("DATA TRANSFORMATION EXAMPLES")
    print("=" * 60 + "\n")

    try:
        example_parse_csv()
        example_create_targets()
        example_validate_data()
        example_standardize_teams()
        example_complete_pipeline()
        example_save_transformed_data()
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Examples failed: {str(e)}", exc_info=True)

    print("=" * 60)
    print("Examples completed!")
    print("Check data/processed/ for transformed data")
    print("Check logs/ directory for detailed logs")
    print("=" * 60)


if __name__ == "__main__":
    main()
