"""
Example usage of complete data pipeline.
Demonstrates end-to-end data processing.
"""

from pathlib import Path

from src.utils import setup_logging, get_logger
from src.features import FeatureEngineer
from pipelines import DataPipeline, run_quick_pipeline


def example_basic_pipeline():
    """Example of basic pipeline execution."""
    print("=" * 60)
    print("Basic Pipeline Example")
    print("=" * 60)

    # Initialize pipeline
    pipeline = DataPipeline()

    print("\nRunning pipeline for single season...")

    # Run for single recent season
    df_final = pipeline.run_full_pipeline(
        source='csv',
        csv_seasons=['2425'],
        save=True
    )

    print(f"\n✓ Pipeline completed!")
    print(f"  Final dataset: {len(df_final)} matches")
    print(f"  Columns: {len(df_final.columns)}")

    print()


def example_multi_season_pipeline():
    """Example of multi-season pipeline."""
    print("=" * 60)
    print("Multi-Season Pipeline Example")
    print("=" * 60)

    pipeline = DataPipeline()

    print("\nRunning pipeline for multiple seasons...")

    # Run for 3 seasons
    df_final = pipeline.run_full_pipeline(
        source='csv',
        csv_seasons=['2223', '2324', '2425'],
        save=True
    )

    print(f"\n✓ Pipeline completed!")
    print(f"  Final dataset: {len(df_final)} matches")
    print(f"  Features: {len(pipeline.feature_engineer.get_feature_names())}")

    # Show feature breakdown
    feature_names = pipeline.feature_engineer.get_feature_names()
    rolling = [f for f in feature_names if 'L' in f]
    match = [f for f in feature_names if 'expected' in f or 'form' in f or 'strength' in f]
    rest = [f for f in feature_names if 'rest' in f]
    h2h = [f for f in feature_names if 'h2h' in f]

    print(f"\n  Feature breakdown:")
    print(f"    - Rolling: {len(rolling)}")
    print(f"    - Match: {len(match)}")
    print(f"    - Rest: {len(rest)}")
    print(f"    - H2H: {len(h2h)}")

    print()


def example_step_by_step():
    """Example of running pipeline step by step."""
    print("=" * 60)
    print("Step-by-Step Pipeline Example")
    print("=" * 60)

    pipeline = DataPipeline()

    # Step 1: Extract
    print("\nStep 1: Extraction")
    print("-" * 40)
    raw_data = pipeline.run_extraction(
        source='csv',
        csv_seasons=['2425']
    )
    print(f"✓ Extracted {len(raw_data)} matches")

    # Step 2: Transform
    print("\nStep 2: Transformation")
    print("-" * 40)
    df_transformed = pipeline.run_transformation(raw_data, source='csv')
    print(f"✓ Transformed to {len(df_transformed)} matches")

    # Step 3: Feature Engineering
    print("\nStep 3: Feature Engineering")
    print("-" * 40)
    df_features = pipeline.run_feature_engineering(df_transformed)
    print(f"✓ Engineered features: {len(df_features)} matches")
    print(f"✓ Features created: {len(pipeline.feature_engineer.get_feature_names())}")

    # Step 4: Save
    print("\nStep 4: Save")
    print("-" * 40)
    files = pipeline.save_final_dataset(df_features)
    print(f"✓ Saved {len(files)} files")

    for file_type, file_path in files.items():
        print(f"  {file_type}: {file_path}")

    print()


def example_custom_features():
    """Example of pipeline with custom feature configuration."""
    print("=" * 60)
    print("Custom Features Pipeline Example")
    print("=" * 60)

    pipeline = DataPipeline()

    # Extract and transform
    print("\nExtracting and transforming data...")
    raw_data = pipeline.run_extraction(source='csv', csv_seasons=['2324', '2425'])
    df_transformed = pipeline.run_transformation(raw_data, source='csv')

    # Custom feature engineering
    print("\nEngineering features with custom configuration...")
    df_features = pipeline.run_feature_engineering(
        df_transformed,
        rolling_windows=[3, 5, 7, 10, 15],  # More windows
        h2h_window=10  # Longer H2H history
    )

    print(f"\n✓ Created {len(pipeline.feature_engineer.get_feature_names())} features")
    print(f"✓ Final dataset: {len(df_features)} matches")

    # Save
    files = pipeline.save_final_dataset(df_features, version='custom_v1')
    print(f"\n✓ Saved with custom version")

    print()


def example_inspect_dataset():
    """Example of inspecting saved dataset."""
    print("=" * 60)
    print("Inspect Dataset Example")
    print("=" * 60)

    pipeline = DataPipeline()

    # Check if latest dataset exists
    latest_file = pipeline.final_dir / 'training_data_latest.parquet'

    if latest_file.exists():
        print(f"\nLoading dataset from {latest_file}")

        import pandas as pd
        df = pd.read_parquet(latest_file)

        # Display info
        pipeline.log_dataset_summary(df)

        # Show sample
        print("\nSample data (first 5 matches):")
        display_cols = [
            'date', 'home_team_name', 'away_team_name',
            'total_goals', 'over_2.5',
            'expected_total_goals', 'combined_over_rate'
        ]

        if all(col in df.columns for col in display_cols):
            print(df[display_cols].head().to_string(index=False))
    else:
        print(f"\n✗ No dataset found at {latest_file}")
        print("  Run pipeline first to create dataset")

    print()


def example_quick_pipeline():
    """Example of quick pipeline for testing."""
    print("=" * 60)
    print("Quick Pipeline Example")
    print("=" * 60)

    print("\nRunning quick pipeline (2 seasons only)...")

    # Use convenience function
    df_final = run_quick_pipeline(source='csv')

    print(f"\n✓ Quick pipeline completed!")
    print(f"  Dataset: {len(df_final)} matches")
    print(f"  Columns: {len(df_final.columns)}")

    print()


def example_error_handling():
    """Example of pipeline error handling."""
    print("=" * 60)
    print("Error Handling Example")
    print("=" * 60)

    pipeline = DataPipeline()

    print("\nAttempting to extract invalid season...")

    try:
        # This might fail depending on data availability
        raw_data = pipeline.run_extraction(
            source='csv',
            csv_seasons=['9999']  # Invalid season
        )

        df_transformed = pipeline.run_transformation(raw_data, source='csv')
        df_features = pipeline.run_feature_engineering(df_transformed)

        print("\n✓ Pipeline completed despite potential issues")

    except Exception as e:
        print(f"\n✗ Pipeline failed as expected: {str(e)}")
        print("  Error was logged and handled gracefully")

    print()


def example_pipeline_comparison():
    """Example of comparing different pipeline configurations."""
    print("=" * 60)
    print("Pipeline Comparison Example")
    print("=" * 60)

    pipeline = DataPipeline()

    # Configuration 1: Standard windows
    print("\nConfiguration 1: Standard windows [3, 5, 10]")
    raw_data = pipeline.run_extraction(source='csv', csv_seasons=['2425'])
    df_transformed = pipeline.run_transformation(raw_data, source='csv')

    df_standard = pipeline.run_feature_engineering(
        df_transformed,
        rolling_windows=[3, 5, 10],
        h2h_window=5
    )

    features_standard = len(pipeline.feature_engineer.get_feature_names())
    matches_standard = len(df_standard)

    print(f"  Features: {features_standard}")
    print(f"  Matches: {matches_standard}")

    # Configuration 2: Extended windows
    print("\nConfiguration 2: Extended windows [3, 5, 10, 15, 20]")

    # Reset feature engineer
    pipeline.feature_engineer = FeatureEngineer()

    df_extended = pipeline.run_feature_engineering(
        df_transformed,
        rolling_windows=[3, 5, 10, 15, 20],
        h2h_window=10
    )

    features_extended = len(pipeline.feature_engineer.get_feature_names())
    matches_extended = len(df_extended)

    print(f"  Features: {features_extended}")
    print(f"  Matches: {matches_extended}")

    # Comparison
    print("\n" + "-" * 40)
    print("Comparison:")
    print(f"  Additional features: {features_extended - features_standard}")
    print(f"  Matches lost: {matches_standard - matches_extended}")

    print()


def main():
    """Run all examples."""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    logger.info("Starting pipeline examples")

    print("\n" + "=" * 60)
    print("DATA PIPELINE EXAMPLES")
    print("=" * 60 + "\n")

    try:
        # Run examples
        example_basic_pipeline()
        example_multi_season_pipeline()
        example_step_by_step()
        example_custom_features()
        example_inspect_dataset()
        example_quick_pipeline()
        example_error_handling()
        example_pipeline_comparison()

    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Examples failed: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()

    print("=" * 60)
    print("Examples completed!")
    print("Check data/final/ for saved datasets")
    print("Check logs/ for detailed logs")
    print("=" * 60)


if __name__ == "__main__":
    main()
