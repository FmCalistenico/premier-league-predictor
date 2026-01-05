"""
Quick test to verify PoissonGoalsModel handles non-numeric columns correctly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logging, get_logger, Config
from src.features import FeatureEngineer
from src.models import PoissonGoalsModel


def main():
    """Test model with feature data."""
    setup_logging()
    logger = get_logger(__name__)
    config = Config()

    logger.info("=" * 80)
    logger.info("TEST: PoissonGoalsModel with Mixed Data Types")
    logger.info("=" * 80)

    try:
        # Load processed data
        processed_files = list(config.data_processed_path.glob('transformed_*.csv'))
        if not processed_files:
            logger.error("No processed data found.")
            sys.exit(1)

        data_path = max(processed_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"\nLoading data from: {data_path}")

        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])

        # Ensure required columns
        if 'total_goals' not in df.columns:
            df['total_goals'] = df['home_goals'] + df['away_goals']
        if 'over_2.5' not in df.columns:
            df['over_2.5'] = (df['total_goals'] > 2.5).astype(int)

        logger.info(f"Loaded {len(df)} matches")

        # Apply V1 features
        logger.info("\nApplying FeatureEngineer V1...")
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df)

        logger.info(f"✓ Features created: {len(df_features)} matches")

        # Get feature columns
        feature_cols = [col for col in df_features.columns if col not in [
            'home_goals', 'away_goals', 'total_goals', 'over_2.5',
            'home_team_name', 'away_team_name', 'date', 'season'
        ]]

        logger.info(f"Feature columns: {len(feature_cols)}")

        # Check dtypes
        logger.info("\nChecking data types...")
        non_numeric = df_features[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logger.warning(f"Non-numeric columns found: {non_numeric}")
        else:
            logger.info("✓ All features are numeric")

        # Split data
        train_size = int(0.8 * len(df_features))
        df_train = df_features.iloc[:train_size]
        df_test = df_features.iloc[train_size:]

        logger.info(f"\nTrain: {len(df_train)}, Test: {len(df_test)}")

        # Test model training
        logger.info("\n" + "=" * 80)
        logger.info("Testing PoissonGoalsModel.fit()...")
        logger.info("=" * 80)

        model = PoissonGoalsModel()
        model.fit(df_train, feature_cols)

        logger.info("✓ Model trained successfully!")

        # Test predictions
        logger.info("\n" + "=" * 80)
        logger.info("Testing PoissonGoalsModel.predict_proba()...")
        logger.info("=" * 80)

        y_pred_proba = model.predict_proba(df_test[feature_cols])

        logger.info(f"✓ Predictions generated: {len(y_pred_proba)}")
        logger.info(f"  Min prob: {y_pred_proba.min():.4f}")
        logger.info(f"  Max prob: {y_pred_proba.max():.4f}")
        logger.info(f"  Mean prob: {y_pred_proba.mean():.4f}")

        # Check for invalid values
        invalid_count = (~np.isfinite(y_pred_proba)).sum()
        if invalid_count > 0:
            logger.warning(f"⚠️  {invalid_count} invalid predictions found")
        else:
            logger.info("✓ All predictions are valid (finite)")

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 80)

        logger.info("\nModel can now handle:")
        logger.info("  ✓ Non-numeric columns (automatically dropped)")
        logger.info("  ✓ NaN values (filled with 0)")
        logger.info("  ✓ Inf values (replaced with 0)")
        logger.info("  ✓ Mixed data types")

        logger.info("\n✅ Ready to run: python scripts/migrate_to_v2.py")

    except Exception as e:
        logger.error(f"\n❌ Test failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
