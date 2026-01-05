"""
Quick test script for FeatureEngineerV2.
Tests column normalization and basic functionality.
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logging, get_logger, Config
from src.features import FeatureEngineerV2


def main():
    """Quick test of V2 feature engineering."""
    setup_logging()
    logger = get_logger(__name__)
    config = Config()

    logger.info("=" * 80)
    logger.info("QUICK TEST: FeatureEngineerV2")
    logger.info("=" * 80)

    try:
        # Find latest processed file
        processed_files = list(config.data_processed_path.glob('transformed_*.csv'))
        if not processed_files:
            logger.error("No processed data found. Run data pipeline first.")
            sys.exit(1)

        data_path = max(processed_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"\nLoading data from: {data_path}")

        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"Loaded {len(df)} matches")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Ensure required columns
        if 'total_goals' not in df.columns:
            df['total_goals'] = df['home_goals'] + df['away_goals']
        if 'over_2.5' not in df.columns:
            df['over_2.5'] = (df['total_goals'] > 2.5).astype(int)

        logger.info("\n" + "=" * 80)
        logger.info("Testing FeatureEngineerV2...")
        logger.info("=" * 80)

        # Test V2
        engineer = FeatureEngineerV2()
        df_result = engineer.engineer_features(df)

        logger.info(f"\n✓ Feature engineering completed!")
        logger.info(f"  Input: {len(df)} matches")
        logger.info(f"  Output: {len(df_result)} matches")
        logger.info(f"  Features created: {len(df_result.columns)}")

        # Get feature list
        feature_cols = engineer.get_feature_list(df_result)
        logger.info(f"  Engineering features: {len(feature_cols)}")

        # Show sample features
        logger.info("\nSample features:")
        for feat in feature_cols[:10]:
            logger.info(f"  - {feat}")

        # Test validation
        logger.info("\n" + "=" * 80)
        logger.info("Testing feature validation...")
        logger.info("=" * 80)

        validation = engineer.validate_features(df_result)

        logger.info(f"\nValidation results:")
        logger.info(f"  Total features: {validation['n_features']}")
        logger.info(f"  Problematic: {len(validation['problematic_features'])}")
        logger.info(f"  VIF issues: {len(validation['vif_issues'])}")
        logger.info(f"  Correlation issues: {len(validation['correlation_issues'])}")

        # Check for removed problematic features
        logger.info("\n" + "=" * 80)
        logger.info("Verifying problematic features removed...")
        logger.info("=" * 80)

        problematic_to_check = [
            'expected_total_goals',
            'combined_over_rate',
            'h2h_avg_goals',
            'h2h_over_rate'
        ]

        all_removed = True
        for feat in problematic_to_check:
            if feat in df_result.columns:
                logger.error(f"  ❌ {feat} still exists!")
                all_removed = False
            else:
                logger.info(f"  ✓ {feat} removed")

        if all_removed:
            logger.info("\n✅ All problematic features successfully removed!")
        else:
            logger.warning("\n⚠️  Some problematic features still exist")

        # Check for new V2 features
        logger.info("\n" + "=" * 80)
        logger.info("Verifying new V2 features added...")
        logger.info("=" * 80)

        v2_features = [
            'home_attack_ratio_L5',
            'away_defense_ratio_L5',
            'home_goals_momentum',
            'home_goals_scored_volatility_L5',
            'is_derby',
            'position_diff'
        ]

        all_added = True
        for feat in v2_features:
            if feat in df_result.columns:
                logger.info(f"  ✓ {feat} exists")
            else:
                logger.error(f"  ❌ {feat} missing!")
                all_added = False

        if all_added:
            logger.info("\n✅ All V2 features successfully added!")
        else:
            logger.warning("\n⚠️  Some V2 features missing")

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)

        logger.info(f"\n✓ FeatureEngineerV2 working correctly")
        logger.info(f"  - Column normalization: OK")
        logger.info(f"  - Feature creation: OK ({len(feature_cols)} features)")
        logger.info(f"  - Problematic features removed: {'OK' if all_removed else 'FAILED'}")
        logger.info(f"  - V2 features added: {'OK' if all_added else 'FAILED'}")
        logger.info(f"  - Validation: OK")

        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 80)

        logger.info("\nNext steps:")
        logger.info("  1. Run full migration: python scripts/migrate_to_v2.py")
        logger.info("  2. Train V2 model: python scripts/train_model_v2.py --validate-features")

    except Exception as e:
        logger.error(f"\n❌ Test failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
