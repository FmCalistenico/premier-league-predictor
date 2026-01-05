"""
Quick migration script to switch from FeatureEngineer V1 to V2.

This script:
1. Loads existing processed data
2. Applies FeatureEngineerV2
3. Validates features
4. Trains and compares both versions
5. Generates comparison report

Usage:
    python scripts/migrate_to_v2.py
    python scripts/migrate_to_v2.py --data data/processed/transformed_latest.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logging, get_logger, Config
from src.features import FeatureEngineer, FeatureEngineerV2
from src.models import PoissonGoalsModel
from src.models.evaluator import ModelEvaluator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Migrate from FeatureEngineer V1 to V2'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to processed data (default: latest transformed file)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: skip full comparison, just validate V2'
    )

    return parser.parse_args()


def compare_feature_sets(df_v1, df_v2, logger):
    """Compare feature sets between V1 and V2."""
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE SET COMPARISON")
    logger.info("=" * 80)

    # Get feature columns
    exclude_cols = ['home_goals', 'away_goals', 'total_goals', 'over_2.5',
                   'home_team', 'away_team', 'date', 'season']

    features_v1 = [col for col in df_v1.columns if col not in exclude_cols]
    features_v2 = [col for col in df_v2.columns if col not in exclude_cols]

    logger.info(f"\nV1 Features: {len(features_v1)}")
    logger.info(f"V2 Features: {len(features_v2)}")
    logger.info(f"Difference: {len(features_v2) - len(features_v1):+d}")

    # Find removed features
    removed = set(features_v1) - set(features_v2)
    logger.info(f"\n❌ Removed features ({len(removed)}):")
    for feat in sorted(removed):
        logger.info(f"  - {feat}")

    # Find added features
    added = set(features_v2) - set(features_v1)
    logger.info(f"\n✅ Added features ({len(added)}):")
    for feat in sorted(added):
        logger.info(f"  - {feat}")

    # Common features
    common = set(features_v1) & set(features_v2)
    logger.info(f"\n🔄 Common features: {len(common)}")

    return features_v1, features_v2


def compare_models(df_v1, df_v2, features_v1, features_v2, logger):
    """Train and compare V1 vs V2 models."""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)

    # Split data (80/20)
    train_size = int(0.8 * min(len(df_v1), len(df_v2)))

    # Ensure same samples are used (align by date)
    common_dates = set(df_v1['date']) & set(df_v2['date'])
    df_v1_aligned = df_v1[df_v1['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
    df_v2_aligned = df_v2[df_v2['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)

    train_size = int(0.8 * len(df_v1_aligned))

    df_v1_train = df_v1_aligned.iloc[:train_size]
    df_v1_test = df_v1_aligned.iloc[train_size:]

    df_v2_train = df_v2_aligned.iloc[:train_size]
    df_v2_test = df_v2_aligned.iloc[train_size:]

    logger.info(f"\nTraining samples: {train_size}")
    logger.info(f"Test samples: {len(df_v1_test)}")

    results = {}

    # Train V1 model
    logger.info("\n" + "-" * 80)
    logger.info("Training V1 Model...")
    logger.info("-" * 80)

    model_v1 = PoissonGoalsModel()
    model_v1.fit(df_v1_train, features_v1)

    y_true = df_v1_test['over_2.5'].values
    y_pred_proba_v1 = model_v1.predict_proba(df_v1_test[features_v1])

    # Extract probabilities for positive class (Over = class 1)
    if y_pred_proba_v1.ndim == 2:
        y_pred_proba_v1 = y_pred_proba_v1[:, 1]

    y_pred_proba_v1 = np.clip(y_pred_proba_v1, 0, 1)

    metrics_v1 = ModelEvaluator.evaluate_classification_metrics(y_true, y_pred_proba_v1)
    results['v1'] = metrics_v1

    logger.info("V1 Results:")
    logger.info(f"  ROC AUC: {metrics_v1['roc_auc']:.4f}")
    logger.info(f"  Accuracy: {metrics_v1['accuracy']:.4f}")
    logger.info(f"  F1 Score: {metrics_v1['f1_score']:.4f}")

    # Prediction distribution
    y_pred_v1 = (y_pred_proba_v1 > 0.5).astype(int)
    pred_over_v1 = y_pred_v1.mean()
    logger.info(f"  Pred Over: {pred_over_v1:.1%}")

    # Train V2 model
    logger.info("\n" + "-" * 80)
    logger.info("Training V2 Model...")
    logger.info("-" * 80)

    model_v2 = PoissonGoalsModel()
    model_v2.fit(df_v2_train, features_v2)

    y_pred_proba_v2 = model_v2.predict_proba(df_v2_test[features_v2])

    # Extract probabilities for positive class (Over = class 1)
    if y_pred_proba_v2.ndim == 2:
        y_pred_proba_v2 = y_pred_proba_v2[:, 1]

    y_pred_proba_v2 = np.clip(y_pred_proba_v2, 0, 1)

    metrics_v2 = ModelEvaluator.evaluate_classification_metrics(y_true, y_pred_proba_v2)
    results['v2'] = metrics_v2

    logger.info("V2 Results:")
    logger.info(f"  ROC AUC: {metrics_v2['roc_auc']:.4f}")
    logger.info(f"  Accuracy: {metrics_v2['accuracy']:.4f}")
    logger.info(f"  F1 Score: {metrics_v2['f1_score']:.4f}")

    # Prediction distribution
    y_pred_v2 = (y_pred_proba_v2 > 0.5).astype(int)
    pred_over_v2 = y_pred_v2.mean()
    logger.info(f"  Pred Over: {pred_over_v2:.1%}")

    # Compare
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)

    improvements = {
        'roc_auc': metrics_v2['roc_auc'] - metrics_v1['roc_auc'],
        'accuracy': metrics_v2['accuracy'] - metrics_v1['accuracy'],
        'f1_score': metrics_v2['f1_score'] - metrics_v1['f1_score'],
        'pred_balance': abs(pred_over_v2 - 0.595) - abs(pred_over_v1 - 0.595)  # Closer to actual rate
    }

    logger.info("\nMetric Improvements (V2 - V1):")
    for metric, improvement in improvements.items():
        sign = "+" if improvement > 0 else ""
        logger.info(f"  {metric:20s}: {sign}{improvement:.4f}")

    # Recommendation
    logger.info("\n" + "=" * 80)
    if metrics_v2['roc_auc'] > metrics_v1['roc_auc'] and abs(pred_over_v2 - 0.595) < abs(pred_over_v1 - 0.595):
        logger.info("✅ RECOMMENDATION: Migrate to V2")
        logger.info("   - Better ROC AUC")
        logger.info("   - More balanced predictions")
    elif metrics_v2['roc_auc'] > metrics_v1['roc_auc']:
        logger.info("⚠️  RECOMMENDATION: Consider V2")
        logger.info("   - Better ROC AUC")
        logger.info("   - But verify prediction balance with threshold optimization")
    else:
        logger.info("⚠️  RECOMMENDATION: Investigate further")
        logger.info("   - V2 may need hyperparameter tuning")
        logger.info("   - Check feature validation results")

    logger.info("=" * 80)

    return results


def main():
    """Main migration function."""
    args = parse_arguments()

    # Setup
    setup_logging()
    logger = get_logger(__name__)
    config = Config()

    logger.info("=" * 80)
    logger.info("MIGRATION TO FEATUREENGINEERV2")
    logger.info("=" * 80)

    try:
        # Load data
        if args.data:
            data_path = Path(args.data)
        else:
            # Find latest processed file
            processed_files = list(config.data_processed_path.glob('transformed_*.csv'))
            if not processed_files:
                raise FileNotFoundError("No processed data found. Run data pipeline first.")
            data_path = max(processed_files, key=lambda p: p.stat().st_mtime)

        logger.info(f"\nLoading data from: {data_path}")
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Loaded {len(df)} matches")

        # Ensure total_goals and over_2.5 exist
        if 'total_goals' not in df.columns:
            df['total_goals'] = df['home_goals'] + df['away_goals']
        if 'over_2.5' not in df.columns:
            df['over_2.5'] = (df['total_goals'] > 2.5).astype(int)

        logger.info(f"Columns available: {df.columns.tolist()}")

        # Apply V1 features
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: APPLY V1 FEATURES")
        logger.info("=" * 80)

        engineer_v1 = FeatureEngineer()
        df_v1 = engineer_v1.engineer_features(df.copy())
        logger.info(f"✓ V1 features created: {len(df_v1)} matches")

        # Apply V2 features
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: APPLY V2 FEATURES")
        logger.info("=" * 80)

        engineer_v2 = FeatureEngineerV2()
        df_v2 = engineer_v2.engineer_features(df.copy())
        logger.info(f"✓ V2 features created: {len(df_v2)} matches")

        # Validate V2 features
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: VALIDATE V2 FEATURES")
        logger.info("=" * 80)

        validation = engineer_v2.validate_features(df_v2)

        logger.info(f"\nValidation Results:")
        logger.info(f"  Total features: {validation['n_features']}")
        logger.info(f"  Problematic: {len(validation['problematic_features'])}")
        logger.info(f"  VIF issues: {len(validation['vif_issues'])}")
        logger.info(f"  Correlation issues: {len(validation['correlation_issues'])}")

        if validation['problematic_features']:
            logger.warning(f"\n⚠️  Problematic features found:")
            for feat in validation['problematic_features'][:10]:  # Show first 10
                logger.warning(f"  - {feat}")

        # Save validation
        validation_path = config.models_path / 'analysis' / 'migration_validation.csv'
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(validation['features']).to_csv(validation_path, index=False)
        logger.info(f"\n✓ Validation saved to {validation_path}")

        # Compare feature sets
        features_v1, features_v2 = compare_feature_sets(df_v1, df_v2, logger)

        # Compare models (unless quick mode)
        if not args.quick:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 4: COMPARE MODELS")
            logger.info("=" * 80)

            results = compare_models(df_v1, df_v2, features_v1, features_v2, logger)

            # Save comparison
            comparison_df = pd.DataFrame({
                'metric': list(results['v1'].keys()),
                'v1': list(results['v1'].values()),
                'v2': list(results['v2'].values()),
                'improvement': [results['v2'][k] - results['v1'][k] for k in results['v1'].keys()]
            })

            comparison_path = config.models_path / 'comparison' / 'migration_comparison.csv'
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(comparison_path, index=False)
            logger.info(f"\n✓ Comparison saved to {comparison_path}")

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("MIGRATION COMPLETE")
        logger.info("=" * 80)

        logger.info("\n📊 Summary:")
        logger.info(f"  V1 features: {len(features_v1)}")
        logger.info(f"  V2 features: {len(features_v2)}")
        logger.info(f"  Removed: {len(set(features_v1) - set(features_v2))}")
        logger.info(f"  Added: {len(set(features_v2) - set(features_v1))}")
        logger.info(f"  V2 problematic features: {len(validation['problematic_features'])}")

        logger.info("\n📁 Output files:")
        logger.info(f"  Validation: {validation_path}")
        if not args.quick:
            logger.info(f"  Comparison: {comparison_path}")

        logger.info("\n✅ Next steps:")
        logger.info("  1. Review validation results")
        if not args.quick:
            logger.info("  2. Check comparison metrics")
        logger.info("  3. If satisfied, train full model with:")
        logger.info("     python scripts/train_model_v2.py --validate-features")
        logger.info("  4. Compare with original model:")
        logger.info("     python scripts/compare_models.py")

        logger.info("\n" + "=" * 80)

    except Exception as e:
        logger.error(f"\n❌ Migration failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
