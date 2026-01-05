"""
Script to train model using FeatureEngineerV2 (enhanced features).

This script demonstrates how to use the improved feature set to train
a model with reduced bias toward Over predictions.

Usage:
    python scripts/train_model_v2.py
    python scripts/train_model_v2.py --seasons 2223 2324 2425
    python scripts/train_model_v2.py --validate-features
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logging, get_logger, Config
from src.data import DataExtractor, DataTransformer
from src.features import FeatureEngineerV2
from src.models import PoissonGoalsModel
from src.models.evaluator import ModelEvaluator, TimeSeriesValidator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train model with FeatureEngineerV2'
    )

    parser.add_argument(
        '--seasons',
        nargs='+',
        default=['2223', '2324', '2425'],
        help='Seasons to use (default: 2223 2324 2425)'
    )

    parser.add_argument(
        '--validate-features',
        action='store_true',
        help='Run feature validation (VIF and correlation checks)'
    )

    parser.add_argument(
        '--cv-splits',
        type=int,
        default=5,
        help='Number of cross-validation splits (default: 5)'
    )

    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip data extraction (use existing raw data)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()

    # Setup
    setup_logging()
    logger = get_logger(__name__)
    config = Config()

    if args.verbose:
        logger.setLevel('DEBUG')

    logger.info("=" * 80)
    logger.info("TRAINING WITH FEATUREENGINEERV2")
    logger.info("=" * 80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Seasons: {args.seasons}")
    logger.info(f"  Validate features: {args.validate_features}")
    logger.info(f"  CV splits: {args.cv_splits}")
    logger.info("=" * 80)

    try:
        # STAGE 1: Extract data
        if not args.skip_extraction:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 1: DATA EXTRACTION")
            logger.info("=" * 80)

            extractor = DataExtractor()
            raw_data = extractor.run_extraction(
                sources=['csv'],
                csv_seasons=args.seasons
            )

            logger.info(f"✓ Extracted {len(raw_data)} matches")
        else:
            logger.info("\n⚠ Skipping extraction, loading from processed data...")

        # STAGE 2: Transform data
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: DATA TRANSFORMATION")
        logger.info("=" * 80)

        if not args.skip_extraction:
            transformer = DataTransformer()
            df_transformed, metadata = transformer.transform(raw_data, source='csv')
            logger.info(f"✓ Transformed {len(df_transformed)} matches")
        else:
            # Load from processed
            processed_files = list(config.data_processed_path.glob('transformed_*.csv'))
            if processed_files:
                latest_file = max(processed_files, key=lambda p: p.stat().st_mtime)
                df_transformed = pd.read_csv(latest_file)
                df_transformed['date'] = pd.to_datetime(df_transformed['date'])
                logger.info(f"✓ Loaded {len(df_transformed)} matches from {latest_file.name}")
            else:
                raise FileNotFoundError("No processed data found. Run without --skip-extraction")

        # Ensure total_goals and over_2.5 exist
        if 'total_goals' not in df_transformed.columns:
            df_transformed['total_goals'] = df_transformed['home_goals'] + df_transformed['away_goals']
        if 'over_2.5' not in df_transformed.columns:
            df_transformed['over_2.5'] = (df_transformed['total_goals'] > 2.5).astype(int)

        # STAGE 3: Feature Engineering V2
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: FEATURE ENGINEERING V2")
        logger.info("=" * 80)

        engineer = FeatureEngineerV2()
        df_features = engineer.engineer_features(df_transformed)

        logger.info(f"✓ Created features: {len(df_features)} matches")

        # Get feature list
        feature_cols = engineer.get_feature_list(df_features)
        logger.info(f"✓ Total features: {len(feature_cols)}")

        # Validate features if requested
        if args.validate_features:
            logger.info("\n" + "-" * 80)
            logger.info("FEATURE VALIDATION")
            logger.info("-" * 80)

            validation = engineer.validate_features(df_features)

            logger.info(f"\nValidation Results:")
            logger.info(f"  Total features: {validation['n_features']}")
            logger.info(f"  Problematic features: {len(validation['problematic_features'])}")
            logger.info(f"  VIF issues (>10): {len(validation['vif_issues'])}")
            logger.info(f"  Correlation issues (>0.7): {len(validation['correlation_issues'])}")

            if validation['problematic_features']:
                logger.warning(f"\nProblematic features found:")
                for feat_info in validation['features']:
                    if feat_info['is_problematic']:
                        logger.warning(f"  - {feat_info['feature_name']}: {feat_info.get('issue', 'unknown')}")

            # Save validation results
            validation_df = pd.DataFrame(validation['features'])
            validation_path = config.models_path / 'analysis' / 'feature_validation_v2.csv'
            validation_path.parent.mkdir(parents=True, exist_ok=True)
            validation_df.to_csv(validation_path, index=False)
            logger.info(f"\n✓ Validation results saved to {validation_path}")

        # STAGE 4: Train-test split
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4: PREPARE DATA")
        logger.info("=" * 80)

        train_size = int(0.8 * len(df_features))
        df_train = df_features.iloc[:train_size]
        df_test = df_features.iloc[train_size:]

        logger.info(f"Training set: {len(df_train)} matches")
        logger.info(f"Test set: {len(df_test)} matches")

        # Check class balance
        train_over_rate = df_train['over_2.5'].mean()
        test_over_rate = df_test['over_2.5'].mean()

        logger.info(f"\nClass distribution:")
        logger.info(f"  Train: {train_over_rate:.1%} Over, {1-train_over_rate:.1%} Under")
        logger.info(f"  Test:  {test_over_rate:.1%} Over, {1-test_over_rate:.1%} Under")

        # STAGE 5: Train model
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 5: TRAIN MODEL")
        logger.info("=" * 80)

        model = PoissonGoalsModel()
        model.fit(df_train, feature_cols)

        logger.info("✓ Model trained successfully")

        # STAGE 6: Evaluate
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 6: EVALUATE MODEL")
        logger.info("=" * 80)

        y_true = df_test['over_2.5'].values
        y_pred_proba = model.predict_proba(df_test[feature_cols])

        # Clean predictions
        import numpy as np
        y_pred_proba = np.clip(y_pred_proba, 0.0, 1.0)

        metrics = ModelEvaluator.evaluate_classification_metrics(
            y_true,
            y_pred_proba,
            threshold=0.5
        )

        logger.info("\nTest Set Metrics:")
        logger.info(f"  Accuracy:           {metrics['accuracy']:.4f}")
        logger.info(f"  Precision:          {metrics['precision']:.4f}")
        logger.info(f"  Recall:             {metrics['recall']:.4f}")
        logger.info(f"  Specificity:        {metrics.get('specificity', 0):.4f}")
        logger.info(f"  F1 Score:           {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC:            {metrics['roc_auc']:.4f}")
        logger.info(f"  Log Loss:           {metrics['log_loss']:.4f}")
        logger.info(f"  Brier Score:        {metrics['brier_score']:.4f}")
        logger.info(f"  Calibration Error:  {metrics.get('calibration_error', 0):.4f}")

        # Prediction distribution
        y_pred = (y_pred_proba > 0.5).astype(int)
        pred_over_rate = y_pred.mean()
        logger.info(f"\nPrediction Distribution:")
        logger.info(f"  Predicted Over:  {pred_over_rate:.1%}")
        logger.info(f"  Predicted Under: {1-pred_over_rate:.1%}")

        # STAGE 7: Cross-validation
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 7: CROSS-VALIDATION")
        logger.info("=" * 80)

        validator = TimeSeriesValidator()
        cv_results = validator.validate(
            model.__class__(),  # New instance for CV
            df_features,
            feature_cols,
            n_splits=args.cv_splits
        )

        mean_row = cv_results[cv_results['fold'] == 'mean'].iloc[0]
        std_row = cv_results[cv_results['fold'] == 'std'].iloc[0]

        logger.info(f"\nCross-Validation Results ({args.cv_splits} folds):")
        logger.info(f"  Accuracy:  {mean_row['accuracy']:.4f} ± {std_row['accuracy']:.4f}")
        logger.info(f"  ROC AUC:   {mean_row['roc_auc']:.4f} ± {std_row['roc_auc']:.4f}")
        logger.info(f"  F1 Score:  {mean_row['f1_score']:.4f} ± {std_row['f1_score']:.4f}")
        logger.info(f"  Precision: {mean_row['precision']:.4f} ± {std_row['precision']:.4f}")
        logger.info(f"  Recall:    {mean_row['recall']:.4f} ± {std_row['recall']:.4f}")

        # STAGE 8: Save model and results
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 8: SAVE MODEL")
        logger.info("=" * 80)

        # Save model
        model_path = config.models_path / 'poisson_model_v2.pkl'
        model.save(model_path)
        logger.info(f"✓ Model saved to {model_path}")

        # Save feature list
        feature_list_path = config.models_path / 'feature_list_v2.txt'
        with open(feature_list_path, 'w') as f:
            f.write("FEATURE LIST - V2 (Enhanced)\n")
            f.write("=" * 80 + "\n\n")
            for i, feat in enumerate(feature_cols, 1):
                f.write(f"{i}. {feat}\n")
        logger.info(f"✓ Feature list saved to {feature_list_path}")

        # Save CV results
        cv_path = config.models_path / 'results' / 'cv_results_v2.csv'
        cv_path.parent.mkdir(parents=True, exist_ok=True)
        cv_results.to_csv(cv_path, index=False)
        logger.info(f"✓ CV results saved to {cv_path}")

        # Save training data
        data_path = config.data_final_path / 'training_data_v2.parquet'
        df_features.to_parquet(data_path, index=False)
        logger.info(f"✓ Training data saved to {data_path}")

        # FINAL SUMMARY
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        logger.info("\n📊 Summary:")
        logger.info(f"  Features (V2): {len(feature_cols)}")
        logger.info(f"  Training samples: {len(df_train)}")
        logger.info(f"  Test ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  CV ROC AUC: {mean_row['roc_auc']:.4f} ± {std_row['roc_auc']:.4f}")
        logger.info(f"  Prediction balance: {pred_over_rate:.1%} Over / {1-pred_over_rate:.1%} Under")

        logger.info("\n📁 Output files:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Data: {data_path}")
        logger.info(f"  CV results: {cv_path}")
        logger.info(f"  Feature list: {feature_list_path}")

        if args.validate_features:
            logger.info(f"  Validation: {validation_path}")

        logger.info("\n✅ Next steps:")
        logger.info("  1. Compare with original model using scripts/compare_models.py")
        logger.info("  2. Optimize threshold using scripts/optimize_threshold.py")
        logger.info("  3. Analyze features using scripts/analyze_features.py")

        logger.info("\n" + "=" * 80)

    except Exception as e:
        logger.error(f"\n❌ Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
