"""
Complete end-to-end pipeline for Premier League prediction.
Executes data extraction, transformation, feature engineering, and model training.
"""

import argparse
from pathlib import Path

from src.utils import setup_logging, get_logger, Config
from pipelines import DataPipeline, ModelPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run complete Premier League prediction pipeline'
    )

    parser.add_argument(
        '--source',
        type=str,
        choices=['csv', 'api'],
        default='csv',
        help='Data source (default: csv)'
    )

    parser.add_argument(
        '--seasons',
        nargs='+',
        default=['2223', '2324', '2425'],
        help='Seasons to extract for CSV (default: 2223 2324 2425)'
    )

    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip data pipeline (use existing training data)'
    )

    parser.add_argument(
        '--skip-cv',
        action='store_true',
        help='Skip cross-validation'
    )

    parser.add_argument(
        '--cv-splits',
        type=int,
        default=5,
        help='Number of cross-validation splits (default: 5)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: 2 seasons, no CV'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Run complete pipeline."""
    args = parse_arguments()

    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    if args.verbose:
        logger.setLevel('DEBUG')

    # Print header
    print("\n" + "=" * 80)
    print(" " * 20 + "PREMIER LEAGUE PREDICTION PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Source: {args.source}")
    print(f"  Seasons: {args.seasons}")
    print(f"  Skip data pipeline: {args.skip_data}")
    print(f"  Cross-validation: {not args.skip_cv}")
    print(f"  CV splits: {args.cv_splits}")
    print(f"  Quick mode: {args.quick}")
    print("=" * 80 + "\n")

    try:
        # PART 1: DATA PIPELINE
        if not args.skip_data:
            print("\n" + "#" * 80)
            print("# PART 1: DATA PIPELINE")
            print("#" * 80 + "\n")

            data_pipeline = DataPipeline()

            if args.quick:
                # Quick mode: only 2 seasons
                df_final = data_pipeline.run_full_pipeline(
                    source=args.source,
                    csv_seasons=args.seasons[:2] if args.source == 'csv' else None,
                    save=True
                )
            else:
                # Full mode: all seasons
                df_final = data_pipeline.run_full_pipeline(
                    source=args.source,
                    csv_seasons=args.seasons if args.source == 'csv' else None,
                    save=True
                )

            print(f"\n✓ Data pipeline completed: {len(df_final)} matches prepared")

        else:
            print("\n⚠ Skipping data pipeline (using existing training data)")

        # PART 2: MODEL TRAINING PIPELINE
        print("\n" + "#" * 80)
        print("# PART 2: MODEL TRAINING PIPELINE")
        print("#" * 80 + "\n")

        model_pipeline = ModelPipeline()

        if args.quick:
            # Quick mode: no cross-validation
            model, metrics, cv_results = model_pipeline.run_full_training(
                cross_validate=False,
                save_model=True
            )
        else:
            # Full mode: with cross-validation
            model, metrics, cv_results = model_pipeline.run_full_training(
                cross_validate=not args.skip_cv,
                save_model=True,
                n_cv_splits=args.cv_splits
            )

        print(f"\n✓ Model training completed")

        # FINAL SUMMARY
        print("\n" + "=" * 80)
        print(" " * 25 + "PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print("\n📊 FINAL RESULTS:")
        print("-" * 80)

        print(f"\nModel Performance:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Precision:          {metrics['precision']:.4f}")
        print(f"  Recall:             {metrics['recall']:.4f}")
        print(f"  F1 Score:           {metrics['f1_score']:.4f}")
        print(f"  ROC AUC:            {metrics['roc_auc']:.4f}")
        print(f"  Log Loss:           {metrics['log_loss']:.4f}")
        print(f"  Brier Score:        {metrics['brier_score']:.4f}")
        print(f"  Calibration Error:  {metrics['calibration_error']:.4f}")

        if cv_results is not None:
            mean_row = cv_results[cv_results['fold'] == 'mean'].iloc[0]
            std_row = cv_results[cv_results['fold'] == 'std'].iloc[0]

            print(f"\nCross-Validation Results ({args.cv_splits} folds):")
            print(f"  Accuracy:  {mean_row['accuracy']:.4f} ± {std_row['accuracy']:.4f}")
            print(f"  ROC AUC:   {mean_row['roc_auc']:.4f} ± {std_row['roc_auc']:.4f}")
            print(f"  F1 Score:  {mean_row['f1_score']:.4f} ± {std_row['f1_score']:.4f}")
            print(f"  Precision: {mean_row['precision']:.4f} ± {std_row['precision']:.4f}")
            print(f"  Recall:    {mean_row['recall']:.4f} ± {std_row['recall']:.4f}")

        print("\n📁 Output Files:")
        print("-" * 80)

        config = Config()

        if not args.skip_data:
            print(f"\nData Files:")
            print(f"  Training data: {config.data_final_path}/training_data_latest.parquet")
            print(f"  CSV export:    {config.data_final_path}/training_data_latest.csv")

        print(f"\nModel Files:")
        print(f"  Model:    {config.models_path}/poisson_model_latest.pkl")
        print(f"  Metadata: {config.models_path}/poisson_model_latest_metadata.json")

        if not args.skip_cv:
            print(f"\nResults Files:")
            print(f"  CV results: {config.models_path}/results/cv_results.csv")

        print(f"\nPlots:")
        print(f"  Location: {config.models_path}/plots/")
        print(f"    - calibration_curve.png")
        print(f"    - roc_curve.png")
        print(f"    - confusion_matrix.png")

        print(f"\nLogs:")
        print(f"  Location: {config.logs_path}/")

        print("\n" + "=" * 80)
        print(" " * 30 + "PIPELINE FINISHED!")
        print("=" * 80 + "\n")

        print("Next steps:")
        print("  1. Review evaluation plots in models/plots/")
        print("  2. Check CV results in models/results/cv_results.csv")
        print("  3. Use the trained model for predictions")
        print("  4. Monitor logs in logs/ directory")

        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print("\n" + "=" * 80)
        print(" " * 30 + "PIPELINE FAILED!")
        print("=" * 80)
        print(f"\n❌ Error: {str(e)}")
        print("\nCheck logs/ directory for details")
        print("=" * 80 + "\n")

        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
