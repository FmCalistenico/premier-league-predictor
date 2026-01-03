"""
Script to train Premier League prediction model.
Complete end-to-end training pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logging, get_logger
from pipelines import ModelPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Premier League prediction model'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to training data (default: training_data_latest.parquet)'
    )

    parser.add_argument(
        '--no-cv',
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
        '--no-save',
        action='store_true',
        help='Do not save trained model'
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

    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    if args.verbose:
        logger.setLevel('DEBUG')

    logger.info("=" * 60)
    logger.info("Premier League Model Training")
    logger.info("=" * 60)

    logger.info(f"Data path: {args.data or 'default (latest)'}")
    logger.info(f"Cross-validation: {not args.no_cv}")
    logger.info(f"CV splits: {args.cv_splits}")
    logger.info(f"Save model: {not args.no_save}")

    try:
        # Initialize pipeline
        pipeline = ModelPipeline()

        # Run training
        logger.info("\nStarting training pipeline...")

        model, metrics, cv_results = pipeline.run_full_training(
            data_path=args.data,
            cross_validate=not args.no_cv,
            save_model=not args.no_save,
            n_cv_splits=args.cv_splits
        )

        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        logger.info("\nEvaluation Metrics:")
        logger.info(f"  Accuracy:           {metrics['accuracy']:.4f}")
        logger.info(f"  Precision:          {metrics['precision']:.4f}")
        logger.info(f"  Recall:             {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:           {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC:            {metrics['roc_auc']:.4f}")
        logger.info(f"  Log Loss:           {metrics['log_loss']:.4f}")
        logger.info(f"  Brier Score:        {metrics['brier_score']:.4f}")
        logger.info(f"  Calibration Error:  {metrics['calibration_error']:.4f}")

        if cv_results is not None:
            mean_row = cv_results[cv_results['fold'] == 'mean'].iloc[0]
            std_row = cv_results[cv_results['fold'] == 'std'].iloc[0]

            logger.info("\nCross-Validation Results:")
            logger.info(f"  Accuracy:  {mean_row['accuracy']:.4f} ± {std_row['accuracy']:.4f}")
            logger.info(f"  ROC AUC:   {mean_row['roc_auc']:.4f} ± {std_row['roc_auc']:.4f}")
            logger.info(f"  F1 Score:  {mean_row['f1_score']:.4f} ± {std_row['f1_score']:.4f}")

        if not args.no_save:
            logger.info(f"\nModel saved to: {pipeline.models_dir}")
            logger.info(f"Plots saved to: {pipeline.plots_dir}")
            logger.info(f"Results saved to: {pipeline.results_dir}")

        logger.info("\n" + "=" * 60)

        sys.exit(0)

    except Exception as e:
        logger.error(f"\nTraining failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
