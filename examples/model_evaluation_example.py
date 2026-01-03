"""
Example usage of model evaluation tools.
Demonstrates metrics calculation, visualization, and time series validation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import setup_logging, get_logger
from src.data import FootballDataCSVClient, DataTransformer
from src.features import FeatureEngineer
from src.models import PoissonGoalsModel, ModelEvaluator, TimeSeriesValidator


def prepare_data():
    """Prepare data for evaluation."""
    print("Preparing data...")

    csv_client = FootballDataCSVClient()
    df_raw = csv_client.get_multiple_seasons(['2223', '2324', '2425'])

    transformer = DataTransformer()
    df_transformed, _ = transformer.transform(df_raw, source='csv')

    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df_transformed)
    feature_cols = engineer.get_feature_names()

    return df_features, feature_cols


def example_basic_metrics():
    """Example of calculating basic metrics."""
    print("=" * 60)
    print("Basic Metrics Example")
    print("=" * 60)

    # Prepare data
    df, feature_cols = prepare_data()

    # Split
    train_df = df[df['date'] < '2024-08-01']
    test_df = df[df['date'] >= '2024-08-01']

    print(f"\nTrain: {len(train_df)} matches")
    print(f"Test: {len(test_df)} matches")

    # Train model
    print("\nTraining model...")
    model = PoissonGoalsModel(threshold=2.5)
    model.fit(train_df, feature_cols)

    # Predict
    print("Making predictions...")
    y_true = test_df['over_2.5'].values
    y_pred_proba = model.predict_proba(test_df)[:, 1]

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = ModelEvaluator.evaluate_classification_metrics(y_true, y_pred_proba)

    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS")
    print("=" * 60)

    for metric_name, value in metrics.items():
        print(f"{metric_name:20s}: {value:.4f}")

    print()


def example_calibration():
    """Example of calibration analysis."""
    print("=" * 60)
    print("Calibration Analysis Example")
    print("=" * 60)

    # Prepare data
    df, feature_cols = prepare_data()
    train_df = df[df['date'] < '2024-08-01']
    test_df = df[df['date'] >= '2024-08-01']

    # Train and predict
    model = PoissonGoalsModel(threshold=2.5)
    model.fit(train_df, feature_cols)

    y_true = test_df['over_2.5'].values
    y_pred_proba = model.predict_proba(test_df)[:, 1]

    # Calculate calibration error
    print("\nCalculating calibration error...")
    cal_error = ModelEvaluator.calculate_calibration_error(
        y_true, y_pred_proba, n_bins=10
    )

    print(f"\nExpected Calibration Error (ECE): {cal_error:.4f}")
    print("\nInterpretation:")
    if cal_error < 0.05:
        print("  ✓ Excellent calibration (< 0.05)")
    elif cal_error < 0.10:
        print("  ✓ Good calibration (< 0.10)")
    elif cal_error < 0.15:
        print("  ⚠ Fair calibration (< 0.15)")
    else:
        print("  ✗ Poor calibration (≥ 0.15)")

    # Plot calibration curve
    print("\nGenerating calibration curve...")
    plots_dir = Path('evaluation_plots')
    plots_dir.mkdir(exist_ok=True)

    ModelEvaluator.plot_calibration_curve(
        y_true,
        y_pred_proba,
        save_path=str(plots_dir / 'calibration_example.png')
    )

    print()


def example_visualization():
    """Example of generating all visualizations."""
    print("=" * 60)
    print("Visualization Example")
    print("=" * 60)

    # Prepare data
    df, feature_cols = prepare_data()
    train_df = df[df['date'] < '2024-08-01']
    test_df = df[df['date'] >= '2024-08-01']

    # Train and predict
    print("\nTraining model...")
    model = PoissonGoalsModel(threshold=2.5)
    model.fit(train_df, feature_cols)

    print("Making predictions...")
    y_true = test_df['over_2.5'].values
    y_pred_proba = model.predict_proba(test_df)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Create plots directory
    plots_dir = Path('evaluation_plots')
    plots_dir.mkdir(exist_ok=True)

    print("\nGenerating plots...")

    # 1. Calibration curve
    print("  1. Calibration curve...")
    ModelEvaluator.plot_calibration_curve(
        y_true,
        y_pred_proba,
        save_path=str(plots_dir / 'calibration_curve.png')
    )

    # 2. ROC curve
    print("  2. ROC curve...")
    ModelEvaluator.plot_roc_curve(
        y_true,
        y_pred_proba,
        save_path=str(plots_dir / 'roc_curve.png')
    )

    # 3. Confusion matrix
    print("  3. Confusion matrix...")
    ModelEvaluator.plot_confusion_matrix(
        y_true,
        y_pred,
        save_path=str(plots_dir / 'confusion_matrix.png')
    )

    print(f"\n✓ All plots saved to {plots_dir}/")

    print()


def example_complete_evaluation():
    """Example of complete model evaluation."""
    print("=" * 60)
    print("Complete Evaluation Example")
    print("=" * 60)

    # Prepare data
    df, feature_cols = prepare_data()
    train_df = df[df['date'] < '2024-08-01']
    test_df = df[df['date'] >= '2024-08-01']

    print(f"\nDataset: {len(df)} total matches")
    print(f"Train: {len(train_df)} matches")
    print(f"Test: {len(test_df)} matches")

    # Train model
    print("\nTraining model...")
    model = PoissonGoalsModel(threshold=2.5)
    model.fit(train_df, feature_cols)

    # Predict
    print("Making predictions...\n")
    y_true = test_df['over_2.5'].values
    y_pred_proba = model.predict_proba(test_df)[:, 1]

    # Complete evaluation
    plots_dir = Path('evaluation_plots')

    metrics = ModelEvaluator.evaluate_model(
        y_true,
        y_pred_proba,
        plot=True,
        save_dir=str(plots_dir)
    )

    print("\nEvaluation complete!")
    print(f"Plots saved to {plots_dir}/")

    print()


def example_time_series_validation():
    """Example of time series cross-validation."""
    print("=" * 60)
    print("Time Series Cross-Validation Example")
    print("=" * 60)

    # Prepare data
    df, feature_cols = prepare_data()

    print(f"\nDataset: {len(df)} matches")
    print("Using TimeSeriesSplit (respects temporal order)")

    # Initialize validator
    validator = TimeSeriesValidator()

    # Create fresh model instance
    model = PoissonGoalsModel(threshold=2.5)

    # Perform validation
    print("\nRunning 5-fold time series cross-validation...\n")

    results_df = validator.validate(
        model=model,
        df=df,
        feature_cols=feature_cols,
        target_col='over_2.5',
        n_splits=5
    )

    # Display results
    print("\n" + "=" * 60)
    print("FOLD-BY-FOLD RESULTS")
    print("=" * 60)

    display_cols = ['fold', 'accuracy', 'roc_auc', 'f1_score', 'train_samples', 'test_samples']
    print(results_df[display_cols].to_string(index=False))

    # Save results
    results_dir = Path('evaluation_results')
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / 'cv_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to {results_file}")

    print()


def example_validation_with_plots():
    """Example of validation with plot generation."""
    print("=" * 60)
    print("Validation with Plots Example")
    print("=" * 60)

    # Prepare data
    df, feature_cols = prepare_data()

    print(f"\nDataset: {len(df)} matches")

    # Initialize validator
    validator = TimeSeriesValidator()

    # Create model
    model = PoissonGoalsModel(threshold=2.5)

    # Validation with plots
    print("\nRunning validation with plot generation...\n")

    plots_dir = Path('evaluation_plots')

    results_df, final_metrics = validator.validate_with_plots(
        model=model,
        df=df,
        feature_cols=feature_cols,
        target_col='over_2.5',
        n_splits=5,
        save_dir=str(plots_dir)
    )

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    # Get mean metrics
    mean_row = results_df[results_df['fold'] == 'mean'].iloc[0]

    print("\nCross-Validation Metrics (Mean ± Std):")
    metric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    for metric in metric_cols:
        mean_val = mean_row[metric]
        std_row = results_df[results_df['fold'] == 'std'].iloc[0]
        std_val = std_row[metric]
        print(f"  {metric:15s}: {mean_val:.4f} ± {std_val:.4f}")

    print(f"\n✓ Plots saved to {plots_dir}/")

    print()


def example_compare_thresholds():
    """Example of comparing different thresholds."""
    print("=" * 60)
    print("Threshold Comparison Example")
    print("=" * 60)

    # Prepare data
    df, feature_cols = prepare_data()
    train_df = df[df['date'] < '2024-08-01']
    test_df = df[df['date'] >= '2024-08-01']

    print(f"\nComparing thresholds: 1.5, 2.5, 3.5 goals")

    thresholds = [1.5, 2.5, 3.5]
    results = []

    for threshold in thresholds:
        print(f"\n{'='*40}")
        print(f"Threshold: Over {threshold} goals")
        print('='*40)

        # Create target variable
        target_col = f'over_{threshold}'

        # Train model
        model = PoissonGoalsModel(threshold=threshold)
        model.fit(train_df, feature_cols, target_home='home_goals', target_away='away_goals')

        # Predict
        y_true = test_df[target_col].values
        y_pred_proba = model.predict_proba(test_df)[:, 1]

        # Evaluate
        metrics = ModelEvaluator.evaluate_classification_metrics(y_true, y_pred_proba)
        metrics['threshold'] = threshold

        results.append(metrics)

        # Print key metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ROC AUC:  {metrics['roc_auc']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("THRESHOLD COMPARISON SUMMARY")
    print("=" * 60)

    comparison_df = pd.DataFrame(results)
    print(comparison_df[['threshold', 'accuracy', 'roc_auc', 'f1_score']].to_string(index=False))

    print()


def main():
    """Run all examples."""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    logger.info("Starting model evaluation examples")

    print("\n" + "=" * 60)
    print("MODEL EVALUATION EXAMPLES")
    print("=" * 60 + "\n")

    try:
        example_basic_metrics()
        example_calibration()
        example_visualization()
        example_complete_evaluation()
        example_time_series_validation()
        example_validation_with_plots()
        example_compare_thresholds()
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Examples failed: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()

    print("=" * 60)
    print("Examples completed!")
    print("Check evaluation_plots/ for visualizations")
    print("Check evaluation_results/ for CSV results")
    print("Check logs/ for detailed logs")
    print("=" * 60)


if __name__ == "__main__":
    main()
