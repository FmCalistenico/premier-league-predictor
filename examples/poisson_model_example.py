"""
Example usage of Poisson Goals Model.
Demonstrates training, prediction, and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from src.utils import setup_logging, get_logger
from src.data import FootballDataCSVClient, DataTransformer
from src.features import FeatureEngineer
from src.models import PoissonGoalsModel


def prepare_data():
    """Prepare data for modeling."""
    print("=" * 60)
    print("Data Preparation")
    print("=" * 60)

    # Get data
    csv_client = FootballDataCSVClient()
    df_raw = csv_client.get_multiple_seasons(['2223', '2324', '2425'])
    print(f"\nRaw data: {len(df_raw)} matches")

    # Transform
    transformer = DataTransformer()
    df_transformed, _ = transformer.transform(df_raw, source='csv')
    print(f"Transformed data: {len(df_transformed)} matches")

    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df_transformed)
    print(f"Feature engineering: {len(df_features)} matches with features")

    # Get feature names
    feature_cols = engineer.get_feature_names()
    print(f"Features created: {len(feature_cols)}")

    return df_features, feature_cols


def example_basic_training():
    """Example of basic model training."""
    print("\n" + "=" * 60)
    print("Basic Model Training Example")
    print("=" * 60)

    # Prepare data
    df, feature_cols = prepare_data()

    # Split data
    train_df = df[df['date'] < '2024-08-01']
    test_df = df[df['date'] >= '2024-08-01']

    print(f"\nTrain set: {len(train_df)} matches")
    print(f"Test set: {len(test_df)} matches")

    # Train model
    print("\nTraining Poisson model...")
    model = PoissonGoalsModel(threshold=2.5)
    model.fit(train_df, feature_cols)

    print(f"\nModel trained: {model}")

    # Get coefficients
    coef_df = model.get_coefficients()
    print("\nTop 10 features by home coefficient:")
    print(coef_df.nlargest(10, 'home_coef')[['feature', 'home_coef', 'home_p_value']])

    print()


def example_predictions():
    """Example of making predictions."""
    print("=" * 60)
    print("Model Predictions Example")
    print("=" * 60)

    # Prepare data
    df, feature_cols = prepare_data()

    # Split
    train_df = df[df['date'] < '2024-08-01']
    test_df = df[df['date'] >= '2024-08-01'].head(10)

    # Train
    model = PoissonGoalsModel(threshold=2.5)
    model.fit(train_df, feature_cols)

    # Predict expected goals
    print("\nPredicting expected goals...")
    lambda_home, lambda_away = model.predict_expected_goals(test_df)

    print("\nSample predictions:")
    for i in range(min(5, len(test_df))):
        row = test_df.iloc[i]
        print(f"\n{row['home_team_name']} vs {row['away_team_name']}")
        print(f"  Expected: {lambda_home[i]:.2f} - {lambda_away[i]:.2f}")
        print(f"  Actual: {row['home_goals']:.0f} - {row['away_goals']:.0f}")

    # Predict over/under
    print("\n" + "-" * 60)
    print("Over/Under Predictions (threshold=2.5)")
    print("-" * 60)

    prob_over, prob_under, expected_total = model.predict_over_under(test_df)

    for i in range(min(5, len(test_df))):
        row = test_df.iloc[i]
        print(f"\n{row['home_team_name']} vs {row['away_team_name']}")
        print(f"  Expected total: {expected_total[i]:.2f}")
        print(f"  P(Over 2.5): {prob_over[i]:.3f}")
        print(f"  P(Under 2.5): {prob_under[i]:.3f}")
        print(f"  Actual total: {row['total_goals']:.0f} ({'Over' if row['total_goals'] > 2.5 else 'Under'})")

    print()


def example_analytical_calculation():
    """Example of analytical probability calculation."""
    print("=" * 60)
    print("Analytical Probability Calculation")
    print("=" * 60)

    model = PoissonGoalsModel(threshold=2.5)

    # Example scenarios
    scenarios = [
        (1.5, 1.5, "Balanced match"),
        (2.0, 1.0, "Home team stronger"),
        (0.8, 0.8, "Defensive match"),
        (2.5, 2.5, "High-scoring match"),
    ]

    print("\nCalculating P(Over 2.5) for different scenarios:\n")

    for lambda_h, lambda_a, description in scenarios:
        prob_over, prob_under = model.predict_over_under_analytical(
            lambda_h, lambda_a, threshold=2.5
        )

        print(f"{description}")
        print(f"  λ_home={lambda_h}, λ_away={lambda_a}")
        print(f"  Expected total: {lambda_h + lambda_a:.2f}")
        print(f"  P(Over 2.5): {prob_over:.3f}")
        print(f"  P(Under 2.5): {prob_under:.3f}")
        print()


def example_evaluation():
    """Example of model evaluation."""
    print("=" * 60)
    print("Model Evaluation Example")
    print("=" * 60)

    # Prepare data
    df, feature_cols = prepare_data()

    # Split
    train_df = df[df['date'] < '2024-08-01']
    test_df = df[df['date'] >= '2024-08-01']

    print(f"\nTrain: {len(train_df)} matches")
    print(f"Test: {len(test_df)} matches")

    # Train
    print("\nTraining model...")
    model = PoissonGoalsModel(threshold=2.5)
    model.fit(train_df, feature_cols)

    # Predict
    print("Making predictions...")
    y_true = test_df['over_2.5'].values
    y_pred = model.predict(test_df)
    y_proba = model.predict_proba(test_df)[:, 1]  # Probability of over

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.3f}")

    try:
        roc_auc = roc_auc_score(y_true, y_proba)
        print(f"ROC AUC: {roc_auc:.3f}")
    except:
        print("ROC AUC: Could not calculate (need both classes)")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Under 2.5', 'Over 2.5']))

    # Distribution of predictions
    print("\nPrediction distribution:")
    print(pd.Series(y_pred).value_counts().to_dict())

    print("\nActual distribution:")
    print(pd.Series(y_true).value_counts().to_dict())

    print()


def example_model_summary():
    """Example of getting model summary."""
    print("=" * 60)
    print("Model Summary Example")
    print("=" * 60)

    # Prepare data
    df, feature_cols = prepare_data()
    train_df = df[df['date'] < '2024-08-01']

    # Train
    model = PoissonGoalsModel(threshold=2.5)
    model.fit(train_df, feature_cols)

    # Get summary
    summary = model.get_model_summary()

    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    print(f"\nThreshold: {summary['threshold']}")
    print(f"Number of features: {summary['n_features']}")

    print("\nHome Goals Model:")
    print(f"  AIC: {summary['home_model']['aic']:.2f}")
    print(f"  BIC: {summary['home_model']['bic']:.2f}")
    print(f"  Deviance: {summary['home_model']['deviance']:.2f}")
    print(f"  Converged: {summary['home_model']['converged']}")

    print("\nAway Goals Model:")
    print(f"  AIC: {summary['away_model']['aic']:.2f}")
    print(f"  BIC: {summary['away_model']['bic']:.2f}")
    print(f"  Deviance: {summary['away_model']['deviance']:.2f}")
    print(f"  Converged: {summary['away_model']['converged']}")

    # Get coefficients
    print("\n" + "=" * 60)
    print("SIGNIFICANT FEATURES (p < 0.05)")
    print("=" * 60)

    coef_df = model.get_coefficients()
    significant = coef_df[
        (coef_df['home_significant']) | (coef_df['away_significant'])
    ].copy()

    print(f"\nFound {len(significant)} significant features:\n")
    for _, row in significant.iterrows():
        print(f"{row['feature']}")
        if row['home_significant']:
            print(f"  Home: coef={row['home_coef']:.4f}, p={row['home_p_value']:.4f}")
        if row['away_significant']:
            print(f"  Away: coef={row['away_coef']:.4f}, p={row['away_p_value']:.4f}")

    print()


def example_save_load():
    """Example of saving and loading model."""
    print("=" * 60)
    print("Save and Load Model Example")
    print("=" * 60)

    # Prepare data
    df, feature_cols = prepare_data()
    train_df = df[df['date'] < '2024-08-01']
    test_df = df[df['date'] >= '2024-08-01'].head(5)

    # Train model
    print("\nTraining model...")
    model = PoissonGoalsModel(threshold=2.5)
    model.fit(train_df, feature_cols)

    # Make predictions
    print("Making predictions with original model...")
    original_pred = model.predict_proba(test_df)

    # Save model
    from pathlib import Path
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / 'poisson_model_test.pkl'
    print(f"\nSaving model to {model_path}...")
    model.save(str(model_path))

    # Load model
    print(f"Loading model from {model_path}...")
    loaded_model = PoissonGoalsModel.load(str(model_path))

    # Make predictions with loaded model
    print("Making predictions with loaded model...")
    loaded_pred = loaded_model.predict_proba(test_df)

    # Verify predictions are identical
    print("\nVerifying predictions match...")
    if np.allclose(original_pred, loaded_pred):
        print("✓ Predictions match perfectly!")
    else:
        print("✗ Predictions do not match")

    print(f"\nOriginal predictions:\n{original_pred[:3]}")
    print(f"\nLoaded predictions:\n{loaded_pred[:3]}")

    print(f"\nModel file size: {model_path.stat().st_size / 1024:.1f} KB")

    print()


def main():
    """Run all examples."""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    logger.info("Starting Poisson model examples")

    print("\n" + "=" * 60)
    print("POISSON GOALS MODEL EXAMPLES")
    print("=" * 60 + "\n")

    try:
        example_basic_training()
        example_predictions()
        example_analytical_calculation()
        example_evaluation()
        example_model_summary()
        example_save_load()
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Examples failed: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()

    print("=" * 60)
    print("Examples completed!")
    print("Check logs/ directory for detailed logs")
    print("=" * 60)


if __name__ == "__main__":
    main()
