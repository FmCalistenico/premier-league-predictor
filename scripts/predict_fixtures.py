"""
Predict Future Fixtures with Confidence Scoring
================================================

Makes predictions on upcoming fixtures with reliability assessment.

Usage:
    python scripts/predict_fixtures.py --input data/predictions/prediction_data_GW23.parquet
    python scripts/predict_fixtures.py --input data/predictions/prediction_data_GW23.parquet --min-confidence 70
"""

import argparse
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, setup_logging

logger = get_logger(__name__)


class FixturePredictor:
    """Makes predictions with confidence scoring for future fixtures."""

    def __init__(self, model_path: str = "models/results/retrain_checkpoint.pkl",
                 optimized_threshold: float = None):
        """
        Args:
            model_path: Path to trained model
            optimized_threshold: Override model's threshold (e.g., 0.62 instead of 0.75)
        """
        self.model_path = Path(model_path)
        self.model = None
        self.optimized_threshold = optimized_threshold
        self.drift_scores = {}  # Will be populated if available

    def load_model(self):
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from: {self.model_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            checkpoint = pickle.load(f)

        # Extract model from checkpoint structure
        if isinstance(checkpoint, dict):
            if 'final' in checkpoint and 'best_model' in checkpoint['final']:
                model_ref = checkpoint['final']['best_model']
                # If it's a string, load the actual model from production folder
                if isinstance(model_ref, str):
                    prod_model = Path('models/production/best_model.pkl')
                    if prod_model.exists():
                        logger.info(f"Loading actual model from: {prod_model}")
                        with open(prod_model, 'rb') as f2:
                            self.model = pickle.load(f2)
                    else:
                        raise ValueError(f"Model reference '{model_ref}' found but actual model not found at {prod_model}")
                else:
                    self.model = model_ref
                logger.info("✓ Loaded model from checkpoint['final']['best_model']")
            elif 'model' in checkpoint:
                self.model = checkpoint['model']
                logger.info("✓ Loaded model from checkpoint['model']")
            else:
                logger.error(f"Checkpoint keys: {list(checkpoint.keys())}")
                raise ValueError("Could not find model in checkpoint")

            # Load drift scores if available
            if 'features' in checkpoint and 'drift_scores' in checkpoint.get('features', {}):
                self.drift_scores = checkpoint['features']['drift_scores']
                logger.info(f"✓ Loaded drift scores for {len(self.drift_scores)} features")

        else:
            # Checkpoint is the model itself
            self.model = checkpoint
            logger.info("✓ Loaded model directly")

        # Override threshold if specified
        if self.optimized_threshold is not None:
            original = getattr(self.model, 'threshold', 0.5)
            logger.info(f"Overriding threshold: {original} → {self.optimized_threshold}")
            self.model.threshold = self.optimized_threshold

        logger.info(f"Model type: {type(self.model).__name__}")
        logger.info(f"Threshold: {getattr(self.model, 'threshold', 'N/A')}")

        return self.model

    def predict(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on fixtures.

        Args:
            prediction_df: DataFrame with V2 features

        Returns:
            DataFrame with predictions and probabilities
        """
        if self.model is None:
            self.load_model()

        logger.info(f"Making predictions for {len(prediction_df)} fixtures")

        # Get feature columns
        feature_cols = [col for col in prediction_df.columns
                        if col not in ['fixture_id', 'date', 'season', 'home_team', 'away_team',
                                       'gameweek', 'home_goals', 'away_goals', 'total_goals']]

        logger.info(f"Using {len(feature_cols)} features")

        # Make predictions
        X = prediction_df[feature_cols].copy()

        # Get probabilities
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probabilities = self.model.predict_proba(X)

        # probabilities shape: (n_samples, 2) → [prob_under, prob_over]
        prob_under = probabilities[:, 0]
        prob_over = probabilities[:, 1]

        # Get binary predictions
        predictions = self.model.predict(X)  # 0 = Under, 1 = Over

        # Get expected goals (if model supports it)
        if hasattr(self.model, 'predict_expected_goals'):
            try:
                lambda_home, lambda_away = self.model.predict_expected_goals(X)
                expected_total_goals = lambda_home + lambda_away
            except Exception as e:
                logger.warning(f"Could not get expected goals: {e}")
                lambda_home = np.full(len(X), np.nan)
                lambda_away = np.full(len(X), np.nan)
                expected_total_goals = np.full(len(X), np.nan)
        else:
            lambda_home = np.full(len(X), np.nan)
            lambda_away = np.full(len(X), np.nan)
            expected_total_goals = np.full(len(X), np.nan)

        # Build results DataFrame
        results = prediction_df[['fixture_id', 'date', 'home_team', 'away_team', 'gameweek']].copy()

        results['prob_over'] = prob_over
        results['prob_under'] = prob_under
        results['prediction'] = predictions
        results['prediction_label'] = results['prediction'].map({0: 'Under 2.5', 1: 'Over 2.5'})
        results['expected_home_goals'] = lambda_home
        results['expected_away_goals'] = lambda_away
        results['expected_total_goals'] = expected_total_goals

        logger.info(f"✓ Predictions completed")
        logger.info(f"  Over 2.5: {(predictions == 1).sum()} ({100 * (predictions == 1).mean():.1f}%)")
        logger.info(f"  Under 2.5: {(predictions == 0).sum()} ({100 * (predictions == 0).mean():.1f}%)")

        return results

    def calculate_confidence(self, results: pd.DataFrame, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate confidence scores for each prediction.

        Args:
            results: DataFrame with predictions
            prediction_df: Original DataFrame with features

        Returns:
            DataFrame with added 'confidence' column
        """
        logger.info("Calculating confidence scores...")

        confidence_scores = []

        for i, (idx, row) in enumerate(results.iterrows()):
            confidence = 100.0

            # Factor 1: Probability extremeness (0-100%)
            # If prob_over = 0.95 or 0.05 → high confidence
            # If prob_over = 0.52 → low confidence
            prob = row['prob_over']
            extremeness = abs(prob - 0.5) * 2  # 0-1 scale
            confidence *= (0.5 + 0.5 * extremeness)  # Weight 50-100%

            # Factor 2: Expected goals alignment
            # If expected_total ~ 2.5, it's uncertain
            if not np.isnan(row['expected_total_goals']):
                distance_from_threshold = abs(row['expected_total_goals'] - 2.5)
                if distance_from_threshold < 0.3:
                    confidence *= 0.7  # -30% if very close to threshold
                elif distance_from_threshold > 1.0:
                    confidence *= 1.1  # +10% if far from threshold

            # Factor 3: Feature drift (if available)
            if self.drift_scores:
                # Get features for this prediction
                feature_cols = [col for col in prediction_df.columns
                                if col in self.drift_scores]

                if feature_cols:
                    avg_drift = np.mean([self.drift_scores[col] for col in feature_cols])

                    if avg_drift > 0.2:
                        confidence *= 0.6  # -40% for high drift
                    elif avg_drift > 0.15:
                        confidence *= 0.8  # -20% for medium drift

            # Factor 4: Special contexts (from features)
            # Get corresponding row from prediction_df using positional index
            pred_row = prediction_df.iloc[i]

            # Bonus for derbies (more predictable patterns)
            if 'is_derby' in pred_row and pred_row['is_derby'] == 1:
                confidence *= 1.05  # +5%

            # Penalty for high volatility
            if 'combined_volatility' in pred_row:
                volatility = pred_row['combined_volatility']
                if not np.isnan(volatility) and volatility > 1.0:
                    confidence *= 0.85  # -15%

            # Penalty for top6 clashes (more unpredictable)
            if 'is_top6_clash' in pred_row and pred_row['is_top6_clash'] == 1:
                confidence *= 0.95  # -5%

            # Factor 5: Team form consistency
            # If both teams have stable form → higher confidence
            if all(col in pred_row for col in ['home_goals_scored_volatility_L5', 'away_goals_scored_volatility_L5']):
                avg_team_volatility = (pred_row['home_goals_scored_volatility_L5'] +
                                        pred_row['away_goals_scored_volatility_L5']) / 2

                if not np.isnan(avg_team_volatility):
                    if avg_team_volatility < 0.5:
                        confidence *= 1.1  # +10% for consistent teams
                    elif avg_team_volatility > 1.5:
                        confidence *= 0.85  # -15% for inconsistent teams

            # Clamp to 0-100
            confidence = max(0, min(100, confidence))
            confidence_scores.append(confidence)

        results['confidence'] = confidence_scores

        # Add confidence tier
        results['confidence_tier'] = pd.cut(
            results['confidence'],
            bins=[0, 50, 65, 80, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        logger.info(f"✓ Confidence scores calculated")
        logger.info(f"  Mean confidence: {results['confidence'].mean():.1f}%")
        logger.info(f"  High/Very High: {(results['confidence'] >= 65).sum()} fixtures")

        return results

    def generate_recommendations(self, results: pd.DataFrame, min_confidence: float = 65) -> pd.DataFrame:
        """
        Generate betting recommendations based on predictions and confidence.

        Args:
            results: DataFrame with predictions and confidence
            min_confidence: Minimum confidence for recommending a bet

        Returns:
            DataFrame with 'recommendation' column
        """
        logger.info(f"Generating recommendations (min confidence: {min_confidence}%)")

        recommendations = []

        for _, row in results.iterrows():
            if row['confidence'] >= min_confidence:
                action = "BET"
                conf_label = row['confidence_tier']
                recommendation = f"{action}: {row['prediction_label']} ({conf_label} Confidence)"
            else:
                recommendation = "SKIP: Low Confidence"

            recommendations.append(recommendation)

        results['recommendation'] = recommendations

        bet_count = (results['recommendation'].str.startswith('BET')).sum()
        logger.info(f"✓ Recommendations: {bet_count} BET, {len(results) - bet_count} SKIP")

        return results

    def save_predictions(self, results: pd.DataFrame, output_path: str):
        """Save predictions to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results.to_csv(output_path, index=False)

        logger.info(f"✓ Saved predictions: {output_path}")

        return output_path


def main():
    parser = argparse.ArgumentParser(description="Predict future fixtures with confidence scoring")
    parser.add_argument('--input', type=str, required=True, help="Input prediction dataset (parquet/csv)")
    parser.add_argument('--model', type=str, default="models/results/retrain_checkpoint.pkl",
                        help="Path to trained model")
    parser.add_argument('--threshold', type=float, help="Override model threshold (e.g., 0.62)")
    parser.add_argument('--min-confidence', type=float, default=65,
                        help="Minimum confidence for BET recommendation")
    parser.add_argument('--output', type=str, help="Output path (default: auto-generated)")

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger.info("=" * 80)
    logger.info("PREDICT FUTURE FIXTURES")
    logger.info("=" * 80)

    # Load prediction dataset
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loading prediction dataset: {input_path}")

    if input_path.suffix == '.parquet':
        prediction_df = pd.read_parquet(input_path)
    else:
        prediction_df = pd.read_csv(input_path)

    logger.info(f"Loaded {len(prediction_df)} fixtures")

    # Create predictor
    predictor = FixturePredictor(
        model_path=args.model,
        optimized_threshold=args.threshold
    )

    # Make predictions
    results = predictor.predict(prediction_df)

    # Calculate confidence
    results = predictor.calculate_confidence(results, prediction_df)

    # Generate recommendations
    results = predictor.generate_recommendations(results, min_confidence=args.min_confidence)

    # Generate output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/predictions/predictions_{timestamp}.csv"

    # Save
    predictor.save_predictions(results, output_path)

    # Display summary
    logger.info("=" * 80)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 80)

    logger.info("\nTop confident predictions:")
    top = results.nlargest(5, 'confidence')[
        ['home_team', 'away_team', 'prediction_label', 'prob_over', 'confidence', 'recommendation']
    ]

    for _, row in top.iterrows():
        logger.info(f"  {row['home_team']:15s} vs {row['away_team']:15s} | "
                    f"{row['prediction_label']:12s} ({row['prob_over']:.1%}) | "
                    f"Conf: {row['confidence']:.0f}% | {row['recommendation']}")

    logger.info(f"\n✓ Complete predictions saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
