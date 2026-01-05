"""
Backtest Prediction System
===========================

Validates the prediction system by simulating predictions on past gameweeks.

Usage:
    python scripts/backtest_predictions.py --start-gameweek 18 --end-gameweek 22
    python scripts/backtest_predictions.py --start-gameweek 18 --end-gameweek 22 --min-confidence 70
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, setup_logging
from build_prediction_dataset import PredictionDatasetBuilder
from predict_fixtures import FixturePredictor

logger = get_logger(__name__)


class PredictionBacktester:
    """Backtests prediction system on historical gameweeks."""

    def __init__(self, historical_data_path: str, model_path: str,
                 threshold: float = None, min_confidence: float = 65):
        self.historical_path = historical_data_path
        self.model_path = model_path
        self.threshold = threshold
        self.min_confidence = min_confidence

        self.dataset_builder = PredictionDatasetBuilder(historical_data_path)
        self.predictor = FixturePredictor(model_path, threshold)

    def run_backtest(self, start_gameweek: int, end_gameweek: int) -> pd.DataFrame:
        """
        Run backtest over range of gameweeks.

        Args:
            start_gameweek: First gameweek to test
            end_gameweek: Last gameweek to test

        Returns:
            DataFrame with backtest results per gameweek
        """
        logger.info(f"=" * 80)
        logger.info(f"BACKTEST: Gameweeks {start_gameweek} to {end_gameweek}")
        logger.info(f"=" * 80)

        # Load historical data
        df_historical = self.dataset_builder.load_historical_data()

        results = []

        for gw in range(start_gameweek, end_gameweek + 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"GAMEWEEK {gw}")
            logger.info(f"{'=' * 60}")

            # Get fixtures for this gameweek
            gw_fixtures = df_historical[df_historical['gameweek'] == gw].copy()

            if gw_fixtures.empty:
                logger.warning(f"No fixtures found for GW {gw}")
                continue

            logger.info(f"Found {len(gw_fixtures)} fixtures")

            # Build prediction dataset (using only data BEFORE this gameweek)
            historical_before_gw = df_historical[df_historical['gameweek'] < gw]

            # Temporarily replace historical data
            original_hist = self.dataset_builder.historical_df
            self.dataset_builder.historical_df = historical_before_gw

            try:
                # Build features for GW fixtures
                prediction_df = self.dataset_builder.build_upcoming_fixtures(
                    gw_fixtures[['date', 'home_team', 'away_team', 'gameweek']]
                )

                # Make predictions
                predictions = self.predictor.predict(prediction_df)
                predictions = self.predictor.calculate_confidence(predictions, prediction_df)
                predictions = self.predictor.generate_recommendations(predictions, self.min_confidence)

                # Get actual results
                # Calculate over_2.5 if not present
                if 'over_2.5' not in gw_fixtures.columns:
                    if 'total_goals' in gw_fixtures.columns:
                        gw_fixtures['over_2.5'] = (gw_fixtures['total_goals'] > 2.5).astype(int)
                    elif 'home_goals' in gw_fixtures.columns and 'away_goals' in gw_fixtures.columns:
                        gw_fixtures['over_2.5'] = (gw_fixtures['home_goals'] + gw_fixtures['away_goals'] > 2.5).astype(int)
                    else:
                        logger.error(f"Cannot calculate over_2.5 for GW {gw} - missing goals columns")
                        continue

                actual_over = gw_fixtures['over_2.5'].values

                # Evaluate
                gw_result = self._evaluate_gameweek(predictions, actual_over, gw)
                results.append(gw_result)

            finally:
                # Restore original historical data
                self.dataset_builder.historical_df = original_hist

        results_df = pd.DataFrame(results)

        # Summary statistics
        logger.info(f"\n{'=' * 80}")
        logger.info(f"BACKTEST SUMMARY")
        logger.info(f"{'=' * 80}")
        logger.info(f"Gameweeks tested: {len(results_df)}")
        logger.info(f"Total fixtures: {results_df['total_fixtures'].sum()}")
        logger.info(f"Predictions made: {results_df['predictions_made'].sum()}")
        logger.info(f"Correct predictions: {results_df['correct'].sum()}")
        logger.info(f"Overall accuracy: {results_df['correct'].sum() / results_df['predictions_made'].sum():.1%}")
        logger.info(f"Average confidence: {results_df['avg_confidence'].mean():.1f}%")
        logger.info(f"Total ROI: {results_df['roi'].sum():.1f} units")

        return results_df

    def _evaluate_gameweek(self, predictions: pd.DataFrame, actual_over: np.ndarray, gw: int) -> dict:
        """Evaluate predictions for one gameweek."""
        # Filter to recommendations that were BET
        bet_mask = predictions['recommendation'].str.startswith('BET')
        bet_predictions = predictions[bet_mask]

        total_fixtures = len(predictions)
        predictions_made = len(bet_predictions)

        if predictions_made == 0:
            logger.warning(f"GW {gw}: No predictions with sufficient confidence")
            return {
                'gameweek': gw,
                'total_fixtures': total_fixtures,
                'predictions_made': 0,
                'correct': 0,
                'accuracy': 0.0,
                'roi': 0.0,
                'avg_confidence': 0.0,
                'calibration_error': np.nan
            }

        # Get predicted and actual
        pred_over = bet_predictions['prediction'].values
        actual = actual_over[bet_mask]

        # Accuracy
        correct = (pred_over == actual).sum()
        accuracy = correct / predictions_made

        # ROI (simple: +1 unit per win, -1 per loss)
        roi = 2 * correct - predictions_made  # Net profit/loss

        # Average confidence
        avg_confidence = bet_predictions['confidence'].mean()

        # Calibration error
        prob_predicted = bet_predictions['prob_over'].values
        prob_actual = actual.mean()
        calibration_error = abs(prob_predicted.mean() - prob_actual)

        logger.info(f"GW {gw} Results:")
        logger.info(f"  Predictions: {predictions_made}/{total_fixtures}")
        logger.info(f"  Accuracy: {accuracy:.1%} ({correct}/{predictions_made})")
        logger.info(f"  ROI: {roi:+.1f} units")
        logger.info(f"  Avg Confidence: {avg_confidence:.1f}%")

        return {
            'gameweek': gw,
            'total_fixtures': total_fixtures,
            'predictions_made': predictions_made,
            'correct': correct,
            'accuracy': accuracy,
            'roi': roi,
            'avg_confidence': avg_confidence,
            'calibration_error': calibration_error
        }

    def save_backtest_results(self, results_df: pd.DataFrame, output_path: str = None):
        """Save backtest results."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"models/results/backtest_report_{timestamp}.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(output_path, index=False)

        logger.info(f"✓ Saved backtest results: {output_path}")

        return output_path


def main():
    parser = argparse.ArgumentParser(description="Backtest prediction system")
    parser.add_argument('--start-gameweek', type=int, required=True, help="First gameweek to test")
    parser.add_argument('--end-gameweek', type=int, required=True, help="Last gameweek to test")
    parser.add_argument('--historical', type=str, default="data/final/training_data_v2.parquet",
                        help="Path to historical data")
    parser.add_argument('--model', type=str, default="models/results/retrain_checkpoint.pkl",
                        help="Path to trained model")
    parser.add_argument('--threshold', type=float, help="Override model threshold")
    parser.add_argument('--min-confidence', type=float, default=65,
                        help="Minimum confidence for BET recommendation")
    parser.add_argument('--output', type=str, help="Output path for results")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Create backtester
    backtester = PredictionBacktester(
        historical_data_path=args.historical,
        model_path=args.model,
        threshold=args.threshold,
        min_confidence=args.min_confidence
    )

    # Run backtest
    results_df = backtester.run_backtest(
        start_gameweek=args.start_gameweek,
        end_gameweek=args.end_gameweek
    )

    # Save
    backtester.save_backtest_results(results_df, args.output)

    return results_df


if __name__ == "__main__":
    main()
