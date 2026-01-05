"""
Optimize Prediction Threshold for Production
=============================================

Finds optimal threshold to balance Sensitivity and Specificity.

Target: Sensitivity ≥ 55%, Specificity ≥ 60%, Pred Over Rate ≈ 50-55%

Usage:
    python scripts/optimize_threshold_production.py
    python scripts/optimize_threshold_production.py --metric f1
    python scripts/optimize_threshold_production.py --target-sensitivity 0.60
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, setup_logging

logger = get_logger(__name__)


class ThresholdOptimizer:
    """Optimizes prediction threshold for production use."""

    def __init__(self, model_path: str, data_path: str):
        """
        Args:
            model_path: Path to trained model
            data_path: Path to validation data
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = None
        self.X_val = None
        self.y_val = None
        self.probabilities = None
        self.prob_over = None

    def load_model_and_data(self):
        """Load model and validation data."""
        logger.info(f"Loading model: {self.model_path}")

        # Try loading directly first (for production models)
        try:
            with open(self.model_path, 'rb') as f:
                checkpoint = pickle.load(f)

            # Extract model
            if isinstance(checkpoint, dict):
                # Try checkpoint structure first
                if 'final' in checkpoint and 'best_model' in checkpoint['final']:
                    model_name = checkpoint['final']['best_model']
                    # If it's a string, try loading the actual model file
                    if isinstance(model_name, str):
                        # Look for the model in production folder
                        prod_model = Path('models/production/best_model.pkl')
                        if prod_model.exists():
                            logger.info(f"Loading actual model from: {prod_model}")
                            with open(prod_model, 'rb') as f2:
                                self.model = pickle.load(f2)
                        else:
                            raise ValueError(f"Model name '{model_name}' found but actual model file not found")
                    else:
                        self.model = model_name
                elif 'model' in checkpoint:
                    self.model = checkpoint['model']
                else:
                    raise ValueError("Could not find model in checkpoint")
            else:
                self.model = checkpoint
        except Exception as e:
            logger.error(f"Error loading from checkpoint: {e}")
            # Try loading production model directly
            prod_model = Path('models/production/best_model.pkl')
            if prod_model.exists():
                logger.info(f"Trying production model: {prod_model}")
                with open(prod_model, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                raise

        logger.info(f"✓ Model loaded: {type(self.model).__name__}")

        # Load validation data
        logger.info(f"Loading data: {self.data_path}")

        if self.data_path.suffix == '.parquet':
            df = pd.read_parquet(self.data_path)
        else:
            df = pd.read_csv(self.data_path)

        logger.info(f"✓ Loaded {len(df)} samples")

        # Split features and target
        target_col = 'over_2.5'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        feature_cols = [col for col in df.columns
                        if col not in ['fixture_id', 'date', 'season', 'home_team', 'away_team',
                                       'gameweek', 'home_goals', 'away_goals', 'total_goals',
                                       'over_0.5', 'over_1.5', 'over_2.5', 'over_3.5', 'over_4.5']]

        self.X_val = df[feature_cols]
        self.y_val = df[target_col].values

        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Target distribution: Over={100 * self.y_val.mean():.1f}%, Under={100 * (1-self.y_val.mean()):.1f}%")

        return self

    def get_probabilities(self):
        """Get prediction probabilities."""
        logger.info("Calculating probabilities...")

        self.probabilities = self.model.predict_proba(self.X_val)
        self.prob_over = self.probabilities[:, 1]  # Probability of Over 2.5

        logger.info(f"✓ Probabilities calculated")
        logger.info(f"  Mean prob(Over): {self.prob_over.mean():.3f}")
        logger.info(f"  Range: [{self.prob_over.min():.3f}, {self.prob_over.max():.3f}]")

        return self.prob_over

    def find_optimal_threshold(self, metric: str = 'balanced_accuracy',
                                target_sensitivity: float = 0.55,
                                target_specificity: float = 0.60):
        """
        Find optimal threshold.

        Args:
            metric: 'balanced_accuracy', 'f1', 'youden', or 'custom'
            target_sensitivity: Minimum sensitivity target
            target_specificity: Minimum specificity target

        Returns:
            Optimal threshold value
        """
        if self.prob_over is None:
            self.get_probabilities()

        logger.info(f"Finding optimal threshold (metric: {metric})")
        logger.info(f"  Target sensitivity: ≥{target_sensitivity:.1%}")
        logger.info(f"  Target specificity: ≥{target_specificity:.1%}")

        # Test range of thresholds
        thresholds = np.linspace(0.3, 0.8, 100)
        results = []

        for threshold in thresholds:
            y_pred = (self.prob_over >= threshold).astype(int)

            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred).ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall, TPR
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            bal_acc = balanced_accuracy_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred, zero_division=0)
            pred_over_rate = y_pred.mean()

            # Youden's J statistic
            youden_j = sensitivity + specificity - 1

            results.append({
                'threshold': threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'balanced_accuracy': bal_acc,
                'f1': f1,
                'youden_j': youden_j,
                'pred_over_rate': pred_over_rate,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            })

        results_df = pd.DataFrame(results)

        # Select best threshold based on metric
        if metric == 'balanced_accuracy':
            best_idx = results_df['balanced_accuracy'].idxmax()
            optimal_threshold = results_df.loc[best_idx, 'threshold']

        elif metric == 'f1':
            best_idx = results_df['f1'].idxmax()
            optimal_threshold = results_df.loc[best_idx, 'threshold']

        elif metric == 'youden':
            best_idx = results_df['youden_j'].idxmax()
            optimal_threshold = results_df.loc[best_idx, 'threshold']

        elif metric == 'custom':
            # Find threshold that meets both sensitivity and specificity targets
            # and is closest to 50% pred over rate
            viable = results_df[
                (results_df['sensitivity'] >= target_sensitivity) &
                (results_df['specificity'] >= target_specificity)
            ]

            if viable.empty:
                logger.warning("No threshold meets both targets, using balanced_accuracy")
                best_idx = results_df['balanced_accuracy'].idxmax()
                optimal_threshold = results_df.loc[best_idx, 'threshold']
            else:
                # From viable thresholds, pick closest to 50% pred over rate
                viable['distance_from_50'] = abs(viable['pred_over_rate'] - 0.50)
                best_idx = viable['distance_from_50'].idxmin()
                optimal_threshold = viable.loc[best_idx, 'threshold']

        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Get metrics at optimal threshold
        best_metrics = results_df.loc[best_idx]

        logger.info("=" * 60)
        logger.info("OPTIMAL THRESHOLD FOUND")
        logger.info("=" * 60)
        logger.info(f"Threshold: {optimal_threshold:.4f}")
        logger.info(f"")
        logger.info(f"Metrics:")
        logger.info(f"  Sensitivity (Recall):  {best_metrics['sensitivity']:.3f} ({best_metrics['sensitivity']:.1%})")
        logger.info(f"  Specificity (TNR):     {best_metrics['specificity']:.3f} ({best_metrics['specificity']:.1%})")
        logger.info(f"  Precision:             {best_metrics['precision']:.3f}")
        logger.info(f"  Balanced Accuracy:     {best_metrics['balanced_accuracy']:.3f}")
        logger.info(f"  F1 Score:              {best_metrics['f1']:.3f}")
        logger.info(f"  Youden's J:            {best_metrics['youden_j']:.3f}")
        logger.info(f"")
        logger.info(f"Prediction Distribution:")
        logger.info(f"  Pred Over 2.5:         {best_metrics['pred_over_rate']:.1%}")
        logger.info(f"  Pred Under 2.5:        {1 - best_metrics['pred_over_rate']:.1%}")
        logger.info(f"")
        logger.info(f"Confusion Matrix:")
        logger.info(f"  True Negatives (TN):   {int(best_metrics['tn'])}")
        logger.info(f"  False Positives (FP):  {int(best_metrics['fp'])}")
        logger.info(f"  False Negatives (FN):  {int(best_metrics['fn'])}")
        logger.info(f"  True Positives (TP):   {int(best_metrics['tp'])}")

        return optimal_threshold, results_df

    def plot_threshold_analysis(self, results_df: pd.DataFrame, optimal_threshold: float,
                                 output_dir: str = "models/results"):
        """Plot threshold optimization analysis."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Threshold Optimization Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Sensitivity vs Specificity
        ax1 = axes[0, 0]
        ax1.plot(results_df['threshold'], results_df['sensitivity'], label='Sensitivity', linewidth=2)
        ax1.plot(results_df['threshold'], results_df['specificity'], label='Specificity', linewidth=2)
        ax1.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal={optimal_threshold:.3f}', linewidth=2)
        ax1.axhline(0.55, color='gray', linestyle=':', alpha=0.5)
        ax1.axhline(0.60, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Sensitivity vs Specificity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Balanced Accuracy & F1
        ax2 = axes[0, 1]
        ax2.plot(results_df['threshold'], results_df['balanced_accuracy'], label='Balanced Accuracy', linewidth=2)
        ax2.plot(results_df['threshold'], results_df['f1'], label='F1 Score', linewidth=2)
        ax2.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Score')
        ax2.set_title('Balanced Accuracy & F1 Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Prediction Over Rate
        ax3 = axes[1, 0]
        ax3.plot(results_df['threshold'], results_df['pred_over_rate'] * 100, linewidth=2, color='purple')
        ax3.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2)
        ax3.axhline(50, color='green', linestyle=':', label='Target=50%', alpha=0.7)
        ax3.axhline(55, color='green', linestyle=':', alpha=0.5)
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Prediction Over Rate (%)')
        ax3.set_title('Predicted Over 2.5 Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Youden's J
        ax4 = axes[1, 1]
        ax4.plot(results_df['threshold'], results_df['youden_j'], linewidth=2, color='orange')
        ax4.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel("Youden's J (Sensitivity + Specificity - 1)")
        ax4.set_title("Youden's J Statistic")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = output_dir / 'threshold_optimization.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved plot: {plot_path}")

        plt.close()

    def save_optimized_model(self, optimal_threshold: float, output_path: str = None):
        """Save model with updated threshold."""
        if output_path is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"models/production_model_optimized_{timestamp}.pkl"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Update model threshold
        self.model.threshold = optimal_threshold

        # Save
        with open(output_path, 'wb') as f:
            pickle.dump(self.model, f)

        logger.info(f"✓ Saved optimized model: {output_path}")

        return output_path


def main():
    parser = argparse.ArgumentParser(description="Optimize prediction threshold for production")
    parser.add_argument('--model', type=str, default="models/results/retrain_checkpoint.pkl",
                        help="Path to trained model")
    parser.add_argument('--data', type=str, default="data/final/training_data_v2.parquet",
                        help="Path to validation data")
    parser.add_argument('--metric', type=str, default='custom',
                        choices=['balanced_accuracy', 'f1', 'youden', 'custom'],
                        help="Optimization metric")
    parser.add_argument('--target-sensitivity', type=float, default=0.55,
                        help="Minimum sensitivity target")
    parser.add_argument('--target-specificity', type=float, default=0.60,
                        help="Minimum specificity target")
    parser.add_argument('--save-model', action='store_true',
                        help="Save model with optimized threshold")
    parser.add_argument('--output', type=str, help="Output path for optimized model")

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger.info("=" * 80)
    logger.info("THRESHOLD OPTIMIZATION FOR PRODUCTION")
    logger.info("=" * 80)

    # Create optimizer
    optimizer = ThresholdOptimizer(
        model_path=args.model,
        data_path=args.data
    )

    # Load model and data
    optimizer.load_model_and_data()

    # Find optimal threshold
    optimal_threshold, results_df = optimizer.find_optimal_threshold(
        metric=args.metric,
        target_sensitivity=args.target_sensitivity,
        target_specificity=args.target_specificity
    )

    # Plot analysis
    optimizer.plot_threshold_analysis(results_df, optimal_threshold)

    # Save results
    results_df.to_csv('models/results/threshold_scan.csv', index=False)
    logger.info("✓ Saved threshold scan results: models/results/threshold_scan.csv")

    # Save optimized model if requested
    if args.save_model:
        model_path = optimizer.save_optimized_model(optimal_threshold, args.output)
        logger.info(f"\n✓ Use optimized model with:")
        logger.info(f"  python scripts/predict_fixtures.py --model {model_path}")

    logger.info("=" * 80)
    logger.info("✓ OPTIMIZATION COMPLETE")
    logger.info("=" * 80)

    return optimal_threshold


if __name__ == "__main__":
    main()
