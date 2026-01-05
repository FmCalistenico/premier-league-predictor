"""
Script to find optimal prediction threshold for Poisson model.
Analyzes threshold impact on metrics and finds optimal value.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils import setup_logging, get_logger
from src.models import PoissonGoalsModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Find optimal threshold by testing range of values.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'accuracy', 'balanced_accuracy')

    Returns:
        Tuple of (optimal_threshold, best_metric_value, results_df)
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    results = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Balanced accuracy (mean of sensitivity and specificity)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (sensitivity + specificity) / 2

        results.append({
            'threshold': threshold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        })

    results_df = pd.DataFrame(results)

    # Find optimal based on metric
    if metric == 'f1':
        best_idx = results_df['f1_score'].idxmax()
    elif metric == 'balanced_accuracy':
        best_idx = results_df['balanced_accuracy'].idxmax()
    else:  # accuracy
        best_idx = results_df['accuracy'].idxmax()

    optimal_threshold = results_df.loc[best_idx, 'threshold']
    best_metric_value = results_df.loc[best_idx, metric]

    return optimal_threshold, best_metric_value, results_df


def plot_threshold_analysis(results_df, optimal_threshold, save_path=None):
    """Plot threshold analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Main metrics vs threshold
    ax1 = axes[0, 0]
    ax1.plot(results_df['threshold'], results_df['accuracy'], 'b-', label='Accuracy', linewidth=2)
    ax1.plot(results_df['threshold'], results_df['f1_score'], 'r-', label='F1 Score', linewidth=2)
    ax1.plot(results_df['threshold'], results_df['balanced_accuracy'], 'g-',
             label='Balanced Accuracy', linewidth=2)
    ax1.axvline(optimal_threshold, color='black', linestyle='--',
                label=f'Optimal={optimal_threshold:.2f}')
    ax1.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Default=0.5')
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Metric Value', fontsize=12)
    ax1.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Precision vs Recall
    ax2 = axes[0, 1]
    ax2.plot(results_df['threshold'], results_df['precision'], 'b-',
             label='Precision', linewidth=2)
    ax2.plot(results_df['threshold'], results_df['recall'], 'r-',
             label='Recall', linewidth=2)
    ax2.axvline(optimal_threshold, color='black', linestyle='--',
                label=f'Optimal={optimal_threshold:.2f}')
    ax2.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Default=0.5')
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Metric Value', fontsize=12)
    ax2.set_title('Precision & Recall vs Threshold', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Sensitivity vs Specificity
    ax3 = axes[1, 0]
    ax3.plot(results_df['threshold'], results_df['sensitivity'], 'b-',
             label='Sensitivity (Recall)', linewidth=2)
    ax3.plot(results_df['threshold'], results_df['specificity'], 'r-',
             label='Specificity', linewidth=2)
    ax3.plot(results_df['threshold'], results_df['balanced_accuracy'], 'g-',
             label='Balanced Accuracy', linewidth=2)
    ax3.axvline(optimal_threshold, color='black', linestyle='--',
                label=f'Optimal={optimal_threshold:.2f}')
    ax3.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Default=0.5')
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('Metric Value', fontsize=12)
    ax3.set_title('Sensitivity & Specificity vs Threshold', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Confusion matrix counts
    ax4 = axes[1, 1]
    ax4.plot(results_df['threshold'], results_df['tp'], 'g-', label='True Positives', linewidth=2)
    ax4.plot(results_df['threshold'], results_df['tn'], 'b-', label='True Negatives', linewidth=2)
    ax4.plot(results_df['threshold'], results_df['fp'], 'r--', label='False Positives', linewidth=1)
    ax4.plot(results_df['threshold'], results_df['fn'], 'orange', linestyle='--',
             label='False Negatives', linewidth=1)
    ax4.axvline(optimal_threshold, color='black', linestyle='--',
                label=f'Optimal={optimal_threshold:.2f}')
    ax4.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Default=0.5')
    ax4.set_xlabel('Threshold', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Confusion Matrix Components vs Threshold', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def analyze_probability_distribution(y_true, y_pred_proba, save_path=None):
    """Analyze distribution of predicted probabilities."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Overall distribution
    ax1 = axes[0, 0]
    ax1.hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(0.5, color='red', linestyle='--', label='Threshold=0.5')
    ax1.axvline(y_pred_proba.mean(), color='green', linestyle='--',
                label=f'Mean={y_pred_proba.mean():.3f}')
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distribution by class
    ax2 = axes[0, 1]
    under_probs = y_pred_proba[y_true == 0]
    over_probs = y_pred_proba[y_true == 1]

    ax2.hist(under_probs, bins=30, alpha=0.5, label='Under 2.5 (y=0)', color='blue', edgecolor='black')
    ax2.hist(over_probs, bins=30, alpha=0.5, label='Over 2.5 (y=1)', color='red', edgecolor='black')
    ax2.axvline(0.5, color='black', linestyle='--', label='Threshold=0.5')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Probability Distribution by Actual Class', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Box plots by class
    ax3 = axes[1, 0]
    ax3.boxplot([under_probs, over_probs], labels=['Under 2.5', 'Over 2.5'])
    ax3.axhline(0.5, color='red', linestyle='--', label='Threshold=0.5')
    ax3.set_ylabel('Predicted Probability', fontsize=12)
    ax3.set_title('Probability Distribution by Class (Boxplot)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
PROBABILITY DISTRIBUTION STATISTICS

Overall:
  Mean:   {y_pred_proba.mean():.4f}
  Median: {np.median(y_pred_proba):.4f}
  Std:    {y_pred_proba.std():.4f}
  Min:    {y_pred_proba.min():.4f}
  Max:    {y_pred_proba.max():.4f}

Under 2.5 (y=0):
  Mean:   {under_probs.mean():.4f}
  Median: {np.median(under_probs):.4f}
  Std:    {under_probs.std():.4f}
  Count:  {len(under_probs)}

Over 2.5 (y=1):
  Mean:   {over_probs.mean():.4f}
  Median: {np.median(over_probs):.4f}
  Std:    {over_probs.std():.4f}
  Count:  {len(over_probs)}

Separation:
  Mean diff: {over_probs.mean() - under_probs.mean():.4f}
  Overlap:   {((under_probs > 0.5).sum() + (over_probs < 0.5).sum()) / len(y_true) * 100:.1f}%
    """

    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def main():
    """Run threshold optimization analysis."""
    setup_logging()
    logger = get_logger(__name__)

    print("\n" + "=" * 80)
    print(" " * 25 + "THRESHOLD OPTIMIZATION ANALYSIS")
    print("=" * 80 + "\n")

    # Load model
    model_path = Path('models/poisson_model_latest.pkl')
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("   Run training pipeline first: python run_complete_pipeline.py")
        return

    print(f"Loading model from {model_path}")
    model = PoissonGoalsModel.load(str(model_path))

    # Load data
    data_path = Path('data/final/training_data_latest.parquet')
    if not data_path.exists():
        print(f"❌ Data not found at {data_path}")
        return

    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # Get predictions
    print("\nGenerating predictions...")
    y_true = df['over_2.5'].values
    y_pred_proba = model.predict_proba(df)[:, 1]

    print(f"\nDataset: {len(df)} matches")
    print(f"  Under 2.5: {(y_true == 0).sum()} ({(y_true == 0).mean() * 100:.1f}%)")
    print(f"  Over 2.5:  {(y_true == 1).sum()} ({(y_true == 1).mean() * 100:.1f}%)")

    # Analyze probability distribution
    print("\n" + "-" * 80)
    print("PROBABILITY DISTRIBUTION ANALYSIS")
    print("-" * 80)

    plots_dir = Path('models/plots')
    plots_dir.mkdir(exist_ok=True)

    analyze_probability_distribution(
        y_true, y_pred_proba,
        save_path=str(plots_dir / 'probability_distribution.png')
    )

    # Find optimal thresholds for different metrics
    print("\n" + "-" * 80)
    print("FINDING OPTIMAL THRESHOLDS")
    print("-" * 80)

    metrics_to_optimize = ['f1', 'balanced_accuracy', 'accuracy']

    for metric in metrics_to_optimize:
        optimal_threshold, best_value, results_df = find_optimal_threshold(
            y_true, y_pred_proba, metric=metric
        )

        print(f"\n✓ Optimal threshold for {metric}: {optimal_threshold:.3f}")
        print(f"  {metric}: {best_value:.4f}")

        # Get metrics at optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()

        print(f"\n  Confusion Matrix:")
        print(f"    TN: {tn:4d}  FP: {fp:4d}")
        print(f"    FN: {fn:4d}  TP: {tp:4d}")

        print(f"\n  Metrics:")
        print(f"    Accuracy:  {accuracy_score(y_true, y_pred_optimal):.4f}")
        print(f"    Precision: {precision_score(y_true, y_pred_optimal, zero_division=0):.4f}")
        print(f"    Recall:    {recall_score(y_true, y_pred_optimal, zero_division=0):.4f}")
        print(f"    F1 Score:  {f1_score(y_true, y_pred_optimal, zero_division=0):.4f}")

    # Plot threshold analysis for balanced accuracy
    print("\n" + "-" * 80)
    print("GENERATING THRESHOLD ANALYSIS PLOT")
    print("-" * 80)

    optimal_threshold, _, results_df = find_optimal_threshold(
        y_true, y_pred_proba, metric='balanced_accuracy'
    )

    plot_threshold_analysis(
        results_df, optimal_threshold,
        save_path=str(plots_dir / 'threshold_analysis.png')
    )

    # Compare default vs optimal threshold
    print("\n" + "=" * 80)
    print("COMPARISON: DEFAULT (0.5) VS OPTIMAL THRESHOLD")
    print("=" * 80)

    y_pred_default = (y_pred_proba >= 0.5).astype(int)
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

    print(f"\nDefault Threshold (0.5):")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_default).ravel()
    print(f"  Confusion Matrix:")
    print(f"    TN: {tn:4d}  FP: {fp:4d}")
    print(f"    FN: {fn:4d}  TP: {tp:4d}")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred_default):.4f}")
    print(f"  F1 Score:  {f1_score(y_true, y_pred_default, zero_division=0):.4f}")
    print(f"  Balanced Accuracy: {((tp/(tp+fn)) + (tn/(tn+fp)))/2:.4f}")

    print(f"\nOptimal Threshold ({optimal_threshold:.3f}):")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()
    print(f"  Confusion Matrix:")
    print(f"    TN: {tn:4d}  FP: {fp:4d}")
    print(f"    FN: {fn:4d}  TP: {tp:4d}")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred_optimal):.4f}")
    print(f"  F1 Score:  {f1_score(y_true, y_pred_optimal, zero_division=0):.4f}")
    print(f"  Balanced Accuracy: {((tp/(tp+fn)) + (tn/(tn+fp)))/2:.4f}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print(f"""
1. Use threshold = {optimal_threshold:.3f} instead of 0.5 for balanced predictions
2. Your model has {(y_pred_proba > 0.5).mean() * 100:.1f}% predictions > 0.5 (biased toward Over)
3. Mean predicted probability: {y_pred_proba.mean():.3f}

Next steps:
- Review feature importance (expected_total_goals, combined_over_rate)
- Consider retraining with class weights
- Check if Poisson assumptions hold (overdispersion?)
    """)

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
