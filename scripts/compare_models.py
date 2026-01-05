"""
Script to compare different model versions and solutions.
Compares original Poisson, threshold-optimized, and balanced models.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import PoissonGoalsModel
from src.models.evaluator import ModelEvaluator
from src.utils import setup_logging, get_logger, Config


def evaluate_model_with_threshold(model, X, y_true, threshold=0.5, model_name="Model"):
    """
    Evaluate model with specific threshold.

    Args:
        model: Trained model
        X: Features
        y_true: True labels
        threshold: Decision threshold
        model_name: Name for logging

    Returns:
        Dictionary with metrics
    """
    logger = get_logger(__name__)

    # Get probabilities
    y_pred_proba = model.predict_proba(X)

    # Apply threshold
    y_pred = (y_pred_proba > threshold).astype(int)

    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, balanced_accuracy_score
    )

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Metrics
    metrics = {
        'model': model_name,
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'pred_over_rate': y_pred.mean(),
        'pred_under_rate': 1 - y_pred.mean()
    }

    return metrics


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """Find optimal threshold for given metric."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []

    from sklearn.metrics import (
        f1_score, balanced_accuracy_score, accuracy_score
    )

    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        scores.append(score)

    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]

    return optimal_threshold, optimal_score


def plot_comparison(results_df, output_dir):
    """
    Create comparison plots for different model configurations.

    Args:
        results_df: DataFrame with results from different models
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Set style
    sns.set_palette("husl")

    # 1. Key Metrics Comparison
    ax = axes[0, 0]
    metrics_to_plot = ['accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc']
    x = np.arange(len(results_df))
    width = 0.2

    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i * width, results_df[metric], width, label=metric.replace('_', ' ').title())

    ax.set_ylabel('Score')
    ax.set_title('Key Metrics Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    # 2. Precision vs Recall
    ax = axes[0, 1]
    ax.scatter(results_df['recall'], results_df['precision'], s=200, alpha=0.6)
    for idx, row in results_df.iterrows():
        ax.annotate(row['model'], (row['recall'], row['precision']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall Trade-off')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.3)

    # 3. Specificity vs Sensitivity
    ax = axes[1, 0]
    ax.scatter(results_df['recall'], results_df['specificity'], s=200, alpha=0.6)
    for idx, row in results_df.iterrows():
        ax.annotate(row['model'], (row['recall'], row['specificity']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Sensitivity (Recall)')
    ax.set_ylabel('Specificity')
    ax.set_title('Sensitivity vs Specificity Trade-off')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot([0, 1], [1, 0], 'r--', alpha=0.3, label='Random')
    ax.legend()

    # 4. Prediction Distribution
    ax = axes[1, 1]
    x = np.arange(len(results_df))
    width = 0.35
    ax.barh(x, results_df['pred_under_rate'], width, label='Predicted Under', alpha=0.8)
    ax.barh(x, results_df['pred_over_rate'], width, left=results_df['pred_under_rate'],
            label='Predicted Over', alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(results_df['model'])
    ax.set_xlabel('Proportion')
    ax.set_title('Prediction Distribution (Over vs Under)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Add actual distribution line
    if len(results_df) > 0:
        actual_over_rate = results_df['true_positives'].sum() + results_df['false_negatives'].sum()
        actual_over_rate /= (results_df['true_positives'].sum() + results_df['false_negatives'].sum() +
                             results_df['true_negatives'].sum() + results_df['false_positives'].sum())
        ax.axvline(x=actual_over_rate, color='red', linestyle='--',
                   label=f'Actual Over Rate: {actual_over_rate:.2%}', linewidth=2)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'models_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_dir / 'models_comparison.png'}")


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description='Compare different model versions and thresholds'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to original model (default: models/poisson_model_latest.pkl)'
    )

    parser.add_argument(
        '--balanced-model',
        type=str,
        default=None,
        help='Path to balanced model (default: models/poisson_model_balanced.pkl)'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to test data (default: uses 20%% holdout)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: models/comparison/)'
    )

    args = parser.parse_args()

    # Setup
    setup_logging()
    logger = get_logger(__name__)
    config = Config()

    # Paths
    model_path = args.model or config.models_path / 'poisson_model_latest.pkl'
    balanced_model_path = args.balanced_model or config.models_path / 'poisson_model_balanced.pkl'
    data_path = args.data or config.data_final_path / 'training_data_latest.parquet'
    output_dir = args.output or config.models_path / 'comparison'

    logger.info("="*80)
    logger.info("MODEL COMPARISON")
    logger.info("="*80)
    logger.info(f"\nOriginal model: {model_path}")
    logger.info(f"Balanced model: {balanced_model_path}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_dir}")

    try:
        # Load data
        logger.info("\nLoading data...")
        df = pd.read_parquet(data_path)

        # Use last 20% as test set
        train_size = int(0.8 * len(df))
        df_test = df.iloc[train_size:]

        feature_cols = [col for col in df.columns if col not in [
            'home_goals', 'away_goals', 'total_goals',
            'over_0.5', 'over_1.5', 'over_2.5', 'over_3.5', 'over_4.5',
            'home_team', 'away_team', 'date', 'season'
        ]]

        X_test = df_test[feature_cols]
        y_test = df_test['over_2.5'].values

        logger.info(f"Test set: {len(df_test)} matches")
        logger.info(f"Actual distribution: {y_test.mean():.1%} Over, {1-y_test.mean():.1%} Under")

        results = []

        # Load original model
        logger.info("\n" + "-"*80)
        logger.info("1. ORIGINAL MODEL (threshold=0.5)")
        logger.info("-"*80)

        model = PoissonGoalsModel.load(model_path)

        # Evaluate with default threshold
        metrics = evaluate_model_with_threshold(
            model, X_test, y_test,
            threshold=0.5,
            model_name="Original (t=0.5)"
        )
        results.append(metrics)

        logger.info(f"Accuracy:          {metrics['accuracy']:.4f}")
        logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"ROC AUC:           {metrics['roc_auc']:.4f}")
        logger.info(f"Specificity:       {metrics['specificity']:.4f}")
        logger.info(f"Sensitivity:       {metrics['recall']:.4f}")
        logger.info(f"Predictions:       {metrics['pred_over_rate']:.1%} Over")

        # Find optimal threshold
        logger.info("\n" + "-"*80)
        logger.info("2. ORIGINAL MODEL (optimized threshold)")
        logger.info("-"*80)

        y_pred_proba = model.predict_proba(X_test)
        optimal_threshold, optimal_score = find_optimal_threshold(
            y_test, y_pred_proba, metric='balanced_accuracy'
        )

        logger.info(f"Optimal threshold: {optimal_threshold:.3f}")

        metrics = evaluate_model_with_threshold(
            model, X_test, y_test,
            threshold=optimal_threshold,
            model_name=f"Original (t={optimal_threshold:.3f})"
        )
        results.append(metrics)

        logger.info(f"Accuracy:          {metrics['accuracy']:.4f}")
        logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"ROC AUC:           {metrics['roc_auc']:.4f}")
        logger.info(f"Specificity:       {metrics['specificity']:.4f}")
        logger.info(f"Sensitivity:       {metrics['recall']:.4f}")
        logger.info(f"Predictions:       {metrics['pred_over_rate']:.1%} Over")

        # Load balanced model if available
        if balanced_model_path.exists():
            logger.info("\n" + "-"*80)
            logger.info("3. BALANCED MODEL (threshold=0.5)")
            logger.info("-"*80)

            from src.models.poisson_model_balanced import BalancedPoissonGoalsModel
            balanced_model = BalancedPoissonGoalsModel.load(balanced_model_path)

            metrics = evaluate_model_with_threshold(
                balanced_model, X_test, y_test,
                threshold=0.5,
                model_name="Balanced (t=0.5)"
            )
            results.append(metrics)

            logger.info(f"Accuracy:          {metrics['accuracy']:.4f}")
            logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            logger.info(f"ROC AUC:           {metrics['roc_auc']:.4f}")
            logger.info(f"Specificity:       {metrics['specificity']:.4f}")
            logger.info(f"Sensitivity:       {metrics['recall']:.4f}")
            logger.info(f"Predictions:       {metrics['pred_over_rate']:.1%} Over")

        # Create comparison dataframe
        results_df = pd.DataFrame(results)

        # Print summary table
        logger.info("\n" + "="*80)
        logger.info("SUMMARY COMPARISON")
        logger.info("="*80 + "\n")

        print(results_df[['model', 'threshold', 'accuracy', 'balanced_accuracy',
                         'roc_auc', 'f1_score', 'specificity', 'recall']].to_string(index=False))

        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(output_dir / 'comparison_results.csv', index=False)
        logger.info(f"\nResults saved to: {output_dir / 'comparison_results.csv'}")

        # Create plots
        plot_comparison(results_df, output_dir)

        # Recommendations
        logger.info("\n" + "="*80)
        logger.info("RECOMMENDATIONS")
        logger.info("="*80)

        best_balanced = results_df.loc[results_df['balanced_accuracy'].idxmax()]
        best_roc = results_df.loc[results_df['roc_auc'].idxmax()]

        logger.info(f"\nBest Balanced Accuracy: {best_balanced['model']} ({best_balanced['balanced_accuracy']:.4f})")
        logger.info(f"Best ROC AUC: {best_roc['model']} ({best_roc['roc_auc']:.4f})")

        improvement = results_df['balanced_accuracy'].max() - results_df.iloc[0]['balanced_accuracy']
        logger.info(f"\nImprovement from original: +{improvement:.4f} ({improvement/results_df.iloc[0]['balanced_accuracy']*100:.1f}%)")

    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
