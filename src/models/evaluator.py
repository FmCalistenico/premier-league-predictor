"""
Model evaluation module.
Provides comprehensive evaluation metrics and visualization tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    confusion_matrix,
    calibration_curve
)
from sklearn.model_selection import TimeSeriesSplit

from ..utils import LoggerMixin


class ModelEvaluator:
    """
    Static methods for model evaluation.

    Provides comprehensive metrics and visualizations for binary classification models.
    """

    @staticmethod
    def evaluate_classification_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.

        Args:
            y_true: True binary labels (0 or 1)
            y_pred_proba: Predicted probabilities for positive class

        Returns:
            Dictionary with metrics:
            - accuracy: Overall accuracy
            - log_loss: Logarithmic loss
            - brier_score: Brier score
            - roc_auc: ROC AUC score
            - precision: Precision for positive class
            - recall: Recall for positive class
            - f1_score: F1 score for positive class

        Example:
            >>> metrics = ModelEvaluator.evaluate_classification_metrics(y_true, y_proba)
            >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        """
        # Binary predictions (threshold 0.5)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'log_loss': log_loss(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
        }

        # ROC AUC (handle case where only one class present)
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics['roc_auc'] = np.nan

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average='binary',
            zero_division=0
        )

        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1

        return metrics

    @staticmethod
    def calculate_calibration_error(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        Uses sklearn's calibration_curve to bin predictions and calculate
        the mean absolute error between predicted and observed frequencies.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration curve (default: 10)

        Returns:
            Mean absolute calibration error

        Example:
            >>> ece = ModelEvaluator.calculate_calibration_error(y_true, y_proba)
            >>> print(f"Calibration Error: {ece:.3f}")
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true,
            y_pred_proba,
            n_bins=n_bins,
            strategy='uniform'
        )

        # Calculate mean absolute error
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

        return calibration_error

    @staticmethod
    def plot_calibration_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None,
        n_bins: int = 10,
        title: str = "Calibration Curve (Reliability Diagram)"
    ) -> None:
        """
        Plot calibration curve (reliability diagram).

        Shows how well predicted probabilities match observed frequencies.
        Perfect calibration would follow the diagonal line.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot (optional)
            n_bins: Number of bins (default: 10)
            title: Plot title

        Example:
            >>> ModelEvaluator.plot_calibration_curve(
            ...     y_true, y_proba,
            ...     save_path='plots/calibration.png'
            ... )
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true,
            y_pred_proba,
            n_bins=n_bins,
            strategy='uniform'
        )

        # Calculate calibration error
        ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot calibration curve
        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            marker='o',
            linewidth=2,
            label=f'Model (ECE={ece:.3f})'
        )

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration curve saved to {save_path}")

        plt.close()

    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "ROC Curve"
    ) -> None:
        """
        Plot ROC curve with AUC score.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot (optional)
            title: Plot title

        Example:
            >>> ModelEvaluator.plot_roc_curve(
            ...     y_true, y_proba,
            ...     save_path='plots/roc_curve.png'
            ... )
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot ROC curve
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f'ROC Curve (AUC = {roc_auc:.3f})'
        )

        # Random classifier line
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")

        plt.close()

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        labels: List[str] = ['Under 2.5', 'Over 2.5'],
        title: str = "Confusion Matrix"
    ) -> None:
        """
        Plot confusion matrix with seaborn heatmap.

        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            save_path: Path to save plot (optional)
            labels: Class labels for display
            title: Plot title

        Example:
            >>> y_pred = (y_proba >= 0.5).astype(int)
            >>> ModelEvaluator.plot_confusion_matrix(
            ...     y_true, y_pred,
            ...     save_path='plots/confusion_matrix.png'
            ... )
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'},
            ax=ax
        )

        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.close()

    @staticmethod
    def evaluate_model(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        plot: bool = True,
        save_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation with metrics and plots.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            plot: Whether to generate plots (default: True)
            save_dir: Directory to save plots (optional)

        Returns:
            Dictionary with all evaluation metrics

        Example:
            >>> metrics = ModelEvaluator.evaluate_model(
            ...     y_true, y_proba,
            ...     plot=True,
            ...     save_dir='evaluation_plots'
            ... )
            >>> print(f"ROC AUC: {metrics['roc_auc']:.3f}")
        """
        print("=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        # Calculate metrics
        metrics = ModelEvaluator.evaluate_classification_metrics(y_true, y_pred_proba)

        # Add calibration error
        metrics['calibration_error'] = ModelEvaluator.calculate_calibration_error(
            y_true, y_pred_proba
        )

        # Print metrics
        print("\nClassification Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Precision:          {metrics['precision']:.4f}")
        print(f"  Recall:             {metrics['recall']:.4f}")
        print(f"  F1 Score:           {metrics['f1_score']:.4f}")
        print(f"\nProbabilistic Metrics:")
        print(f"  ROC AUC:            {metrics['roc_auc']:.4f}")
        print(f"  Log Loss:           {metrics['log_loss']:.4f}")
        print(f"  Brier Score:        {metrics['brier_score']:.4f}")
        print(f"  Calibration Error:  {metrics['calibration_error']:.4f}")

        # Generate plots
        if plot:
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = Path('.')

            print("\nGenerating evaluation plots...")

            # Binary predictions
            y_pred = (y_pred_proba >= 0.5).astype(int)

            # Calibration curve
            ModelEvaluator.plot_calibration_curve(
                y_true,
                y_pred_proba,
                save_path=str(save_dir / 'calibration_curve.png')
            )

            # ROC curve
            ModelEvaluator.plot_roc_curve(
                y_true,
                y_pred_proba,
                save_path=str(save_dir / 'roc_curve.png')
            )

            # Confusion matrix
            ModelEvaluator.plot_confusion_matrix(
                y_true,
                y_pred,
                save_path=str(save_dir / 'confusion_matrix.png')
            )

            print(f"Plots saved to {save_dir}/")

        print("\n" + "=" * 60)

        return metrics


class TimeSeriesValidator(LoggerMixin):
    """
    Time series cross-validation for temporal data.

    CRITICAL: Uses TimeSeriesSplit to respect temporal order.
    NEVER use KFold or random splits for time series data.
    """

    def __init__(self):
        """Initialize time series validator."""
        self.logger.info("TimeSeriesValidator initialized")

    def validate(
        self,
        model,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'over_2.5',
        n_splits: int = 5
    ) -> pd.DataFrame:
        """
        Perform time series cross-validation.

        Uses TimeSeriesSplit to create train/test splits that respect
        temporal order. Each fold uses all previous data for training
        and next chunk for testing.

        Args:
            model: Model instance with fit() and predict_proba() methods
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name (default: 'over_2.5')
            n_splits: Number of splits (default: 5)

        Returns:
            DataFrame with metrics for each fold and summary statistics

        Example:
            >>> validator = TimeSeriesValidator()
            >>> results = validator.validate(
            ...     model, df, feature_cols,
            ...     n_splits=5
            ... )
            >>> print(results)
        """
        self.logger.info(f"Starting time series validation with {n_splits} splits")
        self.logger.info(f"Dataset: {len(df)} samples")

        # Initialize time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Store results for each fold
        fold_results = []

        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
            self.logger.info(f"\nFold {fold}/{n_splits}")

            # Split data
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            self.logger.info(f"  Train: {len(train_df)} samples")
            self.logger.info(f"  Test:  {len(test_df)} samples")

            # Train model
            self.logger.debug("  Training model...")
            model.fit(train_df, feature_cols)

            # Predict
            self.logger.debug("  Making predictions...")
            y_true = test_df[target_col].values
            y_pred_proba = model.predict_proba(test_df)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            # Calculate metrics
            metrics = ModelEvaluator.evaluate_classification_metrics(y_true, y_pred_proba)
            metrics['calibration_error'] = ModelEvaluator.calculate_calibration_error(
                y_true, y_pred_proba
            )

            # Add fold info
            metrics['fold'] = fold
            metrics['train_samples'] = len(train_df)
            metrics['test_samples'] = len(test_df)

            fold_results.append(metrics)

            # Log fold results
            self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"  ROC AUC:  {metrics['roc_auc']:.4f}")

        # Convert to DataFrame
        results_df = pd.DataFrame(fold_results)

        # Calculate summary statistics
        self.logger.info("\n" + "=" * 60)
        self.logger.info("CROSS-VALIDATION SUMMARY")
        self.logger.info("=" * 60)

        metric_cols = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'roc_auc', 'log_loss', 'brier_score', 'calibration_error'
        ]

        summary_rows = []

        for metric in metric_cols:
            values = results_df[metric].values
            summary_rows.append({
                'metric': metric,
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max()
            })

            self.logger.info(
                f"{metric:20s}: {values.mean():.4f} ± {values.std():.4f} "
                f"(min: {values.min():.4f}, max: {values.max():.4f})"
            )

        summary_df = pd.DataFrame(summary_rows)

        # Combine results and summary
        results_df = pd.concat([
            results_df,
            pd.DataFrame([{'fold': 'mean'}]).assign(**summary_df.set_index('metric')['mean'].to_dict()),
            pd.DataFrame([{'fold': 'std'}]).assign(**summary_df.set_index('metric')['std'].to_dict())
        ], ignore_index=True)

        self.logger.info("\n" + "=" * 60)

        return results_df

    def validate_with_plots(
        self,
        model,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'over_2.5',
        n_splits: int = 5,
        save_dir: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform validation with plot generation for final fold.

        Args:
            model: Model instance
            df: DataFrame with features and target
            feature_cols: Feature column names
            target_col: Target column name
            n_splits: Number of splits
            save_dir: Directory to save plots

        Returns:
            Tuple of (results_df, final_fold_metrics)
        """
        self.logger.info("Starting validation with plot generation")

        # Run cross-validation
        results_df = self.validate(model, df, feature_cols, target_col, n_splits)

        # Get final fold for visualization
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(df))
        train_idx, test_idx = splits[-1]

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Train on final fold
        self.logger.info("\nTraining final model for visualization...")
        model.fit(train_df, feature_cols)

        # Predict on final test set
        y_true = test_df[target_col].values
        y_pred_proba = model.predict_proba(test_df)[:, 1]

        # Evaluate with plots
        final_metrics = ModelEvaluator.evaluate_model(
            y_true,
            y_pred_proba,
            plot=True,
            save_dir=save_dir
        )

        return results_df, final_metrics
