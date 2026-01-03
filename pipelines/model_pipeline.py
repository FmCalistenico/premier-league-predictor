"""
Complete model training pipeline for Premier League prediction.
Orchestrates data loading, training, evaluation, and model saving.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Literal
from datetime import datetime

from src.utils import LoggerMixin, Config
from src.features import FeatureEngineer
from src.models import PoissonGoalsModel, ModelEvaluator, TimeSeriesValidator


class ModelPipeline(LoggerMixin):
    """
    Complete model training pipeline orchestration.

    Pipeline stages:
    1. Load: Load prepared training data
    2. Prepare: Extract features and target
    3. Train: Train model
    4. Evaluate: Comprehensive evaluation
    5. Cross-Validate: Time series validation
    6. Save: Save model and metadata

    Example:
        >>> pipeline = ModelPipeline()
        >>> model, metrics, cv_results = pipeline.run_full_training()
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize model training pipeline.

        Args:
            config: Configuration instance (optional)
        """
        self.config = config or Config()

        # Setup directories
        self.data_dir = self.config.data_final_path
        self.models_dir = self.config.models_path
        self.plots_dir = self.models_dir / 'plots'
        self.results_dir = self.models_dir / 'results'

        for directory in [self.models_dir, self.plots_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.logger.info("ModelPipeline initialized")
        self.logger.info(f"Data directory: {self.data_dir}")
        self.logger.info(f"Models directory: {self.models_dir}")

    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load training data.

        Args:
            filepath: Path to data file (default: training_data_latest.parquet)

        Returns:
            DataFrame with features and target

        Example:
            >>> df = pipeline.load_data()
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: LOAD DATA")
        self.logger.info("=" * 60)

        try:
            if filepath is None:
                filepath = self.data_dir / 'training_data_latest.parquet'
            else:
                filepath = Path(filepath)

            self.logger.info(f"Loading data from {filepath}")

            if not filepath.exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")

            # Load data
            if filepath.suffix == '.parquet':
                df = pd.read_parquet(filepath)
            elif filepath.suffix == '.csv':
                df = pd.read_csv(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")

            self.logger.info(f"Loaded {len(df)} matches")
            self.logger.info(f"Columns: {len(df.columns)}")

            # Check for required columns
            required_cols = ['over_2.5', 'date']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            self.logger.info("✓ Data loaded successfully")
            return df

        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}", exc_info=True)
            raise

    def prepare_features_and_target(
        self,
        df: pd.DataFrame,
        target_col: str = 'over_2.5'
    ) -> Tuple[pd.DataFrame, np.ndarray, list]:
        """
        Prepare features and target for modeling.

        Args:
            df: DataFrame with features and target
            target_col: Target column name (default: 'over_2.5')

        Returns:
            Tuple of (X, y, feature_names)

        Example:
            >>> X, y, features = pipeline.prepare_features_and_target(df)
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: PREPARE FEATURES AND TARGET")
        self.logger.info("=" * 60)

        try:
            # Get feature columns
            # Exclude non-feature columns
            exclude_cols = [
                'fixture_id', 'date', 'season',
                'home_team_name', 'away_team_name',
                'home_goals', 'away_goals', 'total_goals',
                'over_0.5', 'over_1.5', 'over_2.5', 'over_3.5', 'over_4.5'
            ]

            feature_cols = [col for col in df.columns if col not in exclude_cols]

            self.logger.info(f"Feature columns: {len(feature_cols)}")

            # Extract features and target
            X = df[feature_cols].copy()
            y = df[target_col].values

            # Handle missing values
            missing_counts = X.isnull().sum()
            missing_features = missing_counts[missing_counts > 0]

            if len(missing_features) > 0:
                self.logger.warning(f"Missing values found in {len(missing_features)} features")

                for col, count in missing_features.items():
                    pct = (count / len(X)) * 100
                    self.logger.warning(f"  {col}: {count} ({pct:.1f}%)")

                # Fill missing values with 0 (conservative approach)
                self.logger.info("Filling missing values with 0")
                X = X.fillna(0)

            # Check for infinite values
            inf_counts = np.isinf(X).sum()
            inf_features = inf_counts[inf_counts > 0]

            if len(inf_features) > 0:
                self.logger.warning(f"Infinite values found in {len(inf_features)} features")
                self.logger.info("Replacing infinite values with 0")
                X = X.replace([np.inf, -np.inf], 0)

            self.logger.info(f"\nFeature matrix shape: {X.shape}")
            self.logger.info(f"Target shape: {y.shape}")
            self.logger.info(f"Target distribution: {np.bincount(y.astype(int))}")

            self.logger.info("✓ Features and target prepared successfully")
            return X, y, feature_cols

        except Exception as e:
            self.logger.error(f"Failed to prepare features: {str(e)}", exc_info=True)
            raise

    def train_model(
        self,
        df: pd.DataFrame,
        model_type: Literal['poisson'] = 'poisson',
        threshold: float = 2.5
    ) -> PoissonGoalsModel:
        """
        Train model.

        Args:
            df: DataFrame with features and targets
            model_type: Type of model to train (default: 'poisson')
            threshold: Goals threshold (default: 2.5)

        Returns:
            Trained model

        Example:
            >>> model = pipeline.train_model(df)
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3: TRAIN MODEL")
        self.logger.info("=" * 60)

        try:
            # Prepare data
            X, y, feature_cols = self.prepare_features_and_target(df)

            self.logger.info(f"Training {model_type} model")
            self.logger.info(f"Threshold: {threshold}")

            # Train model
            if model_type == 'poisson':
                model = PoissonGoalsModel(threshold=threshold)
                model.fit(df, feature_cols)

                # Log model summary
                summary = model.get_model_summary()
                self.logger.info(f"\nModel Summary:")
                self.logger.info(f"  Home AIC: {summary['home_model']['aic']:.2f}")
                self.logger.info(f"  Away AIC: {summary['away_model']['aic']:.2f}")
                self.logger.info(f"  Home converged: {summary['home_model']['converged']}")
                self.logger.info(f"  Away converged: {summary['away_model']['converged']}")

                # Log significant coefficients
                coef_df = model.get_coefficients()
                significant = coef_df[
                    (coef_df['home_significant']) | (coef_df['away_significant'])
                ]

                self.logger.info(f"\nSignificant features (p < 0.05): {len(significant)}")

                # Show top 5 by absolute coefficient
                if len(significant) > 0:
                    significant['abs_home_coef'] = significant['home_coef'].abs()
                    top_home = significant.nlargest(5, 'abs_home_coef')

                    self.logger.info("\nTop 5 features for home goals:")
                    for _, row in top_home.iterrows():
                        self.logger.info(
                            f"  {row['feature']:30s}: coef={row['home_coef']:+.4f}, "
                            f"p={row['home_p_value']:.4f}"
                        )

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            self.logger.info("\n✓ Model trained successfully")
            return model

        except Exception as e:
            self.logger.error(f"Failed to train model: {str(e)}", exc_info=True)
            raise

    def evaluate_model(
        self,
        model: PoissonGoalsModel,
        df: pd.DataFrame,
        save_plots: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model with comprehensive metrics.

        Args:
            model: Trained model
            df: DataFrame with features and target
            save_plots: Whether to save evaluation plots

        Returns:
            Dictionary with evaluation metrics

        Example:
            >>> metrics = pipeline.evaluate_model(model, df_test)
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 4: EVALUATE MODEL")
        self.logger.info("=" * 60)

        try:
            # Get predictions
            y_true = df['over_2.5'].values
            y_pred_proba = model.predict_proba(df)[:, 1]

            # Evaluate
            if save_plots:
                plot_dir = self.plots_dir
            else:
                plot_dir = None

            metrics = ModelEvaluator.evaluate_model(
                y_true,
                y_pred_proba,
                plot=save_plots,
                save_dir=str(plot_dir) if plot_dir else None
            )

            self.logger.info("✓ Model evaluated successfully")
            return metrics

        except Exception as e:
            self.logger.error(f"Failed to evaluate model: {str(e)}", exc_info=True)
            raise

    def cross_validate(
        self,
        model_class: type,
        df: pd.DataFrame,
        n_splits: int = 5,
        threshold: float = 2.5
    ) -> pd.DataFrame:
        """
        Perform time series cross-validation.

        Args:
            model_class: Model class (e.g., PoissonGoalsModel)
            df: DataFrame with features and target
            n_splits: Number of CV splits
            threshold: Goals threshold

        Returns:
            DataFrame with CV results

        Example:
            >>> cv_results = pipeline.cross_validate(PoissonGoalsModel, df)
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 5: CROSS-VALIDATION")
        self.logger.info("=" * 60)

        try:
            # Prepare features
            _, _, feature_cols = self.prepare_features_and_target(df)

            # Initialize validator
            validator = TimeSeriesValidator()

            # Create model instance
            model = model_class(threshold=threshold)

            # Run validation
            self.logger.info(f"Running {n_splits}-fold time series cross-validation")

            cv_results = validator.validate(
                model=model,
                df=df,
                feature_cols=feature_cols,
                target_col='over_2.5',
                n_splits=n_splits
            )

            # Save results
            results_file = self.results_dir / 'cv_results.csv'
            cv_results.to_csv(results_file, index=False)
            self.logger.info(f"CV results saved to {results_file}")

            self.logger.info("✓ Cross-validation completed successfully")
            return cv_results

        except Exception as e:
            self.logger.error(f"Failed to cross-validate: {str(e)}", exc_info=True)
            raise

    def save_model(
        self,
        model: PoissonGoalsModel,
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Save model with versioning and metadata.

        Saves:
        1. Versioned model: poisson_model_YYYYMMDD_HHMMSS.pkl
        2. Latest model: poisson_model_latest.pkl
        3. Metadata: poisson_model_latest_metadata.json

        Args:
            model: Trained model
            metadata: Model metadata (metrics, features, etc.)
            version: Custom version string

        Returns:
            Dictionary with paths to saved files

        Example:
            >>> files = pipeline.save_model(model, metadata={'accuracy': 0.65})
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 6: SAVE MODEL")
        self.logger.info("=" * 60)

        try:
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if version is None:
                version = timestamp

            # File paths
            model_versioned = self.models_dir / f'poisson_model_{version}.pkl'
            model_latest = self.models_dir / 'poisson_model_latest.pkl'
            metadata_file = self.models_dir / 'poisson_model_latest_metadata.json'

            # Save versioned model
            self.logger.info(f"Saving versioned model: {model_versioned}")
            model.save(str(model_versioned))

            # Save latest model
            self.logger.info(f"Saving latest model: {model_latest}")
            model.save(str(model_latest))

            # Prepare metadata
            if metadata is None:
                metadata = {}

            model_metadata = {
                'model_type': 'PoissonGoalsModel',
                'threshold': model.threshold,
                'n_features': len(model.feature_cols),
                'feature_names': model.feature_cols,
                'saved_at': timestamp,
                'model_summary': model.get_model_summary(),
                **metadata
            }

            # Save metadata
            self.logger.info(f"Saving metadata: {metadata_file}")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(model_metadata, f, indent=2, ensure_ascii=False, default=str)

            # Calculate file sizes
            files = {
                'model_versioned': model_versioned,
                'model_latest': model_latest,
                'metadata': metadata_file
            }

            self.logger.info("\nFiles saved:")
            for file_type, file_path in files.items():
                size_kb = file_path.stat().st_size / 1024
                self.logger.info(f"  {file_type:20s}: {file_path} ({size_kb:.1f} KB)")

            self.logger.info("✓ Model saved successfully")
            return files

        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}", exc_info=True)
            raise

    def run_full_training(
        self,
        data_path: Optional[str] = None,
        cross_validate: bool = True,
        save_model: bool = True,
        n_cv_splits: int = 5
    ) -> Tuple[PoissonGoalsModel, Dict[str, float], Optional[pd.DataFrame]]:
        """
        Run complete model training pipeline.

        Pipeline stages:
        1. Load data
        2. Prepare features and target
        3. Train model
        4. Evaluate model
        5. Cross-validate (optional)
        6. Save model (optional)

        Args:
            data_path: Path to training data (optional)
            cross_validate: Whether to run cross-validation
            save_model: Whether to save trained model
            n_cv_splits: Number of CV splits

        Returns:
            Tuple of (trained_model, evaluation_metrics, cv_results)

        Example:
            >>> pipeline = ModelPipeline()
            >>> model, metrics, cv_results = pipeline.run_full_training()
        """
        pipeline_start = datetime.now()

        self.logger.info("\n" + "#" * 60)
        self.logger.info("# STARTING COMPLETE MODEL TRAINING PIPELINE")
        self.logger.info("#" * 60)
        self.logger.info(f"Data path: {data_path or 'default (latest)'}")
        self.logger.info(f"Cross-validation: {cross_validate}")
        self.logger.info(f"Save model: {save_model}")
        self.logger.info(f"Start time: {pipeline_start}")
        self.logger.info("#" * 60 + "\n")

        try:
            # Stage 1: Load data
            df = self.load_data(data_path)

            # Stage 2 & 3: Train model (prepare happens inside)
            model = self.train_model(df, model_type='poisson')

            # Stage 4: Evaluate model
            metrics = self.evaluate_model(model, df, save_plots=True)

            # Stage 5: Cross-validation (optional)
            cv_results = None
            if cross_validate:
                cv_results = self.cross_validate(
                    PoissonGoalsModel,
                    df,
                    n_splits=n_cv_splits
                )

            # Stage 6: Save model (optional)
            if save_model:
                save_metadata = {
                    'evaluation_metrics': metrics,
                    'data_path': str(data_path) if data_path else 'training_data_latest.parquet',
                    'n_samples': len(df)
                }

                if cv_results is not None:
                    # Add CV summary to metadata
                    mean_row = cv_results[cv_results['fold'] == 'mean'].iloc[0]
                    save_metadata['cv_metrics'] = {
                        'mean_accuracy': float(mean_row['accuracy']),
                        'mean_roc_auc': float(mean_row['roc_auc']),
                        'mean_f1_score': float(mean_row['f1_score'])
                    }

                self.save_model(model, metadata=save_metadata)

            # Pipeline summary
            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()

            self.logger.info("\n" + "#" * 60)
            self.logger.info("# TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("#" * 60)
            self.logger.info(f"Start time: {pipeline_start}")
            self.logger.info(f"End time:   {pipeline_end}")
            self.logger.info(f"Duration:   {duration:.1f} seconds")
            self.logger.info(f"\nFinal Results:")
            self.logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            self.logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
            self.logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")

            if cv_results is not None:
                mean_row = cv_results[cv_results['fold'] == 'mean'].iloc[0]
                std_row = cv_results[cv_results['fold'] == 'std'].iloc[0]

                self.logger.info(f"\nCross-Validation Results:")
                self.logger.info(f"  Accuracy:  {mean_row['accuracy']:.4f} ± {std_row['accuracy']:.4f}")
                self.logger.info(f"  ROC AUC:   {mean_row['roc_auc']:.4f} ± {std_row['roc_auc']:.4f}")

            self.logger.info("#" * 60 + "\n")

            return model, metrics, cv_results

        except Exception as e:
            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()

            self.logger.error("\n" + "#" * 60)
            self.logger.error("# TRAINING PIPELINE FAILED")
            self.logger.error("#" * 60)
            self.logger.error(f"Error: {str(e)}")
            self.logger.error(f"Duration before failure: {duration:.1f} seconds")
            self.logger.error("#" * 60 + "\n")

            raise


def run_quick_training() -> Tuple[PoissonGoalsModel, Dict[str, float]]:
    """
    Quick training pipeline for testing.

    Runs training without cross-validation for faster execution.

    Returns:
        Tuple of (trained_model, evaluation_metrics)

    Example:
        >>> model, metrics = run_quick_training()
    """
    from src.utils import setup_logging

    # Setup logging
    setup_logging()

    # Run pipeline
    pipeline = ModelPipeline()

    model, metrics, _ = pipeline.run_full_training(
        cross_validate=False,
        save_model=True
    )

    return model, metrics


def main():
    """Run complete model training pipeline."""
    from src.utils import setup_logging

    # Setup logging
    setup_logging()

    # Initialize pipeline
    pipeline = ModelPipeline()

    # Run full training
    model, metrics, cv_results = pipeline.run_full_training(
        cross_validate=True,
        save_model=True,
        n_cv_splits=5
    )

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED!")
    print("=" * 60)
    print(f"\nEvaluation Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")

    if cv_results is not None:
        mean_row = cv_results[cv_results['fold'] == 'mean'].iloc[0]

        print(f"\nCross-Validation Metrics:")
        print(f"  Accuracy:  {mean_row['accuracy']:.4f}")
        print(f"  ROC AUC:   {mean_row['roc_auc']:.4f}")
        print(f"  F1 Score:  {mean_row['f1_score']:.4f}")

    print(f"\nModel saved to: {pipeline.models_dir}")
    print(f"Plots saved to: {pipeline.plots_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
