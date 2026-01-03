"""
Poisson Regression Model for predicting match goals.
Uses dual independent Poisson GLMs for home and away goals.
"""

import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from scipy.stats import poisson
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson

from ..utils import LoggerMixin


class PoissonGoalsModel(LoggerMixin):
    """
    Dual Poisson Regression Model for Premier League goals prediction.

    Uses two independent Generalized Linear Models (GLM) with Poisson family:
    - home_model: Predicts expected home team goals (lambda_home)
    - away_model: Predicts expected away team goals (lambda_away)

    Calculates over/under probabilities analytically using Poisson distributions.

    Attributes:
        home_model: GLM for home goals
        away_model: GLM for away goals
        threshold: Goals threshold for over/under (default: 2.5)
        feature_cols: List of feature column names used for training
    """

    def __init__(self, threshold: float = 2.5):
        """
        Initialize Poisson Goals Model.

        Args:
            threshold: Goals threshold for over/under predictions (default: 2.5)
        """
        self.home_model = None
        self.away_model = None
        self.threshold = threshold
        self.feature_cols = None

        self.logger.info(f"PoissonGoalsModel initialized with threshold={threshold}")

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_home: str = 'home_goals',
        target_away: str = 'away_goals'
    ) -> 'PoissonGoalsModel':
        """
        Fit dual Poisson regression models.

        Trains two independent GLM models:
        1. home_model: X -> home_goals
        2. away_model: X -> away_goals

        Args:
            df: Training DataFrame
            feature_cols: List of feature column names to use
            target_home: Column name for home goals (default: 'home_goals')
            target_away: Column name for away goals (default: 'away_goals')

        Returns:
            Self for method chaining

        Example:
            >>> model = PoissonGoalsModel()
            >>> model.fit(df_train, feature_cols=['expected_total_goals', 'form_difference'])
        """
        self.logger.info("Fitting Poisson regression models")
        self.logger.info(f"Training samples: {len(df)}")
        self.logger.info(f"Features: {len(feature_cols)}")

        # Store feature columns
        self.feature_cols = feature_cols.copy()

        # Prepare features
        X = df[feature_cols].copy()

        # Add constant term for intercept
        X = sm.add_constant(X, has_constant='add')

        # Prepare targets
        y_home = df[target_home].values
        y_away = df[target_away].values

        # Fit home goals model
        self.logger.info("Training home goals model...")
        self.home_model = sm.GLM(
            y_home,
            X,
            family=Poisson()
        ).fit()

        self.logger.info(f"Home model AIC: {self.home_model.aic:.2f}")
        self.logger.info(f"Home model converged: {self.home_model.converged}")

        # Fit away goals model
        self.logger.info("Training away goals model...")
        self.away_model = sm.GLM(
            y_away,
            X,
            family=Poisson()
        ).fit()

        self.logger.info(f"Away model AIC: {self.away_model.aic:.2f}")
        self.logger.info(f"Away model converged: {self.away_model.converged}")

        self.logger.info("Model training completed successfully")

        return self

    def predict_expected_goals(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict expected goals (lambda values) for home and away teams.

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (lambda_home, lambda_away) arrays

        Example:
            >>> lambda_home, lambda_away = model.predict_expected_goals(df_test)
        """
        if self.home_model is None or self.away_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Prepare features
        X = df[self.feature_cols].copy()
        X = sm.add_constant(X, has_constant='add')

        # Predict expected goals
        lambda_home = self.home_model.predict(X)
        lambda_away = self.away_model.predict(X)

        return lambda_home, lambda_away

    def predict_over_under_analytical(
        self,
        lambda_home: float,
        lambda_away: float,
        threshold: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate over/under probabilities analytically using Poisson distributions.

        P(Total goals > threshold) = Sum of P(home=i, away=j) for all i+j > threshold

        Since home and away goals are independent:
        P(home=i, away=j) = P(home=i) * P(away=j)

        Args:
            lambda_home: Expected home goals (Poisson parameter)
            lambda_away: Expected away goals (Poisson parameter)
            threshold: Goals threshold (default: self.threshold)

        Returns:
            Tuple of (prob_over, prob_under)

        Example:
            >>> prob_over, prob_under = model.predict_over_under_analytical(1.5, 1.2, 2.5)
        """
        if threshold is None:
            threshold = self.threshold

        # Calculate probabilities for all reasonable goal combinations
        # Using max_goals = 10 covers 99.9%+ of cases
        max_goals = 10

        prob_over = 0.0

        # Sum probabilities for all combinations where total > threshold
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                total_goals = home_goals + away_goals

                if total_goals > threshold:
                    # P(home=i) * P(away=j)
                    prob_home = poisson.pmf(home_goals, lambda_home)
                    prob_away = poisson.pmf(away_goals, lambda_away)
                    prob_combination = prob_home * prob_away

                    prob_over += prob_combination

        prob_under = 1.0 - prob_over

        return prob_over, prob_under

    def predict_over_under(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict over/under probabilities for multiple matches.

        Args:
            df: DataFrame with features
            threshold: Goals threshold (default: self.threshold)

        Returns:
            Tuple of (prob_over, prob_under, expected_total_goals) arrays

        Example:
            >>> prob_over, prob_under, expected_goals = model.predict_over_under(df_test)
        """
        if threshold is None:
            threshold = self.threshold

        self.logger.debug(f"Predicting over/under with threshold={threshold}")

        # Get expected goals
        lambda_home, lambda_away = self.predict_expected_goals(df)

        # Calculate expected total goals
        expected_total_goals = lambda_home + lambda_away

        # Calculate over/under probabilities for each match
        prob_over = np.zeros(len(df))
        prob_under = np.zeros(len(df))

        for i in range(len(df)):
            prob_over[i], prob_under[i] = self.predict_over_under_analytical(
                lambda_home[i],
                lambda_away[i],
                threshold
            )

        return prob_over, prob_under, expected_total_goals

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (sklearn-compatible).

        Args:
            df: DataFrame with features

        Returns:
            Array of shape (n_samples, 2) with [prob_under, prob_over]

        Example:
            >>> proba = model.predict_proba(df_test)
            >>> proba[0]  # [0.35, 0.65] = [prob_under, prob_over]
        """
        prob_over, prob_under, _ = self.predict_over_under(df, threshold=self.threshold)

        # Stack as [prob_under, prob_over] to match sklearn convention
        # where class 0 = under, class 1 = over
        return np.column_stack([prob_under, prob_over])

    def predict(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict binary class (0 = under, 1 = over).

        Args:
            df: DataFrame with features
            threshold: Goals threshold (default: self.threshold)

        Returns:
            Binary predictions array (0 or 1)

        Example:
            >>> predictions = model.predict(df_test)
            >>> predictions  # [1, 0, 1, 1, 0, ...]
        """
        prob_over, prob_under, _ = self.predict_over_under(df, threshold)

        # Predict 1 (over) if prob_over > prob_under, else 0 (under)
        return (prob_over > prob_under).astype(int)

    def get_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients with statistics.

        Returns:
            DataFrame with coefficients, standard errors, z-values, and p-values
            for both home and away models

        Example:
            >>> coef_df = model.get_coefficients()
            >>> print(coef_df)
        """
        if self.home_model is None or self.away_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Extract home model coefficients
        home_summary = pd.DataFrame({
            'feature': self.home_model.params.index,
            'home_coef': self.home_model.params.values,
            'home_std_err': self.home_model.bse.values,
            'home_z_value': self.home_model.tvalues.values,
            'home_p_value': self.home_model.pvalues.values
        })

        # Extract away model coefficients
        away_summary = pd.DataFrame({
            'feature': self.away_model.params.index,
            'away_coef': self.away_model.params.values,
            'away_std_err': self.away_model.bse.values,
            'away_z_value': self.away_model.tvalues.values,
            'away_p_value': self.away_model.pvalues.values
        })

        # Merge both
        coef_df = home_summary.merge(away_summary, on='feature', how='outer')

        # Add significance indicators
        coef_df['home_significant'] = coef_df['home_p_value'] < 0.05
        coef_df['away_significant'] = coef_df['away_p_value'] < 0.05

        return coef_df

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary statistics.

        Returns:
            Dictionary with model statistics

        Example:
            >>> summary = model.get_model_summary()
            >>> print(summary['home_aic'])
        """
        if self.home_model is None or self.away_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        summary = {
            'threshold': self.threshold,
            'n_features': len(self.feature_cols),
            'feature_names': self.feature_cols,
            'home_model': {
                'aic': self.home_model.aic,
                'bic': self.home_model.bic,
                'deviance': self.home_model.deviance,
                'pearson_chi2': self.home_model.pearson_chi2,
                'converged': self.home_model.converged,
                'n_iterations': self.home_model.fit_history['iteration'] if hasattr(self.home_model, 'fit_history') else None
            },
            'away_model': {
                'aic': self.away_model.aic,
                'bic': self.away_model.bic,
                'deviance': self.away_model.deviance,
                'pearson_chi2': self.away_model.pearson_chi2,
                'converged': self.away_model.converged,
                'n_iterations': self.away_model.fit_history['iteration'] if hasattr(self.away_model, 'fit_history') else None
            }
        }

        return summary

    def save(self, filepath: str) -> None:
        """
        Save model to file using pickle.

        Args:
            filepath: Path to save the model

        Example:
            >>> model.save('models/poisson_model.pkl')
        """
        self.logger.info(f"Saving model to {filepath}")

        model_data = {
            'home_model': self.home_model,
            'away_model': self.away_model,
            'threshold': self.threshold,
            'feature_cols': self.feature_cols
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Model saved successfully")

    @classmethod
    def load(cls, filepath: str) -> 'PoissonGoalsModel':
        """
        Load model from file.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded PoissonGoalsModel instance

        Example:
            >>> model = PoissonGoalsModel.load('models/poisson_model.pkl')
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create new instance
        model = cls(threshold=model_data['threshold'])
        model.home_model = model_data['home_model']
        model.away_model = model_data['away_model']
        model.feature_cols = model_data['feature_cols']

        model.logger.info(f"Model loaded from {filepath}")
        model.logger.info(f"Threshold: {model.threshold}")
        model.logger.info(f"Features: {len(model.feature_cols)}")

        return model

    def __repr__(self) -> str:
        """String representation of the model."""
        if self.home_model is None:
            return "PoissonGoalsModel(not trained)"

        return (
            f"PoissonGoalsModel(threshold={self.threshold}, "
            f"n_features={len(self.feature_cols)}, "
            f"home_AIC={self.home_model.aic:.2f}, "
            f"away_AIC={self.away_model.aic:.2f})"
        )
