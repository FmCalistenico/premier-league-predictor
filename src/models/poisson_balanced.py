import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import gammaln
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from .poisson_model import PoissonGoalsModel


@dataclass
class _AlphaCVResult:
    alpha: float
    score: float


class _PrefitPoissonOverClassifier:
    def __init__(self, model: 'BalancedPoissonGoalsModel'):
        self.model = model
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X)
        # IMPORTANT: must use uncalibrated probabilities to avoid recursion
        return self.model._predict_proba_uncalibrated(df)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class BalancedPoissonGoalsModel(PoissonGoalsModel):
    def __init__(
        self,
        threshold: float = 2.5,
        alpha_grid: Optional[List[float]] = None,
        cv_splits: int = 3,
        calibration_fraction: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__(threshold=threshold)
        self.alpha_grid = alpha_grid or [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
        self.cv_splits = cv_splits
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

        self.optimal_threshold: float = 0.5
        self.best_alpha: float = 0.0
        self.calibrator: Optional[IsotonicRegression] = None
        self.sample_weights_: Optional[np.ndarray] = None

    def _compute_over_label(self, df: pd.DataFrame) -> np.ndarray:
        if 'over_2.5' in df.columns:
            return df['over_2.5'].astype(int).values
        if 'total_goals' in df.columns:
            total = df['total_goals'].values
        else:
            total = (df['home_goals'].values + df['away_goals'].values)
        return (total > self.threshold).astype(int)

    def _compute_class_weights(self, y_over: np.ndarray) -> Tuple[float, float, np.ndarray]:
        n_total = int(len(y_over))
        n_over = int(y_over.sum())
        n_under = int(n_total - n_over)

        if n_over == 0 or n_under == 0:
            self.logger.warning(
                "Class distribution is degenerate; skipping class weighting"
            )
            weights = np.ones(n_total, dtype=float)
            return 1.0, 1.0, weights

        weight_under = n_total / (2.0 * n_under)
        weight_over = n_total / (2.0 * n_over)
        weights = np.where(y_over == 1, weight_over, weight_under).astype(float)
        return weight_under, weight_over, weights

    def _poisson_loglik(self, y: np.ndarray, mu: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        mu = np.clip(mu, 1e-10, 1e10)
        ll = y * np.log(mu) - mu - gammaln(y + 1.0)
        if weights is not None:
            ll = ll * weights
        return float(np.mean(ll))

    def _select_alpha_via_cv(
        self,
        X: pd.DataFrame,
        y_home: np.ndarray,
        y_away: np.ndarray,
        sample_weights: Optional[np.ndarray],
    ) -> float:
        if len(self.alpha_grid) == 1:
            return float(self.alpha_grid[0])

        y_over = (y_home + y_away > self.threshold).astype(int)
        cv = StratifiedKFold(
            n_splits=max(2, int(self.cv_splits)),
            shuffle=True,
            random_state=self.random_state,
        )

        results: List[_AlphaCVResult] = []
        for alpha in self.alpha_grid:
            fold_scores: List[float] = []
            for train_idx, val_idx in cv.split(X, y_over):
                X_tr = X.iloc[train_idx]
                X_va = X.iloc[val_idx]
                y_home_tr = y_home[train_idx]
                y_away_tr = y_away[train_idx]
                y_home_va = y_home[val_idx]
                y_away_va = y_away[val_idx]

                w_tr = sample_weights[train_idx] if sample_weights is not None else None
                w_va = sample_weights[val_idx] if sample_weights is not None else None

                exog_tr = sm.add_constant(X_tr, has_constant='add')
                exog_va = sm.add_constant(X_va, has_constant='add')

                mod_home = sm.GLM(y_home_tr, exog_tr, family=sm.families.Poisson(), var_weights=w_tr)
                mod_away = sm.GLM(y_away_tr, exog_tr, family=sm.families.Poisson(), var_weights=w_tr)

                if alpha and alpha > 0:
                    res_home = mod_home.fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=2000)
                    res_away = mod_away.fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=2000)
                else:
                    res_home = mod_home.fit(method='newton', maxiter=200, tol=1e-8)
                    res_away = mod_away.fit(method='newton', maxiter=200, tol=1e-8)

                mu_home = res_home.predict(exog_va)
                mu_away = res_away.predict(exog_va)

                ll_home = self._poisson_loglik(y_home_va, mu_home, weights=w_va)
                ll_away = self._poisson_loglik(y_away_va, mu_away, weights=w_va)
                fold_scores.append((ll_home + ll_away) / 2.0)

            results.append(_AlphaCVResult(alpha=float(alpha), score=float(np.mean(fold_scores))))

        best = max(results, key=lambda r: r.score)
        self.logger.info(
            f"Selected alpha via CV: {best.alpha} (score={best.score:.6f})"
        )
        return best.alpha

    def _fit_poisson_glm(
        self,
        y: np.ndarray,
        exog: pd.DataFrame,
        label: str,
        alpha: float,
        sample_weights: Optional[np.ndarray],
    ):
        mod = sm.GLM(y, exog, family=sm.families.Poisson(), var_weights=sample_weights)
        try:
            if alpha and alpha > 0:
                res = mod.fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=2000)
            else:
                res = mod.fit(method='newton', maxiter=200, tol=1e-8)

            if getattr(res, 'converged', True) is False:
                raise ValueError("non_converged")

            return res
        except Exception as e:
            self.logger.warning(
                f"{label} model fit failed ({type(e).__name__}: {e}); retrying with stronger regularization"
            )
            alpha_fallback = max(float(alpha), 1e-4)
            res = mod.fit_regularized(alpha=alpha_fallback, L1_wt=0.0, maxiter=5000)
            return res

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_home: str = 'home_goals',
        target_away: str = 'away_goals',
    ) -> 'BalancedPoissonGoalsModel':
        self.logger.info("Fitting Balanced Poisson regression models")

        X = df[feature_cols].copy()
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            self.logger.warning(f"Dropping non-numeric columns: {non_numeric}")
            X = X.select_dtypes(include=[np.number])
            feature_cols = X.columns.tolist()
            self.logger.info(f"Using {len(feature_cols)} numeric features")

        if X.isnull().any().any():
            self.logger.warning("Filling missing values in features with 0")
            X = X.fillna(0)

        X_values = X.values
        if np.isinf(X_values).any():
            self.logger.warning("Replacing infinite values in features with 0")
            X = X.replace([np.inf, -np.inf], 0)

        X_values = X.values
        extreme_mask = np.abs(X_values) > 1e6
        if extreme_mask.any():
            extreme_count = int(extreme_mask.sum())
            self.logger.warning(f"Clipping {extreme_count} extreme values to ±1e6")
            X = X.clip(-1e6, 1e6)

        feature_std = X.std()
        low_variance_features = feature_std[feature_std < 1e-10].index.tolist()
        if low_variance_features:
            self.logger.warning(f"Dropping {len(low_variance_features)} near-constant features")
            X = X.drop(columns=low_variance_features)
            feature_cols = X.columns.tolist()

        self.feature_cols = feature_cols.copy()

        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        X_values = X.values
        if np.isnan(X_values).any() or np.isinf(X_values).any():
            raise ValueError("Features contain NaN or infinite values after cleaning and scaling")

        y_home = df[target_home].values
        y_away = df[target_away].values
        y_mask = (
            np.isfinite(y_home)
            & np.isfinite(y_away)
            & (y_home >= 0)
            & (y_away >= 0)
        )
        if not y_mask.all():
            dropped = int((~y_mask).sum())
            self.logger.warning(
                f"Dropping {dropped} rows with invalid targets (NaN/inf/negative) before GLM fit"
            )
            X = X.loc[X.index[y_mask]]
            y_home = y_home[y_mask]
            y_away = y_away[y_mask]

        y_over = self._compute_over_label(df.loc[X.index])
        weight_under, weight_over, sample_weights = self._compute_class_weights(y_over)
        self.sample_weights_ = sample_weights
        self.logger.info(
            f"Class weights: under={weight_under:.4f}, over={weight_over:.4f} (base_rate_over={y_over.mean():.3f})"
        )

        self.best_alpha = self._select_alpha_via_cv(X, y_home, y_away, sample_weights)

        exog = sm.add_constant(X, has_constant='add')
        self.logger.info("Training home goals model...")
        self.home_model = self._fit_poisson_glm(y_home, exog, "Home", self.best_alpha, sample_weights)

        self.logger.info("Training away goals model...")
        self.away_model = self._fit_poisson_glm(y_away, exog, "Away", self.best_alpha, sample_weights)

        self.logger.info("Model training completed successfully")

        self._fit_calibration_and_threshold(df.loc[X.index])

        return self

    def _fit_calibration_and_threshold(self, df_train: pd.DataFrame) -> None:
        y_over = self._compute_over_label(df_train)
        n = len(df_train)
        n_cal = int(max(1, round(n * float(self.calibration_fraction))))
        if n_cal >= n:
            n_cal = max(1, n - 1)

        rng = np.random.RandomState(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        cal_idx = idx[:n_cal]

        df_cal = df_train.iloc[cal_idx]
        y_cal = y_over[cal_idx]

        # Fit isotonic regression calibrator on raw probabilities
        proba_raw = self._predict_proba_uncalibrated(df_cal)[:, 1]
        self.calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds='clip',
        )
        self.calibrator.fit(proba_raw, y_cal)

        proba_cal = self.calibrator.transform(proba_raw)
        self.optimal_threshold = self.find_optimal_threshold(y_cal, proba_cal)

    def _predict_proba_uncalibrated(self, df: pd.DataFrame) -> np.ndarray:
        prob_over, prob_under, _ = self.predict_over_under(df, threshold=self.threshold)

        prob_over = np.clip(prob_over, 0.0, 1.0)
        prob_under = np.clip(prob_under, 0.0, 1.0)

        total = prob_over + prob_under
        prob_over = np.where(total > 0, prob_over / total, 0.5)
        prob_under = np.where(total > 0, prob_under / total, 0.5)

        prob_over = np.where(np.isfinite(prob_over), prob_over, 0.5)
        prob_under = np.where(np.isfinite(prob_under), prob_under, 0.5)

        total = prob_over + prob_under
        prob_over = np.where(total > 0, prob_over / total, 0.5)
        prob_under = np.where(total > 0, prob_under / total, 0.5)

        return np.column_stack([prob_under, prob_over])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        proba_raw = self._predict_proba_uncalibrated(df)
        prob_under = proba_raw[:, 0]
        prob_over = proba_raw[:, 1]

        if self.calibrator is not None:
            prob_over_cal = self.calibrator.transform(prob_over)
            prob_over = np.clip(prob_over_cal, 0.0, 1.0)
            prob_under = 1.0 - prob_over

        total = prob_over + prob_under
        prob_over = np.where(total > 0, prob_over / total, 0.5)
        prob_under = np.where(total > 0, prob_under / total, 0.5)

        proba = np.column_stack([prob_under, prob_over])

        mean_over = float(np.mean(prob_over))
        self.logger.info(
            f"Pred prob_over distribution: mean={mean_over:.3f}, p10={np.quantile(prob_over, 0.1):.3f}, p50={np.quantile(prob_over, 0.5):.3f}, p90={np.quantile(prob_over, 0.9):.3f}"
        )

        preds = (prob_over >= 0.5).astype(int)
        frac_over = float(preds.mean())
        frac_under = 1.0 - frac_over
        if max(frac_over, frac_under) > 0.75:
            warnings.warn(
                f"Model predicts >75% of a single class (over={frac_over:.1%}, under={frac_under:.1%})",
                RuntimeWarning,
            )

        return proba

    def find_optimal_threshold(self, y_true: np.ndarray, prob_over: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return 0.5
        fpr, tpr, thresholds = roc_curve(y_true, prob_over)
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        thr = float(thresholds[best_idx])
        return float(np.clip(thr, 0.0, 1.0))

    def predict_with_optimal_threshold(self, df: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(df)
        return (proba[:, 1] >= float(self.optimal_threshold)).astype(int)

    def get_prediction_confidence(self, df: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(df)[:, 1]
        conf = np.abs(proba - 0.5) * 2.0
        return np.clip(conf, 0.0, 1.0)

    def explain_prediction(self, df: pd.DataFrame, idx: int) -> Dict[str, Any]:
        if self.home_model is None or self.away_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        row = df.iloc[[idx]]
        X = row[self.feature_cols].copy().select_dtypes(include=[np.number])
        if X.isnull().any().any():
            X = X.fillna(0)
        if np.isinf(X.values).any():
            X = X.replace([np.inf, -np.inf], 0)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        X = sm.add_constant(X, has_constant='add')

        home_params = self.home_model.params
        away_params = self.away_model.params

        # Align by index to avoid any ordering issues
        X_row = X.iloc[0]
        home_lin = float((X_row[home_params.index] * home_params).sum())
        away_lin = float((X_row[away_params.index] * away_params).sum())

        contrib_home = (X_row[home_params.index] * home_params).sort_values(
            key=lambda s: np.abs(s), ascending=False
        )
        contrib_away = (X_row[away_params.index] * away_params).sort_values(
            key=lambda s: np.abs(s), ascending=False
        )

        proba = self.predict_proba(row)[0]

        return {
            'idx': idx,
            'prob_under': float(proba[0]),
            'prob_over': float(proba[1]),
            'optimal_threshold': float(self.optimal_threshold),
            'home_linear_predictor': home_lin,
            'away_linear_predictor': away_lin,
            'top_contributions_home': contrib_home.head(10).to_dict(),
            'top_contributions_away': contrib_away.head(10).to_dict(),
        }

    def get_balanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob_over: Optional[np.ndarray] = None) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if y_prob_over is not None and len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob_over))
        return metrics
