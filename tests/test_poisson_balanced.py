import warnings

import numpy as np
import pandas as pd
import pytest

from src.models.poisson_balanced import BalancedPoissonGoalsModel


@pytest.fixture
def sample_poisson_data():
    np.random.seed(123)
    n = 200

    # Simple synthetic numeric features
    df = pd.DataFrame({
        'f1': np.random.normal(size=n),
        'f2': np.random.normal(size=n),
        'f3': np.random.normal(size=n),
    })

    # Generate goals with mild dependence on f1/f2
    lam_home = np.exp(0.2 + 0.3 * df['f1'] - 0.2 * df['f2'])
    lam_away = np.exp(0.1 - 0.2 * df['f1'] + 0.3 * df['f2'])

    df['home_goals'] = np.random.poisson(lam=np.clip(lam_home, 0.1, 5.0))
    df['away_goals'] = np.random.poisson(lam=np.clip(lam_away, 0.1, 5.0))
    df['total_goals'] = df['home_goals'] + df['away_goals']
    df['over_2.5'] = (df['total_goals'] > 2.5).astype(int)

    return df


def test_predict_proba_sums_to_one(sample_poisson_data):
    model = BalancedPoissonGoalsModel(
        threshold=2.5,
        alpha_grid=[0.0],
        cv_splits=2,
        calibration_fraction=0.25,
        random_state=42,
    )

    features = ['f1', 'f2', 'f3']
    model.fit(sample_poisson_data, features)

    proba = model.predict_proba(sample_poisson_data.head(25))
    assert proba.shape == (25, 2)

    sums = proba.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-8)
    assert np.isfinite(proba).all()


def test_optimal_threshold_in_range(sample_poisson_data):
    model = BalancedPoissonGoalsModel(
        threshold=2.5,
        alpha_grid=[0.0],
        cv_splits=2,
        calibration_fraction=0.25,
        random_state=7,
    )

    features = ['f1', 'f2', 'f3']
    model.fit(sample_poisson_data, features)

    assert 0.0 <= model.optimal_threshold <= 1.0


def test_predict_with_optimal_threshold_runs(sample_poisson_data):
    model = BalancedPoissonGoalsModel(
        threshold=2.5,
        alpha_grid=[0.0],
        cv_splits=2,
        calibration_fraction=0.25,
        random_state=7,
    )

    features = ['f1', 'f2', 'f3']
    model.fit(sample_poisson_data, features)

    preds = model.predict_with_optimal_threshold(sample_poisson_data.head(10))
    assert preds.shape == (10,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_bias_warning_if_single_class_dominates(sample_poisson_data, monkeypatch):
    model = BalancedPoissonGoalsModel(
        threshold=2.5,
        alpha_grid=[0.0],
        cv_splits=2,
        calibration_fraction=0.25,
        random_state=7,
    )

    features = ['f1', 'f2', 'f3']
    model.fit(sample_poisson_data, features)

    # Force uncalibrated probabilities to be extremely skewed to class 1 (over)
    def _skewed(_df):
        n = len(_df)
        return np.column_stack([np.full(n, 0.01), np.full(n, 0.99)])

    monkeypatch.setattr(model, '_predict_proba_uncalibrated', _skewed)
    model.calibrator = None  # keep it simple for the warning test

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _ = model.predict_proba(sample_poisson_data.head(50))
        assert any('>75%' in str(wi.message) for wi in w)


def test_get_prediction_confidence_range(sample_poisson_data):
    model = BalancedPoissonGoalsModel(
        threshold=2.5,
        alpha_grid=[0.0],
        cv_splits=2,
        calibration_fraction=0.25,
        random_state=7,
    )

    features = ['f1', 'f2', 'f3']
    model.fit(sample_poisson_data, features)

    conf = model.get_prediction_confidence(sample_poisson_data.head(20))
    assert conf.shape == (20,)
    assert (conf >= 0.0).all() and (conf <= 1.0).all()


def test_explain_prediction_has_keys(sample_poisson_data):
    model = BalancedPoissonGoalsModel(
        threshold=2.5,
        alpha_grid=[0.0],
        cv_splits=2,
        calibration_fraction=0.25,
        random_state=7,
    )

    features = ['f1', 'f2', 'f3']
    model.fit(sample_poisson_data, features)

    info = model.explain_prediction(sample_poisson_data, idx=0)
    for key in [
        'idx',
        'prob_under',
        'prob_over',
        'optimal_threshold',
        'home_linear_predictor',
        'away_linear_predictor',
        'top_contributions_home',
        'top_contributions_away',
    ]:
        assert key in info


def test_get_balanced_metrics(sample_poisson_data):
    model = BalancedPoissonGoalsModel(
        threshold=2.5,
        alpha_grid=[0.0],
        cv_splits=2,
        calibration_fraction=0.25,
        random_state=7,
    )

    features = ['f1', 'f2', 'f3']
    model.fit(sample_poisson_data, features)

    y_true = sample_poisson_data['over_2.5'].values
    proba = model.predict_proba(sample_poisson_data)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    metrics = model.get_balanced_metrics(y_true, y_pred, y_prob_over=proba)
    for k in ['balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        assert k in metrics
        assert np.isfinite(metrics[k])
