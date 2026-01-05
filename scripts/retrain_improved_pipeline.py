import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit
try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # pragma: no cover
    XGBClassifier = None

from src.data import DataExtractor, DataTransformer
from src.features import FeatureEngineerV2
from src.features.engineering import FeatureEngineer
from src.models.evaluator import ModelEvaluator
from src.models.poisson_balanced import BalancedPoissonGoalsModel
from src.models.poisson_model import PoissonGoalsModel
from src.utils import Config, get_logger, setup_logging


class EnsembleOverUnderModel:
    def __init__(
        self,
        poisson_balanced: BalancedPoissonGoalsModel,
        xgb: Any,
        feature_cols: List[str],
        weight_poisson: float = 0.5,
        weight_xgb: float = 0.5,
    ):
        self.poisson_balanced = poisson_balanced
        self.xgb = xgb
        self.feature_cols = feature_cols
        self.weight_poisson = float(weight_poisson)
        self.weight_xgb = float(weight_xgb)

        total = self.weight_poisson + self.weight_xgb
        if total <= 0:
            self.weight_poisson = 0.5
            self.weight_xgb = 0.5
        else:
            self.weight_poisson /= total
            self.weight_xgb /= total

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p_p = self.poisson_balanced.predict_proba(df)[:, 1]
        p_x = self.xgb.predict_proba(df[self.feature_cols])[:, 1]
        p_over = np.clip(self.weight_poisson * p_p + self.weight_xgb * p_x, 0.0, 1.0)
        p_under = 1.0 - p_over
        total = p_over + p_under
        p_over = np.where(total > 0, p_over / total, 0.5)
        p_under = np.where(total > 0, p_under / total, 0.5)
        return np.column_stack([p_under, p_over])


def _timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _setup_run_logging(config: Config) -> Path:
    ts = _timestamp()
    log_path = config.logs_path / f"retrain_{ts}.log"
    setup_logging(log_dir=str(config.logs_path))
    logger = get_logger(__name__)

    import logging

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fh)

    logger.info(f"Logging to {log_path}")
    return log_path


def parse_arguments():
    p = argparse.ArgumentParser(description='Retrain improved pipeline (V2 features + balanced models)')
    p.add_argument(
        '--seasons',
        nargs='+',
        default=['2324', '2425', '2526'],
        help='Seasons to use for CSV extraction when needed (default: 2324 2425 2526)'
    )
    p.add_argument('--skip-baseline', action='store_true', help='Skip original Poisson baseline')
    p.add_argument('--quick', action='store_true', help='Quick mode (1 fold CV)')
    p.add_argument(
        '--models',
        type=str,
        default='all',
        help='Comma-separated subset: poisson,poisson_opt,poisson_balanced,xgb,ensemble or all',
    )
    return p.parse_args()


def _ensure_targets(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    df = df.copy()
    if 'total_goals' not in df.columns:
        df['total_goals'] = df['home_goals'] + df['away_goals']
    if 'over_2.5' not in df.columns:
        df['over_2.5'] = (df['total_goals'] > threshold).astype(int)
    return df


def _basic_integrity_checks(df: pd.DataFrame) -> List[str]:
    issues: List[str] = []
    required = ['date', 'home_goals', 'away_goals']
    for c in required:
        if c not in df.columns:
            issues.append(f"missing_column:{c}")
    if 'date' in df.columns and not np.issubdtype(df['date'].dtype, np.datetime64):
        issues.append('date_not_datetime')
    if 'home_goals' in df.columns and (df['home_goals'] < 0).any():
        issues.append('home_goals_negative')
    if 'away_goals' in df.columns and (df['away_goals'] < 0).any():
        issues.append('away_goals_negative')
    return issues


def _youden_optimal_threshold(y_true: np.ndarray, prob_over: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, prob_over)
    j = tpr - fpr
    idx = int(np.argmax(j))
    thr = float(thresholds[idx])
    return float(np.clip(thr, 0.0, 1.0))


def _numeric_feature_cols(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    # Keep only columns that exist and are numeric/bool
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols]
    X_num = X.select_dtypes(include=[np.number, bool])
    return X_num.columns.tolist()


def _eval_binary_metrics(y_true: np.ndarray, prob_over: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (prob_over >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    out: Dict[str, Any] = {
        'roc_auc': float(roc_auc_score(y_true, prob_over)) if len(np.unique(y_true)) > 1 else np.nan,
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'specificity': float(specificity),
        'sensitivity': float(sensitivity),
        'pred_over_rate': float(y_pred.mean()),
        'calibration_error': float(ModelEvaluator.calculate_calibration_error(y_true, prob_over)),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }
    return out


def _get_feature_sets(df_transformed: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    engineer_v2 = FeatureEngineerV2()
    df_features = engineer_v2.engineer_features(df_transformed)
    feature_cols = engineer_v2.get_feature_list(df_features)

    validation = engineer_v2.validate_features(df_features)
    feature_metadata = engineer_v2.get_feature_metadata()
    if not feature_metadata:
        feature_metadata = validation

    return df_features, feature_cols, feature_metadata


def _compare_with_v1(df_transformed: pd.DataFrame, df_v2: pd.DataFrame) -> Dict[str, Any]:
    engineer_v1 = FeatureEngineer()
    df_v1 = engineer_v1.engineer_features(df_transformed.copy())

    exclude = {
        'home_goals', 'away_goals', 'total_goals', 'over_0.5', 'over_1.5', 'over_2.5', 'over_3.5', 'over_4.5',
        'home_team', 'away_team', 'date', 'season',
        'home_team_name', 'away_team_name',
    }
    features_v1 = [c for c in df_v1.columns if c not in exclude]

    engineer_v2 = FeatureEngineerV2()
    features_v2 = engineer_v2.get_feature_list(df_v2)

    removed = sorted(list(set(features_v1) - set(features_v2)))
    added = sorted(list(set(features_v2) - set(features_v1)))
    common = sorted(list(set(features_v1) & set(features_v2)))

    return {
        'v1_count': len(features_v1),
        'v2_count': len(features_v2),
        'removed': removed,
        'added': added,
        'common': common,
    }


def _train_poisson_original(train_df: pd.DataFrame, feature_cols: List[str]) -> PoissonGoalsModel:
    m = PoissonGoalsModel()
    m.fit(train_df, feature_cols)
    return m


def _train_poisson_balanced(train_df: pd.DataFrame, feature_cols: List[str]) -> BalancedPoissonGoalsModel:
    m = BalancedPoissonGoalsModel()
    m.fit(train_df, feature_cols)
    return m


def _train_xgb(train_df: pd.DataFrame, feature_cols: List[str]) -> XGBClassifier:
    if XGBClassifier is None:
        raise ModuleNotFoundError(
            "xgboost is not installed. Install it with: pip install xgboost"
        )
    xgb_cols = _numeric_feature_cols(train_df, feature_cols)
    X = train_df[xgb_cols]
    y = train_df['over_2.5'].astype(int).values

    clf = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=max(1, os.cpu_count() or 1),
        random_state=42,
    )
    clf.fit(X, y)
    # attach feature columns used for safety
    setattr(clf, 'feature_cols_', xgb_cols)
    return clf


def _predict_over_proba(model_name: str, model: Any, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    if model_name in {'poisson', 'poisson_opt'}:
        proba = model.predict_proba(df)[:, 1]
        return proba
    if model_name == 'poisson_balanced':
        proba = model.predict_proba(df)[:, 1]
        return proba
    if model_name == 'xgb':
        cols = getattr(model, 'feature_cols_', None)
        if cols is None:
            cols = _numeric_feature_cols(df, feature_cols)
        return model.predict_proba(df[cols])[:, 1]
    raise ValueError(f"Unknown model_name={model_name}")


def _make_plots(results_df: pd.DataFrame, output_dir: Path) -> List[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    df_mean = results_df[results_df['fold'] == 'mean'].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_mean, x='model', y='balanced_accuracy', ax=ax)
    ax.set_title('Balanced Accuracy (mean CV)')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha='right')
    p = output_dir / 'plot_balanced_accuracy.png'
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_mean, x='model', y='specificity', ax=ax)
    ax.set_title('Specificity (mean CV)')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha='right')
    p = output_dir / 'plot_specificity.png'
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_mean, x='specificity', y='sensitivity', hue='model', s=120, ax=ax)
    ax.set_title('Specificity vs Sensitivity (mean CV)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    p = output_dir / 'plot_spec_vs_sens.png'
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_mean, x='model', y='pred_over_rate', ax=ax)
    ax.set_title('Predicted Over Rate (mean CV)')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha='right')
    p = output_dir / 'plot_pred_over_rate.png'
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_mean, x='model', y='calibration_error', ax=ax)
    ax.set_title('Calibration Error (mean CV)')
    plt.xticks(rotation=30, ha='right')
    p = output_dir / 'plot_calibration_error.png'
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_mean, x='model', y='roc_auc', ax=ax)
    ax.set_title('ROC AUC (mean CV)')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha='right')
    p = output_dir / 'plot_roc_auc.png'
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    return paths


def _save_dashboard_html(results_df: pd.DataFrame, html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)

    df_mean = results_df[results_df['fold'] == 'mean'].copy()
    df_mean = df_mean.sort_values(['balanced_accuracy', 'specificity'], ascending=False)

    # Avoid pandas Styler to remove jinja2 dependency
    table_html = df_mean.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Retrain Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 8px; }}
    .meta {{ color: #555; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>Retrain Comparison (CV mean)</h1>
  <div class='meta'>Generated at {datetime.now().isoformat()}</div>
  {table_html}
</body>
</html>
"""
    html_path.write_text(html, encoding='utf-8')


def _select_best_model(results_df: pd.DataFrame) -> str:
    df_mean = results_df[results_df['fold'] == 'mean'].copy()
    df_mean = df_mean[df_mean['specificity'] >= 0.5]
    if len(df_mean) == 0:
        df_mean = results_df[results_df['fold'] == 'mean'].copy()

    df_mean = df_mean.sort_values(['balanced_accuracy', 'specificity'], ascending=False)
    return str(df_mean.iloc[0]['model'])


def _export_production(
    config: Config,
    best_model_name: str,
    best_model_obj: Any,
    feature_cols: List[str],
    feature_metadata: Dict[str, Any],
    comparison: Dict[str, Any],
    results_df: pd.DataFrame,
    log_path: Path,
) -> None:
    prod_dir = config.models_path / 'production'
    prod_dir.mkdir(parents=True, exist_ok=True)

    model_path = prod_dir / 'best_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_obj, f)

    (prod_dir / 'feature_metadata.json').write_text(
        json.dumps(feature_metadata, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    model_card = {
        'created_at': datetime.now().isoformat(),
        'best_model': best_model_name,
        'feature_count': len(feature_cols),
        'feature_names': feature_cols,
        'comparison_v1_v2': comparison,
        'cv_summary': results_df[results_df['fold'] == 'mean'].to_dict(orient='records'),
        'log_file': str(log_path),
    }

    if hasattr(best_model_obj, 'optimal_threshold'):
        model_card['optimal_threshold'] = float(getattr(best_model_obj, 'optimal_threshold'))
    if hasattr(best_model_obj, 'best_alpha'):
        model_card['best_alpha'] = float(getattr(best_model_obj, 'best_alpha'))

    (prod_dir / 'model_card.json').write_text(
        json.dumps(model_card, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    predict_py = prod_dir / 'predict.py'
    predict_py.write_text(
        """import argparse\nimport pickle\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\n\n\ndef main():\n    p = argparse.ArgumentParser(description='Predict Over/Under with exported model')\n    p.add_argument('--model', type=str, default='best_model.pkl')\n    p.add_argument('--input', type=str, required=True, help='CSV file with feature columns')\n    p.add_argument('--output', type=str, default='predictions.csv')\n    args = p.parse_args()\n\n    model_path = Path(args.model)\n    with open(model_path, 'rb') as f:\n        model = pickle.load(f)\n\n    df = pd.read_csv(args.input)\n\n    if not hasattr(model, 'predict_proba'):\n        raise ValueError('Loaded object does not have predict_proba')\n\n    proba = model.predict_proba(df)\n    if hasattr(proba, 'shape') and len(proba.shape) == 2 and proba.shape[1] == 2:\n        p_over = proba[:, 1]\n    else:\n        p_over = np.asarray(proba).reshape(-1)\n\n    p_over = np.clip(p_over, 0.0, 1.0)\n    out = pd.DataFrame({'prob_over': p_over, 'prob_under': 1 - p_over})\n    out.to_csv(args.output, index=False)\n    print(f'Saved predictions to {args.output}')\n\n\nif __name__ == '__main__':\n    main()\n""",
        encoding='utf-8',
    )


def main():
    args = parse_arguments()
    config = Config()
    log_path = _setup_run_logging(config)
    logger = get_logger(__name__)

    results_dir = config.models_path / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / f"retrain_plots_{_timestamp()}"

    checkpoint_path = results_dir / 'retrain_checkpoint.pkl'

    allowed_models = {'poisson', 'poisson_opt', 'poisson_balanced', 'xgb', 'ensemble'}
    selected = [m.strip() for m in args.models.split(',')] if args.models != 'all' else list(allowed_models)
    selected = [m for m in selected if m in allowed_models]

    if XGBClassifier is None:
        if 'xgb' in selected or 'ensemble' in selected:
            logger.warning(
                "xgboost is not installed; disabling xgb/ensemble models. "
                "Install with: pip install xgboost"
            )
        selected = [m for m in selected if m not in {'xgb', 'ensemble'}]

    if args.skip_baseline and 'poisson' in selected:
        selected.remove('poisson')

    if 'poisson_opt' not in selected and 'poisson' in selected:
        pass

    # TimeSeriesSplit requires n_splits >= 2.
    # In --quick mode we run a single (temporal) holdout fold using n_splits=2.
    n_splits = 2 if args.quick else 5

    state: Dict[str, Any] = {}
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        except Exception:
            state = {}

    logger.info("=" * 80)
    logger.info("RETRAIN IMPROVED PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Seasons (when extracting): {args.seasons}")
    logger.info(f"Models: {selected}")
    logger.info(f"CV folds: {n_splits}")

    try:
        if 'data' not in state:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 1: DATA LOADING")
            logger.info("=" * 80)

            processed_files = list(config.data_processed_path.glob('transformed_*.csv'))
            if processed_files:
                latest = max(processed_files, key=lambda p: p.stat().st_mtime)
                df_transformed = pd.read_csv(latest)
                df_transformed['date'] = pd.to_datetime(df_transformed['date'])
                logger.info(f"Loaded processed data from {latest.name} ({len(df_transformed)} rows)")
            else:
                extractor = DataExtractor()
                extractor.run_extraction(sources=['csv'], csv_seasons=args.seasons)

                raw_csv_files = list((config.data_raw_path / 'csv').glob('*.csv'))
                if not raw_csv_files:
                    raise FileNotFoundError('No raw CSV files found after extraction')
                latest_raw = max(raw_csv_files, key=lambda p: p.stat().st_mtime)
                df_raw = pd.read_csv(latest_raw)
                transformer = DataTransformer()
                df_transformed, _ = transformer.transform(df_raw, source='csv')

            df_transformed = _ensure_targets(df_transformed)

            issues = _basic_integrity_checks(df_transformed)
            if issues:
                logger.warning(f"Data integrity issues: {issues}")

            logger.info(f"Date range: {df_transformed['date'].min()} -> {df_transformed['date'].max()}")
            logger.info(f"Matches: {len(df_transformed)}")
            logger.info(f"Over rate: {df_transformed['over_2.5'].mean():.2%}")
            logger.info(f"Home goals mean: {df_transformed['home_goals'].mean():.3f}")
            logger.info(f"Away goals mean: {df_transformed['away_goals'].mean():.3f}")

            state['data'] = {'df_transformed': df_transformed}
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)

        df_transformed = state['data']['df_transformed']

        if 'features' not in state:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 2: FEATURE ENGINEERING V2")
            logger.info("=" * 80)

            df_features, feature_cols, feature_metadata = _get_feature_sets(df_transformed)
            comparison = _compare_with_v1(df_transformed, df_features)

            logger.info(f"V2 features: {len(feature_cols)}")
            logger.info(f"V1 features: {comparison['v1_count']}")
            logger.info(f"Added: {len(comparison['added'])}, Removed: {len(comparison['removed'])}")

            results_dir.mkdir(parents=True, exist_ok=True)
            # Also export a copy to results for quick inspection
            (results_dir / 'feature_metadata.json').write_text(
                json.dumps(feature_metadata, indent=2, ensure_ascii=False), encoding='utf-8'
            )

            state['features'] = {
                'df_features': df_features,
                'feature_cols': feature_cols,
                'feature_metadata': feature_metadata,
                'comparison': comparison,
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)

        df_features = state['features']['df_features']
        feature_cols = state['features']['feature_cols']
        feature_metadata = state['features']['feature_metadata']
        comparison = state['features']['comparison']

        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3-4: TRAIN + COMPREHENSIVE EVALUATION")
        logger.info("=" * 80)

        df_features = df_features.sort_values('date').reset_index(drop=True)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        folds = list(tscv.split(df_features))
        if args.quick:
            folds = [folds[-1]]

        all_rows: List[Dict[str, Any]] = []

        if tqdm is None:
            class _NoTqdm:
                def __init__(self, total: int):
                    self.total = int(total)
                    self.n = 0

                def update(self, k: int = 1):
                    self.n += int(k)

                def close(self):
                    pass

            outer_bar = _NoTqdm(total=len(selected) * len(folds))
        else:
            outer_bar = tqdm(total=len(selected) * len(folds), desc='Models x Folds', unit='fold')

        for model_name in selected:
            for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
                train_df = df_features.iloc[train_idx].copy()
                test_df = df_features.iloc[test_idx].copy()

                y_true = test_df['over_2.5'].astype(int).values

                if model_name == 'poisson':
                    model = _train_poisson_original(train_df, feature_cols)
                    prob = _predict_over_proba('poisson', model, test_df, feature_cols)
                    thr = 0.5
                    metrics = _eval_binary_metrics(y_true, prob, thr)

                elif model_name == 'poisson_opt':
                    model = _train_poisson_original(train_df, feature_cols)
                    prob_train = _predict_over_proba('poisson', model, train_df, feature_cols)
                    y_train = train_df['over_2.5'].astype(int).values
                    thr = _youden_optimal_threshold(y_train, prob_train)
                    prob = _predict_over_proba('poisson', model, test_df, feature_cols)
                    metrics = _eval_binary_metrics(y_true, prob, thr)
                    metrics['threshold'] = float(thr)

                elif model_name == 'poisson_balanced':
                    model = _train_poisson_balanced(train_df, feature_cols)
                    prob = _predict_over_proba('poisson_balanced', model, test_df, feature_cols)
                    thr = getattr(model, 'optimal_threshold', 0.5)
                    metrics = _eval_binary_metrics(y_true, prob, float(thr))
                    metrics['threshold'] = float(thr)

                elif model_name == 'xgb':
                    model = _train_xgb(train_df, feature_cols)
                    prob = _predict_over_proba('xgb', model, test_df, feature_cols)
                    thr = 0.5
                    metrics = _eval_binary_metrics(y_true, prob, thr)

                elif model_name == 'ensemble':
                    m_p = _train_poisson_balanced(train_df, feature_cols)
                    m_x = _train_xgb(train_df, feature_cols)

                    p_p = _predict_over_proba('poisson_balanced', m_p, test_df, feature_cols)
                    p_x = _predict_over_proba('xgb', m_x, test_df, feature_cols)

                    prob = np.clip(0.5 * p_p + 0.5 * p_x, 0.0, 1.0)
                    thr = float(getattr(m_p, 'optimal_threshold', 0.5))
                    metrics = _eval_binary_metrics(y_true, prob, thr)
                    metrics['threshold'] = float(thr)

                else:
                    raise ValueError(model_name)

                row = {
                    'model': model_name,
                    'fold': fold_idx,
                    'train_samples': int(len(train_df)),
                    'test_samples': int(len(test_df)),
                }
                row.update(metrics)
                all_rows.append(row)

                outer_bar.update(1)

        outer_bar.close()

        results_df = pd.DataFrame(all_rows)

        summary_rows: List[Dict[str, Any]] = []
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
        # Never aggregate these identifiers
        for col in ['fold']:
            if col in numeric_cols:
                numeric_cols.remove(col)

        for model_name in selected:
            sub = results_df[results_df['model'] == model_name]
            mean = sub[numeric_cols].mean().to_dict() if numeric_cols else {}
            std = sub[numeric_cols].std().to_dict() if numeric_cols else {}
            summary_rows.append({'model': model_name, **mean, 'fold': 'mean'})
            summary_rows.append({'model': model_name, **std, 'fold': 'std'})

        results_df = pd.concat([results_df, pd.DataFrame(summary_rows)], ignore_index=True)

        csv_path = results_dir / 'retrain_comparison.csv'
        results_df.to_csv(csv_path, index=False)

        html_path = results_dir / 'retrain_comparison.html'
        _save_dashboard_html(results_df, html_path)

        plot_paths = _make_plots(results_df, plots_dir)

        logger.info(f"Saved CSV: {csv_path}")
        logger.info(f"Saved HTML: {html_path}")
        logger.info(f"Saved plots dir: {plots_dir}")

        logger.info("\n" + "=" * 80)
        logger.info("STAGE 5: COMPARISON & SELECTION")
        logger.info("=" * 80)

        best_name = _select_best_model(results_df)
        logger.info(f"Selected best model: {best_name}")

        logger.info("\n" + "=" * 80)
        logger.info("STAGE 6: PRODUCTION EXPORT")
        logger.info("=" * 80)

        full_train = df_features.copy()

        if best_name == 'poisson':
            best_obj = _train_poisson_original(full_train, feature_cols)
        elif best_name == 'poisson_opt':
            best_obj = _train_poisson_original(full_train, feature_cols)
            prob_train = _predict_over_proba('poisson', best_obj, full_train, feature_cols)
            y_train = full_train['over_2.5'].astype(int).values
            best_obj.optimal_threshold = _youden_optimal_threshold(y_train, prob_train)
        elif best_name == 'poisson_balanced':
            best_obj = _train_poisson_balanced(full_train, feature_cols)
        elif best_name == 'xgb':
            best_obj = _train_xgb(full_train, feature_cols)
        elif best_name == 'ensemble':
            best_obj = EnsembleOverUnderModel(
                poisson_balanced=_train_poisson_balanced(full_train, feature_cols),
                xgb=_train_xgb(full_train, feature_cols),
                feature_cols=feature_cols,
                weight_poisson=0.5,
                weight_xgb=0.5,
            )
        else:
            raise ValueError(best_name)

        _export_production(
            config,
            best_name,
            best_obj,
            feature_cols,
            feature_metadata,
            comparison,
            results_df,
            log_path,
        )

        state['final'] = {
            'results_csv': str(csv_path),
            'results_html': str(html_path),
            'plots_dir': str(plots_dir),
            'best_model': best_name,
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(state, f)

        logger.info("\n" + "=" * 80)
        logger.info("RETRAIN COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Retrain failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
