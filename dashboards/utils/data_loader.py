from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class LoadedData:
    df: pd.DataFrame
    source: str


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _expand_glob(path_pattern: str) -> Optional[str]:
    matches = glob.glob(path_pattern)
    if not matches:
        return None
    return sorted(matches)[-1]


def load_dataset_from_path(path_or_glob: str) -> LoadedData:
    if any(ch in path_or_glob for ch in ['*', '?', '[']):
        resolved = _expand_glob(path_or_glob)
        if resolved is None:
            raise FileNotFoundError(f"No files matched glob: {path_or_glob}")
        path_or_glob = resolved

    path = Path(path_or_glob)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    elif path.suffix.lower() in {'.parquet', '.pq'}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    return LoadedData(df=df, source=str(path))


def load_dataset_from_upload(uploaded_file) -> LoadedData:
    # uploaded_file is a Streamlit UploadedFile
    name = getattr(uploaded_file, 'name', 'uploaded')
    lower = name.lower()

    if lower.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif lower.endswith('.parquet') or lower.endswith('.pq'):
        df = pd.read_parquet(uploaded_file)
    else:
        raise ValueError('Upload must be a .csv or .parquet file')

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    return LoadedData(df=df, source=f"upload:{name}")


def list_models_in_dir(models_dir: str) -> List[str]:
    p = Path(models_dir)
    if not p.exists():
        return []
    files = sorted([f.name for f in p.glob('*.pkl')])
    return files


def load_model(models_dir: str, model_file: str) -> Any:
    path = Path(models_dir) / model_file
    if not path.exists():
        raise FileNotFoundError(str(path))
    with open(path, 'rb') as f:
        return pickle.load(f)


def infer_feature_cols(model: Any, df: pd.DataFrame) -> List[str]:
    # Prefer model-provided columns
    if hasattr(model, 'feature_cols') and getattr(model, 'feature_cols') is not None:
        cols = list(getattr(model, 'feature_cols'))
        return [c for c in cols if c in df.columns]
    if hasattr(model, 'feature_cols_'):
        cols = list(getattr(model, 'feature_cols_'))
        return [c for c in cols if c in df.columns]

    # Fallback: numeric columns excluding obvious targets/metadata
    exclude = {
        'home_goals', 'away_goals', 'total_goals',
        'over_0.5', 'over_1.5', 'over_2.5', 'over_3.5', 'over_4.5',
        'date', 'season', 'home_team', 'away_team', 'home_team_name', 'away_team_name',
    }
    cols = [c for c in df.columns if c not in exclude]
    X = df[cols]
    return X.select_dtypes(include=[np.number, bool]).columns.tolist()


def predict_over_proba(model: Any, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
    # For Poisson-style models we can pass the full df (they will pick self.feature_cols)
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(df)
        except Exception:
            # Some sklearn models require explicit X
            if feature_cols is None:
                feature_cols = infer_feature_cols(model, df)
            proba = model.predict_proba(df[feature_cols])

        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.reshape(-1)

    raise ValueError('Model does not implement predict_proba')


def filter_df(
    df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    seasons: Optional[List[str]] = None,
    gameweeks: Optional[List[int]] = None,
    teams: Optional[List[str]] = None,
) -> pd.DataFrame:
    out = df.copy()
    if 'date' in out.columns:
        if start_date is not None:
            out = out[out['date'] >= start_date]
        if end_date is not None:
            out = out[out['date'] <= end_date]

    if seasons and 'season' in out.columns:
        out = out[out['season'].astype(str).isin([str(s) for s in seasons])]

    if gameweeks and 'gameweek' in out.columns:
        out = out[out['gameweek'].isin(gameweeks)]

    if teams:
        team_cols = [c for c in ['home_team', 'away_team', 'home_team_name', 'away_team_name'] if c in out.columns]
        if team_cols:
            mask = False
            for c in team_cols:
                mask = mask | out[c].astype(str).isin(teams)
            out = out[mask]

    return out
