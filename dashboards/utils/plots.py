from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import binom


def gauge(title: str, value: float, delta: Optional[float] = None, value_format: str = '.3f', domain_y=(0, 1)):
    d = None
    if delta is not None and np.isfinite(delta):
        d = {'reference': value - delta, 'increasing': {'color': '#2E7D32'}, 'decreasing': {'color': '#C62828'}}

    fig = go.Figure(
        go.Indicator(
            mode='gauge+number' + ('+delta' if d else ''),
            value=value,
            number={'valueformat': value_format},
            title={'text': title},
            delta=d,
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': '#1f77b4'},
            },
            domain={'x': [0, 1], 'y': list(domain_y)},
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=220)
    return fig


def prob_histogram(prob_over: np.ndarray, y_true: Optional[np.ndarray] = None, threshold: float = 0.5):
    prob_over = np.asarray(prob_over).reshape(-1)
    df = pd.DataFrame({'prob_over': prob_over})
    if y_true is not None:
        df['y_true'] = np.asarray(y_true).reshape(-1).astype(int)
        df['class'] = df['y_true'].map({0: 'Under', 1: 'Over'})
        fig = px.histogram(df, x='prob_over', color='class', nbins=40, barmode='overlay', opacity=0.6)
    else:
        fig = px.histogram(df, x='prob_over', nbins=40)

    fig.add_vline(x=float(threshold), line_width=2, line_dash='dash', line_color='red')
    fig.update_layout(xaxis_title='P(Over)', yaxis_title='Count')
    return fig


def confusion_heatmap(tn: int, fp: int, fn: int, tp: int):
    z = [[tn, fp], [fn, tp]]
    fig = go.Figure(data=go.Heatmap(z=z, x=['Pred Under', 'Pred Over'], y=['True Under', 'True Over'], colorscale='Blues'))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=320)
    return fig


def calibration_curve(prob: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> Tuple[go.Figure, pd.DataFrame]:
    prob = np.asarray(prob).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob >= lo) & (prob < hi if i < n_bins - 1 else prob <= hi)
        if not mask.any():
            continue
        p_mean = float(np.mean(prob[mask]))
        y_mean = float(np.mean(y_true[mask]))
        n = int(mask.sum())

        # Wilson-ish CI via binomial (normal approx via exact quantiles)
        alpha = 0.05
        ci_low = float(binom.ppf(alpha / 2, n, y_mean) / n) if n > 0 else y_mean
        ci_high = float(binom.ppf(1 - alpha / 2, n, y_mean) / n) if n > 0 else y_mean
        rows.append({'p_mean': p_mean, 'y_mean': y_mean, 'n': n, 'ci_low': ci_low, 'ci_high': ci_high})

    df = pd.DataFrame(rows).sort_values('p_mean')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect', line=dict(dash='dash')))

    if not df.empty:
        fig.add_trace(go.Scatter(x=df['p_mean'], y=df['y_mean'], mode='lines+markers', name='Model'))
        fig.add_trace(go.Scatter(
            x=df['p_mean'], y=df['ci_high'], mode='lines', name='CI high', line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df['p_mean'], y=df['ci_low'], mode='lines', name='CI', fill='tonexty', line=dict(width=0),
            fillcolor='rgba(31, 119, 180, 0.15)'
        ))

    fig.update_layout(xaxis_title='Mean predicted P(Over)', yaxis_title='Observed frequency', height=420)
    return fig, df


def roc_curve_plot(curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    for name, (fpr, tpr, auc) in curves.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={auc:.3f})"))
    fig.update_layout(xaxis_title='FPR', yaxis_title='TPR', height=420)
    return fig
