from __future__ import annotations

import io
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboards.utils.data_loader import (
    filter_df,
    infer_feature_cols,
    list_models_in_dir,
    load_config,
    load_dataset_from_path,
    load_dataset_from_upload,
    load_model,
    predict_over_proba,
)
from dashboards.utils.metrics import brier_decomposition, compute_metrics, ks_test_by_class
from dashboards.utils.plots import calibration_curve, confusion_heatmap, gauge, prob_histogram, roc_curve_plot


CONFIG_PATH = str(Path(__file__).parent / "config.yaml")


def _get_team_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    home = next((c for c in ["home_team", "home_team_name"] if c in df.columns), None)
    away = next((c for c in ["away_team", "away_team_name"] if c in df.columns), None)
    return home, away


def _safe_threshold(model: Any, default: float = 0.5) -> float:
    for attr in ["optimal_threshold", "threshold_optimal", "best_threshold"]:
        if hasattr(model, attr):
            try:
                return float(getattr(model, attr))
            except Exception:
                pass
    return float(default)


def _ensure_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "total_goals" not in out.columns and all(c in out.columns for c in ["home_goals", "away_goals"]):
        out["total_goals"] = out["home_goals"] + out["away_goals"]
    if "over_2.5" not in out.columns and "total_goals" in out.columns:
        out["over_2.5"] = (out["total_goals"] > 2.5).astype(int)
    return out


def _download_df_csv(df: pd.DataFrame, filename: str):
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def _baseline_from_retrain_csv(path: str, model_key: str) -> Optional[Dict[str, float]]:
    # Optional: uses models/results/retrain_comparison.csv if available
    try:
        p = Path(path)
        if not p.exists():
            return None
        df = pd.read_csv(p)
        if "fold" in df.columns:
            df = df[df["fold"] == "mean"]
        if "model" in df.columns:
            df = df[df["model"].astype(str) == str(model_key)]
        if df.empty:
            return None
        r = df.iloc[0].to_dict()
        return {
            "pred_over_pct": float(r.get("pred_over_rate", np.nan)) * 100.0,
            "balanced_accuracy": float(r.get("balanced_accuracy", np.nan)),
            "specificity": float(r.get("specificity", np.nan)),
            "sensitivity": float(r.get("sensitivity", np.nan)),
        }
    except Exception:
        return None


def _kpi_gauges(current: Dict[str, float], baseline: Optional[Dict[str, float]]):
    cols = st.columns(4)

    def d(key: str) -> Optional[float]:
        if baseline is None or key not in baseline:
            return None
        try:
            return float(current[key]) - float(baseline[key])
        except Exception:
            return None

    with cols[0]:
        st.plotly_chart(
            gauge("% Pred Over", current["pred_over_pct"] / 100.0, delta=(d("pred_over_pct") / 100.0 if baseline else None), value_format=".2f"),
            use_container_width=True,
        )
    with cols[1]:
        st.plotly_chart(gauge("Balanced Acc", current["balanced_accuracy"], delta=d("balanced_accuracy")), use_container_width=True)
    with cols[2]:
        st.plotly_chart(gauge("Specificity", current["specificity"], delta=d("specificity")), use_container_width=True)
    with cols[3]:
        st.plotly_chart(gauge("Sensitivity", current["sensitivity"], delta=d("sensitivity")), use_container_width=True)


def _maybe_autorefresh(minutes: int):
    # Optional dependency: streamlit-autorefresh
    if minutes <= 0:
        return
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=minutes * 60 * 1000, key=f"autorefresh_{minutes}")
    except Exception:
        st.info("Auto-refresh requiere 'streamlit-autorefresh' (opcional).")


def _export_pdf_report(title: str, sections: Dict[str, str]) -> Optional[bytes]:
    # Optional dependency: reportlab
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        y = height - 50
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, title)
        y -= 24
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Generated at: {datetime.now().isoformat()}")
        y -= 24

        c.setFont("Helvetica", 10)
        for name, text in sections.items():
            if y < 100:
                c.showPage()
                y = height - 50
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, name)
            y -= 16
            c.setFont("Helvetica", 9)
            for line in (text or "").splitlines():
                if y < 60:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 9)
                c.drawString(50, y, line[:120])
                y -= 12
            y -= 10

        c.save()
        buffer.seek(0)
        return buffer.read()
    except Exception:
        return None


def load_source_metadata():
    """Load source metadata from predictions."""
    import json
    from pathlib import Path

    metadata_path = Path("data/predictions/source_metadata.json")

    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def render_simulation_warning():
    """Render warning banner for SIMULATION mode."""
    metadata = load_source_metadata()

    if metadata is None:
        st.info("ℹ️ No predictions loaded. Run `python scripts/build_prediction_dataset.py` first.")
        return

    source_type = metadata.get('source_type', 'UNKNOWN')

    if source_type == "SIMULATION":
        st.warning(
            f"""
            ⚠️ **SIMULATION MODE ACTIVE**

            **Fixtures displayed are SYNTHETIC and generated from historical data.**

            These matches do NOT represent the official Premier League schedule.

            ---

            **Intended Use:**
            - Model experimentation
            - Backtesting strategies
            - Development and testing

            **NOT for:**
            - Real-world predictions
            - Betting or decision-making
            - Official fixture reference

            ---

            **Source Details:**
            - Provider: `{metadata.get('provider', 'unknown')}`
            - Confidence: `{metadata.get('confidence_score', 0.0):.0%}`
            - Last Updated: `{metadata.get('last_updated', 'unknown')}`
            - Total Fixtures: `{metadata.get('total_fixtures', 0)}`

            ---

            💡 **To use official fixtures:** Wait for Phase 2 (API integration)
            """,
            icon="⚠️"
        )
    elif source_type == "REAL_FIXTURES":
        st.success(
            f"""
            ✅ **OFFICIAL FIXTURES MODE**

            Fixtures are from official Premier League API.

            ---

            **Source:** `{metadata.get('provider', 'unknown')}`
            **Confidence:** `{metadata.get('confidence_score', 1.0):.0%}`
            **Last Updated:** `{metadata.get('last_updated', 'unknown')}`
            """,
            icon="✅"
        )
    else:
        st.error(f"❌ Unknown source type: {source_type}")


def main():
    cfg = load_config(CONFIG_PATH)

    st.set_page_config(page_title=cfg.get("app", {}).get("title", "Bias Monitor"), layout="wide")
    st.title(cfg.get("app", {}).get("title", "Model Bias Monitor"))

    # PHASE 1: SIMULATION MODE WARNING BANNER
    render_simulation_warning()
    st.markdown("---")

    auto_refresh = int(cfg.get("app", {}).get("auto_refresh_minutes", 0) or 0)
    _maybe_autorefresh(auto_refresh)

    models_dir = cfg.get("models", {}).get("production_dir", "./models/production")
    model_files = list_models_in_dir(models_dir)

    if not model_files:
        st.error(f"No models found in {models_dir}.")
        return

    with st.sidebar:
        st.header("Configuración")

        default_model = cfg.get("models", {}).get("default_model_file", "best_model.pkl")
        default_idx = model_files.index(default_model) if default_model in model_files else 0
        model_file = st.selectbox("Modelo", options=model_files, index=default_idx)

        st.subheader("Datos")
        dataset_path = st.text_input(
            "Dataset path (CSV/Parquet o glob)",
            value=str(cfg.get("data", {}).get("default_dataset_path", "")),
        )
        uploaded = st.file_uploader("...o sube un archivo", type=["csv", "parquet", "pq"])

        st.subheader("Filtros")
        refresh = st.button("Refresh Data")

    @st.cache_resource(show_spinner=False)
    def _cached_model(mdir: str, mfile: str):
        return load_model(mdir, mfile)

    model = _cached_model(models_dir, model_file)

    if uploaded is not None:
        loaded = load_dataset_from_upload(uploaded)
    else:
        if not dataset_path:
            st.warning("Debes indicar un dataset path o subir un archivo.")
            return
        loaded = load_dataset_from_path(dataset_path)

    df = _ensure_targets(loaded.df)
    if "date" not in df.columns:
        st.error("El dataset debe contener columna 'date'.")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    home_col, away_col = _get_team_columns(df)
    season_col = "season" if "season" in df.columns else None
    gw_col = "gameweek" if "gameweek" in df.columns else None

    with st.sidebar:
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        d0, d1 = st.date_input("Período", value=(min_date, max_date))

        teams = []
        if home_col and away_col:
            all_teams = sorted(set(df[home_col].astype(str).unique()) | set(df[away_col].astype(str).unique()))
            teams = st.multiselect("Equipos", options=all_teams, default=[])

        seasons = []
        if season_col:
            season_opts = sorted(df[season_col].astype(str).dropna().unique().tolist())
            seasons = st.multiselect("Temporada", options=season_opts, default=[])

        gws = []
        if gw_col:
            gw_opts = sorted(df[gw_col].dropna().unique().tolist())
            gws = st.multiselect("Jornada", options=gw_opts, default=[])

    df_f = filter_df(
        df,
        start_date=pd.Timestamp(d0),
        end_date=pd.Timestamp(d1),
        seasons=seasons or None,
        gameweeks=gws or None,
        teams=teams or None,
    )

    st.caption(f"Data source: {loaded.source} | Filtered rows: {len(df_f)}")
    if len(df_f) < 10:
        st.warning("Muy pocos registros tras filtros.")
        return

    # Predict
    feature_cols = infer_feature_cols(model, df_f)
    prob_over = predict_over_proba(model, df_f, feature_cols=feature_cols)
    prob_over = np.clip(prob_over, 0.0, 1.0)

    threshold = _safe_threshold(model, default=float(cfg.get("app", {}).get("default_threshold", 0.5)))

    y_true = df_f["over_2.5"].astype(int).values if "over_2.5" in df_f.columns else None

    # Baseline comparison (optional)
    baseline_csv = cfg.get("models", {}).get("baseline_metrics_csv", "")
    baseline = _baseline_from_retrain_csv(baseline_csv, model_key=model_file.replace(".pkl", ""))

    tabs = st.tabs(
        [
            "Tab 1: Overview",
            "Tab 2: Prediction Distribution",
            "Tab 3: Confusion Matrix Temporal",
            "Tab 4: Calibration Analysis",
            "Tab 5: Feature Drift",
            "Tab 6: Error Analysis",
            "Tab 7: Model Comparison",
        ]
    )

    # ---------------- TAB 1 ----------------
    with tabs[0]:
        st.subheader("Overview")

        if y_true is None:
            y_pred = (prob_over >= threshold).astype(int)
            current = {
                "pred_over_pct": float(np.mean(y_pred) * 100.0),
                "balanced_accuracy": float("nan"),
                "specificity": float("nan"),
                "sensitivity": float("nan"),
            }
            _kpi_gauges(current, baseline=None)
            st.info("Sin columna 'over_2.5' no se calculan métricas de clasificación.")
        else:
            m = compute_metrics(y_true, prob_over, threshold=threshold)
            current = {
                "pred_over_pct": m.pred_over_pct,
                "balanced_accuracy": m.balanced_accuracy,
                "specificity": m.specificity,
                "sensitivity": m.sensitivity,
            }
            _kpi_gauges(current, baseline=baseline)

            st.write(
                {
                    "roc_auc": m.roc_auc,
                    "ece": m.ece,
                    "brier": m.brier,
                    "threshold_used": threshold,
                }
            )

        st.divider()
        st.subheader("Downloads")
        scored = df_f.copy()
        scored["prob_over"] = prob_over
        scored["prob_under"] = 1.0 - prob_over
        _download_df_csv(scored, "bias_monitor_scored.csv")

        sections = {
            "Overview": f"Rows: {len(scored)}\nThreshold: {threshold:.3f}\nModel: {model_file}\n",
        }
        pdf_bytes = _export_pdf_report("Bias Monitor Report", sections)
        if pdf_bytes is not None:
            st.download_button("Export PDF report", data=pdf_bytes, file_name="bias_report.pdf", mime="application/pdf")
        else:
            st.caption("PDF export requiere 'reportlab' (opcional): pip install reportlab")

    # ---------------- TAB 2 ----------------
    with tabs[1]:
        st.subheader("Prediction Distribution")
        if y_true is not None:
            ks = ks_test_by_class(prob_over, y_true)
            st.write({"KS_stat": ks["statistic"], "p_value": ks["p_value"], "threshold": threshold})
        st.plotly_chart(prob_histogram(prob_over, y_true=y_true, threshold=threshold), use_container_width=True)

    # ---------------- TAB 3 ----------------
    with tabs[2]:
        st.subheader("Confusion Matrix Temporal")
        if y_true is None:
            st.info("Requiere 'over_2.5'.")
        else:
            tmp = pd.DataFrame({"date": df_f["date"], "y_true": y_true, "prob_over": prob_over})
            tmp["y_pred"] = (tmp["prob_over"] >= threshold).astype(int)

            freq = cfg.get("ui", {}).get("temporal_aggregation", "W")
            tmp["bucket"] = tmp["date"].dt.to_period(freq).dt.to_timestamp()
            buckets = sorted(tmp["bucket"].unique().tolist())
            idx = st.slider("Time bucket", 0, max(0, len(buckets) - 1), max(0, len(buckets) - 1))
            b = buckets[idx]
            sub = tmp[tmp["bucket"] == b]

            tn, fp, fn, tp = (0, 0, 0, 0)
            if len(sub) > 0:
                tn = int(((sub["y_true"] == 0) & (sub["y_pred"] == 0)).sum())
                fp = int(((sub["y_true"] == 0) & (sub["y_pred"] == 1)).sum())
                fn = int(((sub["y_true"] == 1) & (sub["y_pred"] == 0)).sum())
                tp = int(((sub["y_true"] == 1) & (sub["y_pred"] == 1)).sum())

            st.plotly_chart(confusion_heatmap(tn, fp, fn, tp), use_container_width=True)

            rows = []
            for bb, ss in tmp.groupby("bucket"):
                if len(ss) < 10:
                    continue
                tn2 = int(((ss["y_true"] == 0) & (ss["y_pred"] == 0)).sum())
                fp2 = int(((ss["y_true"] == 0) & (ss["y_pred"] == 1)).sum())
                fn2 = int(((ss["y_true"] == 1) & (ss["y_pred"] == 0)).sum())
                tp2 = int(((ss["y_true"] == 1) & (ss["y_pred"] == 1)).sum())
                spec = tn2 / (tn2 + fp2) if (tn2 + fp2) else 0.0
                sens = tp2 / (tp2 + fn2) if (tp2 + fn2) else 0.0
                rows.append({"bucket": bb, "specificity": spec, "sensitivity": sens, "n": len(ss)})
            trend = pd.DataFrame(rows)

            if not trend.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=trend["bucket"], y=trend["specificity"], mode="lines+markers", name="Specificity"))
                fig.add_trace(go.Scatter(x=trend["bucket"], y=trend["sensitivity"], mode="lines+markers", name="Sensitivity"))
                fig.update_layout(height=360, yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig, use_container_width=True)

            alert_thr = float(cfg.get("ui", {}).get("alert_specificity_threshold", 0.45))
            last_n = int(cfg.get("ui", {}).get("alert_last_n", 10))
            recent = tmp.sort_values("date").tail(last_n)
            if len(recent) >= 10:
                tn_r = int(((recent["y_true"] == 0) & (recent["y_pred"] == 0)).sum())
                fp_r = int(((recent["y_true"] == 0) & (recent["y_pred"] == 1)).sum())
                spec_r = tn_r / (tn_r + fp_r) if (tn_r + fp_r) else 0.0
                if spec_r < alert_thr:
                    st.error(f"ALERTA: Specificity {spec_r:.2%} en últimos {last_n} partidos (< {alert_thr:.0%})")

    # ---------------- TAB 4 ----------------
    with tabs[3]:
        st.subheader("Calibration Analysis")
        if y_true is None:
            st.info("Requiere 'over_2.5'.")
        else:
            fig, bins_df = calibration_curve(prob_over, y_true, n_bins=10)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(bins_df, use_container_width=True)

            dec = brier_decomposition(y_true, prob_over)
            st.write("Brier Score decomposition:", dec)

    # ---------------- TAB 5 ----------------
    with tabs[4]:
        st.subheader("Feature Drift")
        st.caption("Drift KS test: referencia = primer 60% vs ventana reciente (último 40%).")

        df_sorted = df_f.sort_values("date").reset_index(drop=True)
        split = int(len(df_sorted) * 0.6)
        ref = df_sorted.iloc[:split]
        cur = df_sorted.iloc[split:]

        numeric_cols = df_sorted[feature_cols].select_dtypes(include=[np.number, bool]).columns.tolist()
        drift_rows = []
        for c in numeric_cols:
            a = ref[c].dropna().values
            b = cur[c].dropna().values
            if len(a) < 30 or len(b) < 30:
                continue
            stat, pval = ks_2samp(a, b)
            drift_rows.append({"feature": c, "ks_stat": float(stat), "p_value": float(pval)})

        drift = pd.DataFrame(drift_rows)
        if drift.empty:
            st.info("No hay suficientes datos para drift.")
        else:
            drift = drift.sort_values("p_value")
            st.dataframe(drift.head(10), use_container_width=True)
            n_alert = int((drift["p_value"] < 0.05).sum())
            if n_alert:
                st.warning(f"ALERTA: {n_alert} features con drift (p < 0.05)")

    # ---------------- TAB 6 ----------------
    with tabs[5]:
        st.subheader("Error Analysis")
        if y_true is None:
            st.info("Requiere 'over_2.5'.")
        else:
            err = df_f.copy()
            err["prob_over"] = prob_over
            err["y_true"] = y_true
            err["y_pred"] = (err["prob_over"] >= threshold).astype(int)
            err["error_type"] = "OK"
            err.loc[(err["y_true"] == 0) & (err["y_pred"] == 1), "error_type"] = "FP"
            err.loc[(err["y_true"] == 1) & (err["y_pred"] == 0), "error_type"] = "FN"

            bad = err[err["error_type"].isin(["FP", "FN"])].sort_values("date", ascending=False)
            st.write(f"Mis-predictions: {len(bad)}")

            cols = ["date", "prob_over", "y_true", "y_pred", "error_type"]
            if home_col:
                cols.insert(1, home_col)
            if away_col:
                cols.insert(2, away_col)
            if "total_goals" in err.columns:
                cols.insert(3, "total_goals")

            st.dataframe(bad[cols].head(50), use_container_width=True)

            if home_col and away_col and not bad.empty:
                team_err = pd.concat(
                    [
                        bad[[home_col]].rename(columns={home_col: "team"}),
                        bad[[away_col]].rename(columns={away_col: "team"}),
                    ]
                )
                st.subheader("Equipos con mayor error")
                st.bar_chart(team_err["team"].value_counts().head(15))

    # ---------------- TAB 7 ----------------
    with tabs[6]:
        st.subheader("Model Comparison")
        st.caption("Selecciona hasta 3 modelos para comparar.")

        compare = st.multiselect("Modelos", options=model_files, default=[model_file], max_selections=3)
        if len(compare) == 0:
            st.info("Selecciona al menos un modelo.")
        else:
            curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
            metrics_rows = []

            for mf in compare:
                m2 = _cached_model(models_dir, mf)
                p2 = np.clip(predict_over_proba(m2, df_f, feature_cols=infer_feature_cols(m2, df_f)), 0.0, 1.0)

                if y_true is not None and len(np.unique(y_true)) > 1:
                    fpr, tpr, _ = roc_curve(y_true, p2)
                    auc = float(roc_auc_score(y_true, p2))
                    curves[mf] = (fpr, tpr, auc)

                    cm = compute_metrics(y_true, p2, threshold=_safe_threshold(m2, threshold))
                    metrics_rows.append(
                        {
                            "model": mf,
                            "pred_over_pct": cm.pred_over_pct,
                            "balanced_accuracy": cm.balanced_accuracy,
                            "specificity": cm.specificity,
                            "sensitivity": cm.sensitivity,
                            "roc_auc": cm.roc_auc,
                            "ece": cm.ece,
                            "brier": cm.brier,
                        }
                    )
                else:
                    metrics_rows.append({"model": mf, "pred_over_pct": float(np.mean(p2 >= 0.5) * 100.0)})

            if y_true is not None and curves:
                st.plotly_chart(roc_curve_plot(curves), use_container_width=True)

            if metrics_rows:
                st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)

            # Agreement matrix
            st.subheader("Agreement matrix")
            if len(compare) >= 2:
                preds = {}
                for mf in compare:
                    m2 = _cached_model(models_dir, mf)
                    p2 = np.clip(predict_over_proba(m2, df_f, feature_cols=infer_feature_cols(m2, df_f)), 0.0, 1.0)
                    thr = _safe_threshold(m2, threshold)
                    preds[mf] = (p2 >= thr).astype(int)

                names = list(preds.keys())
                mat = np.zeros((len(names), len(names)))
                for i, a in enumerate(names):
                    for j, b in enumerate(names):
                        mat[i, j] = float(np.mean(preds[a] == preds[b]))

                fig = go.Figure(data=go.Heatmap(z=mat, x=names, y=names, colorscale="Viridis", zmin=0, zmax=1))
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
