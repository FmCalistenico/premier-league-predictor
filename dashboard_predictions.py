"""
Dashboard de Predicciones - Premier League Over/Under 2.5
==========================================================

Visualiza predicciones semanales con métricas de probabilidad y confianza.

Uso:
    streamlit run dashboard_predictions.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from pathlib import Path
import numpy as np
import datetime as dt

import yaml
from dotenv import load_dotenv

from src.data.sources.fixture_api import FixtureAPISource


predictions_path = Path('data/predictions/predictions_jan_5_11_2026.csv')
predictions_mtime = predictions_path.stat().st_mtime if predictions_path.exists() else None

if predictions_mtime is None:
    last_update = "(archivo no encontrado)"
else:
    last_update = dt.datetime.fromtimestamp(predictions_mtime).strftime('%d/%m/%Y %H:%M')

# Configuración de la página
st.set_page_config(
    page_title="Premier League Predictions - GW 20-21",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    .confidence-high {
        color: #00CC66;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FFA500;
        font-weight: bold;
    }
    .confidence-low {
        color: #FF4444;
        font-weight: bold;
    }
    .prediction-over {
        background-color: #E8F5E9;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
    }
    .prediction-under {
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("⚽ Predicciones Premier League - Semana 5-11 Enero 2026")
st.markdown("### Gameweeks 20-21 | Sistema de Predicción Over/Under 2.5 Goles")

# Sidebar con información del sistema
with st.sidebar:
    st.header("📊 Información del Sistema")

    load_dotenv()
    mode_env = os.getenv('PREDICTION_MODE', '(no seteado)')
    api_key_present = bool(os.getenv('API_FOOTBALL_KEY')) and os.getenv('API_FOOTBALL_KEY') != 'your_rapidapi_key_here'

    st.caption(f"Modo (env): {mode_env}")
    st.caption(f"API_FOOTBALL_KEY: {'OK' if api_key_present else 'NO'}")

    if st.button("🔄 Limpiar cache y recargar"):
        st.cache_data.clear()
        st.rerun()

    st.markdown(f"""
    **Modelo:** Balanced Poisson GLM
    **Features:** 78 V2 (Rolling + Context)
    **Backtest Accuracy:** 71.7% (GW 15-22)
    **ROI Backtest:** +23 units

    ---

    **Niveles de Confianza:**
    - 🟢 **Alta (≥65%):** Apostar
    - 🟡 **Media (50-64%):** Considerar
    - 🔴 **Baja (<50%):** Evitar

    ---

    **Última actualización:** {last_update}
    **Modelo re-entrenado:** Datos 2024-2025
    """)

# Cargar datos
@st.cache_data(ttl=10 * 60)
def load_predictions(csv_path: str, file_mtime: float | None):
    df = pd.read_csv(csv_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


def _normalize_team_name(s: object) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return str(s).strip().lower()


def _add_match_key(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['match_date'] = df['date'].dt.normalize()
    else:
        df['match_date'] = pd.NaT
    if 'home_team' in df.columns:
        df['home_team_key'] = df['home_team'].map(_normalize_team_name)
    elif 'home_team_name' in df.columns:
        df['home_team_key'] = df['home_team_name'].map(_normalize_team_name)
    else:
        df['home_team_key'] = ""
    if 'away_team' in df.columns:
        df['away_team_key'] = df['away_team'].map(_normalize_team_name)
    elif 'away_team_name' in df.columns:
        df['away_team_key'] = df['away_team_name'].map(_normalize_team_name)
    else:
        df['away_team_key'] = ""
    df['match_key'] = (
        df['match_date'].astype('datetime64[ns]').astype('int64').astype(str)
        + "|" + df['home_team_key']
        + "|" + df['away_team_key']
    )
    return df


def _compute_logloss(y_true: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())


def _load_prediction_config(config_path: str = 'config/prediction.yaml') -> dict:
    p = Path(config_path)
    if not p.exists():
        return {}
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


@st.cache_data(ttl=10 * 60)
def load_official_fixtures(
    historical_teams: list[str],
    league_id: int,
    season: int,
    cache_ttl_hours: int = 6,
    num_gameweeks: int = 3
) -> pd.DataFrame:
    load_dotenv()
    api_key = os.getenv('API_FOOTBALL_KEY')
    if not api_key or api_key == 'your_rapidapi_key_here':
        raise ValueError('API_FOOTBALL_KEY no está configurada en tu .env')

    source = FixtureAPISource(
        historical_teams=historical_teams,
        api_key=api_key,
        league_id=league_id,
        season=season,
        cache_enabled=True,
        cache_ttl_hours=cache_ttl_hours,
    )
    df_fix = source.get_upcoming_fixtures(num_gameweeks=num_gameweeks, future_only=True)
    if 'date' in df_fix.columns:
        df_fix['date'] = pd.to_datetime(df_fix['date'], errors='coerce')
    return df_fix

try:
    csv_path = str(predictions_path)
    df = load_predictions(csv_path, predictions_mtime)

    # Mostrar rango real de fechas (si existe)
    if 'date' in df.columns and not df['date'].isna().all():
        date_min = df['date'].min()
        date_max = df['date'].max()
        st.sidebar.caption(f"Rango de fechas en dataset: {date_min:%d/%m/%Y} - {date_max:%d/%m/%Y}")
    else:
        st.sidebar.caption("Rango de fechas en dataset: (sin columna date válida)")

    # Métricas generales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Fixtures",
            value=len(df)
        )

    with col2:
        bet_count = (df['recommendation'].str.contains('BET')).sum()
        st.metric(
            label="Recomendaciones BET",
            value=f"{bet_count}/{len(df)}",
            delta=f"{100*bet_count/len(df):.0f}%"
        )

    with col3:
        avg_conf = df['confidence'].mean()
        st.metric(
            label="Confianza Promedio",
            value=f"{avg_conf:.1f}%"
        )

    with col4:
        over_pct = (df['prediction'] == 1).mean()
        st.metric(
            label="Predicciones Over 2.5",
            value=f"{100*over_pct:.0f}%"
        )

    st.markdown("---")

    # Tabs para diferentes vistas
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Recomendaciones",
        "📊 Todas las Predicciones",
        "📈 Análisis de Probabilidades",
        "🔍 Detalles por Partido",
        "📅 Próximos Partidos",
        "✅ Validación"
    ])

    with tab1:
        st.subheader("🎯 Mejores Apuestas (Ordenadas por Confianza)")

        # Filtrar solo BET
        bets_df = df[df['recommendation'].str.contains('BET')].copy()
        bets_df = bets_df.sort_values('confidence', ascending=False)

        if len(bets_df) > 0:
            for idx, row in bets_df.iterrows():
                # Determinar color de confianza
                if row['confidence'] >= 65:
                    conf_class = 'confidence-high'
                    conf_emoji = '🟢'
                elif row['confidence'] >= 50:
                    conf_class = 'confidence-medium'
                    conf_emoji = '🟡'
                else:
                    conf_class = 'confidence-low'
                    conf_emoji = '🔴'

                # Box de predicción
                pred_class = 'prediction-over' if row['prediction'] == 1 else 'prediction-under'

                with st.container():
                    st.markdown(f"""
                    <div class="{pred_class}">
                        <h3>{conf_emoji} {row['home_team']} vs {row['away_team']}</h3>
                        <p><strong>Fecha:</strong> {row['date']} | <strong>Gameweek:</strong> {row['gameweek']}</p>
                        <p><strong>Predicción:</strong> {row['prediction_label']}</p>
                        <p><strong>Probabilidad Over:</strong> {row['prob_over']:.1%} | <strong>Under:</strong> {row['prob_under']:.1%}</p>
                        <p><strong>Confianza:</strong> <span class="{conf_class}">{row['confidence']:.1f}%</span> ({row['confidence_tier']})</p>
                        <p><strong>Expected Goals:</strong> {row['expected_total_goals']:.2f} goles</p>
                        <p><strong>✅ Recomendación:</strong> {row['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("")
        else:
            st.warning("No hay apuestas recomendadas con la confianza mínima establecida.")

    with tab2:
        st.subheader("📊 Todas las Predicciones")

        # Crear DataFrame para mostrar
        display_df = df[[
            'date', 'home_team', 'away_team', 'gameweek',
            'prediction_label', 'prob_over', 'prob_under',
            'expected_total_goals', 'confidence', 'recommendation'
        ]].copy()

        display_df.columns = [
            'Fecha', 'Equipo Local', 'Equipo Visitante', 'GW',
            'Predicción', 'Prob Over', 'Prob Under',
            'xG Total', 'Confianza %', 'Recomendación'
        ]

        # Aplicar formato
        display_df['Prob Over'] = display_df['Prob Over'].apply(lambda x: f"{x:.1%}")
        display_df['Prob Under'] = display_df['Prob Under'].apply(lambda x: f"{x:.1%}")
        display_df['xG Total'] = display_df['xG Total'].apply(lambda x: f"{x:.2f}")
        display_df['Confianza %'] = display_df['Confianza %'].apply(lambda x: f"{x:.1f}%")

        st.dataframe(
            display_df,
            use_container_width=True,
            height=600
        )

    with tab3:
        st.subheader("📈 Distribución de Probabilidades")

        # Gráfico de distribución de probabilidades
        fig1 = go.Figure()

        fig1.add_trace(go.Histogram(
            x=df['prob_over'],
            nbinsx=20,
            name='Probabilidad Over 2.5',
            marker_color='green',
            opacity=0.7
        ))

        fig1.update_layout(
            title="Distribución de Probabilidades Over 2.5",
            xaxis_title="Probabilidad",
            yaxis_title="Frecuencia",
            showlegend=True,
            height=400
        )

        st.plotly_chart(fig1, use_container_width=True)

        # Gráfico de confianza vs probabilidad
        fig2 = px.scatter(
            df,
            x='prob_over',
            y='confidence',
            color='prediction_label',
            size='expected_total_goals',
            hover_data=['home_team', 'away_team', 'recommendation'],
            title='Confianza vs Probabilidad Over 2.5',
            labels={
                'prob_over': 'Probabilidad Over 2.5',
                'confidence': 'Confianza (%)',
                'prediction_label': 'Predicción'
            },
            color_discrete_map={'Over 2.5': 'green', 'Under 2.5': 'blue'}
        )

        fig2.add_hline(y=65, line_dash="dash", line_color="orange",
                      annotation_text="Confianza Alta (65%)")
        fig2.add_vline(x=0.5, line_dash="dash", line_color="gray",
                      annotation_text="50% probabilidad")

        fig2.update_layout(height=500)

        st.plotly_chart(fig2, use_container_width=True)

        # Gráfico de barras: Confianza por partido
        df_sorted = df.sort_values('confidence', ascending=True)
        df_sorted['match'] = df_sorted['home_team'].str[:3] + ' vs ' + df_sorted['away_team'].str[:3]

        fig3 = go.Figure()

        colors = ['green' if c >= 65 else 'orange' if c >= 50 else 'red'
                  for c in df_sorted['confidence']]

        fig3.add_trace(go.Bar(
            x=df_sorted['confidence'],
            y=df_sorted['match'],
            orientation='h',
            marker_color=colors,
            text=df_sorted['confidence'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto'
        ))

        fig3.update_layout(
            title="Confianza por Partido (Ordenado)",
            xaxis_title="Confianza (%)",
            yaxis_title="Partido",
            height=600,
            showlegend=False
        )

        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.subheader("🔍 Detalles Completos por Partido")

        # Selector de partido
        matches = df.apply(lambda r: f"{r['home_team']} vs {r['away_team']} ({r['date']})", axis=1)
        selected_match = st.selectbox("Selecciona un partido:", matches)

        selected_idx = matches[matches == selected_match].index[0]
        match_data = df.loc[selected_idx]

        # Mostrar detalles
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📋 Información del Partido")
            st.markdown(f"""
            - **Local:** {match_data['home_team']}
            - **Visitante:** {match_data['away_team']}
            - **Fecha:** {match_data['date']}
            - **Gameweek:** {match_data['gameweek']}
            """)

            st.markdown("### 🎯 Predicción")
            pred_color = 'green' if match_data['prediction'] == 1 else 'blue'
            st.markdown(f"""
            <div style='background-color: {pred_color}22; padding: 20px; border-radius: 10px; border: 2px solid {pred_color}'>
                <h2 style='color: {pred_color}; text-align: center;'>{match_data['prediction_label']}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 📊 Métricas de Probabilidad")

            # Gauge de probabilidad Over
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=match_data['prob_over'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilidad Over 2.5 (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightblue"},
                        {'range': [50, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))

            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Métricas adicionales
        st.markdown("### 📈 Métricas Detalladas")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Prob Over", f"{match_data['prob_over']:.1%}")

        with col2:
            st.metric("Prob Under", f"{match_data['prob_under']:.1%}")

        with col3:
            st.metric("Expected Goals", f"{match_data['expected_total_goals']:.2f}")

        with col4:
            conf_delta = match_data['confidence'] - 50
            st.metric(
                "Confianza",
                f"{match_data['confidence']:.1f}%",
                delta=f"{conf_delta:+.1f}%"
            )

        # Expected goals desglosados
        if 'expected_home_goals' in df.columns:
            st.markdown("### ⚽ Expected Goals Desglosados")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    f"{match_data['home_team']} (Local)",
                    f"{match_data['expected_home_goals']:.2f} goles"
                )

            with col2:
                st.metric(
                    f"{match_data['away_team']} (Visitante)",
                    f"{match_data['expected_away_goals']:.2f} goles"
                )

            # Gráfico de barras comparativo
            fig_xg = go.Figure()

            fig_xg.add_trace(go.Bar(
                x=['Local', 'Visitante'],
                y=[match_data['expected_home_goals'], match_data['expected_away_goals']],
                marker_color=['#4CAF50', '#2196F3'],
                text=[f"{match_data['expected_home_goals']:.2f}",
                      f"{match_data['expected_away_goals']:.2f}"],
                textposition='auto'
            ))

            fig_xg.add_hline(y=2.5/2, line_dash="dash", line_color="red",
                            annotation_text="Umbral 2.5 (promedio por equipo)")

            fig_xg.update_layout(
                title="Expected Goals por Equipo",
                yaxis_title="Expected Goals",
                showlegend=False,
                height=300
            )

            st.plotly_chart(fig_xg, use_container_width=True)

    with tab5:
        st.subheader("📅 Análisis de Predicciones Futuras")

        config = _load_prediction_config('config/prediction.yaml')
        real_cfg = (config.get('real_fixtures') or {}).get('api', {}).get('api_football', {})
        default_league_id = int(real_cfg.get('league_id', 39) or 39)
        default_season = int(real_cfg.get('season', 2024) or 2024)

        show_official = st.toggle("Mostrar fixtures oficiales (API-Football)", value=True)
        if show_official:
            teams = sorted(set(df.get('home_team', pd.Series([], dtype=str)).dropna().astype(str)).union(
                set(df.get('away_team', pd.Series([], dtype=str)).dropna().astype(str))
            ))
            if not teams:
                st.warning("No se pudo inferir lista de equipos desde el CSV de predicciones. Necesito columnas home_team/away_team.")
            else:
                colx, coly, colz = st.columns([1, 1, 2])
                with colx:
                    league_id = st.number_input("League ID", min_value=1, value=default_league_id, step=1)
                with coly:
                    season = st.number_input("Season (año)", min_value=2000, value=default_season, step=1)
                with colz:
                    st.caption("Requiere `API_FOOTBALL_KEY` en tu `.env`.\nSi no ves partidos, revisa el año de season (ej. 2025 para 2025/26).")

                try:
                    fixtures_official = load_official_fixtures(
                        historical_teams=teams,
                        league_id=int(league_id),
                        season=int(season),
                    )

                    if 'is_official' in fixtures_official.columns:
                        official_count = int(fixtures_official['is_official'].fillna(False).astype(bool).sum())
                    else:
                        official_count = 0

                    st.success(f"Fixtures cargados desde API: {len(fixtures_official)} | Oficiales: {official_count}")

                    # Rango por defecto: próxima semana
                    today = pd.Timestamp.now().normalize()
                    next_week_end = today + pd.Timedelta(days=7)
                    fixtures_next_week = fixtures_official[fixtures_official['date'].notna()].copy()
                    fixtures_next_week = fixtures_next_week[(fixtures_next_week['date'] >= today) & (fixtures_next_week['date'] <= next_week_end)]
                    fixtures_next_week = fixtures_next_week.sort_values('date')

                    st.markdown("### ✅ Próxima semana (fixtures oficiales)")
                    st.metric("Partidos próximos 7 días", len(fixtures_next_week))

                    cols_show = [c for c in ['date', 'gameweek', 'home_team', 'away_team', 'venue', 'kickoff_time', 'fixture_id', 'api_fixture_id', 'source_type', 'is_official'] if c in fixtures_next_week.columns]
                    st.dataframe(fixtures_next_week[cols_show], use_container_width=True, height=450)

                except Exception as e:
                    st.error(f"No se pudieron cargar fixtures oficiales: {e}")
                    st.info("Solución típica: crear `.env` desde `.env.example` y poner tu `API_FOOTBALL_KEY`.\nOpcional: setear `PREDICTION_MODE=REAL_FIXTURES` para que el pipeline genere predicciones con fixtures reales.")

        if not show_official:
            st.warning("Estás viendo solo el CSV de predicciones. Si ese CSV fue generado en SIMULATION, los partidos pueden ser ficticios.")

        if 'date' not in df.columns or df['date'].isna().all():
            st.warning("El dataset no tiene una columna 'date' válida para análisis por fechas.")
        else:
            today = pd.Timestamp.now().normalize()
            col_a, col_b, col_c = st.columns([1, 1, 2])
            with col_a:
                future_mode = st.radio(
                    "Rango",
                    options=["Solo futuros", "Rango personalizado"],
                    horizontal=True
                )
            with col_b:
                next_days = st.number_input("Próximos N días", min_value=1, max_value=60, value=14, step=1)
            with col_c:
                if future_mode == "Rango personalizado":
                    start = st.date_input("Desde", value=today.date())
                    end = st.date_input("Hasta", value=(today + pd.Timedelta(days=int(next_days))).date())
                    start_ts = pd.Timestamp(start)
                    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                else:
                    start_ts = today
                    end_ts = today + pd.Timedelta(days=int(next_days))

            future_df = df[df['date'].notna()].copy()
            future_df = future_df[(future_df['date'] >= start_ts) & (future_df['date'] <= end_ts)].copy()
            sort_cols = ['date']
            if 'gameweek' in future_df.columns:
                sort_cols.append('gameweek')
            future_df = future_df.sort_values(sort_cols)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Partidos en rango", len(future_df))
            with col2:
                st.metric("BET en rango", int(future_df['recommendation'].str.contains('BET').sum()) if 'recommendation' in future_df.columns else 0)
            with col3:
                st.metric("Confianza prom.", f"{future_df['confidence'].mean():.1f}%" if len(future_df) else "-")
            with col4:
                st.metric("Prob. Over prom.", f"{future_df['prob_over'].mean():.1%}" if len(future_df) else "-")

            if len(future_df) == 0:
                st.info("No hay partidos en el rango seleccionado.")
            else:
                future_df['match_day'] = future_df['date'].dt.date
                agg_spec = {
                    'fixtures': ('match_day', 'count'),
                }
                if 'recommendation' in future_df.columns:
                    agg_spec['bet_count'] = ('recommendation', lambda s: s.astype(str).str.contains('BET').sum())
                if 'confidence' in future_df.columns:
                    agg_spec['avg_conf'] = ('confidence', 'mean')
                if 'prob_over' in future_df.columns:
                    agg_spec['avg_prob_over'] = ('prob_over', 'mean')

                by_day = future_df.groupby('match_day', as_index=False).agg(**agg_spec)

                fig_day = px.bar(
                    by_day,
                    x='match_day',
                    y='fixtures',
                    title='Partidos por día (rango seleccionado)',
                    labels={'match_day': 'Día', 'fixtures': 'Partidos'}
                )
                st.plotly_chart(fig_day, use_container_width=True)

                st.markdown("### 📋 Partidos del rango")
                cols = [c for c in [
                    'date', 'gameweek', 'home_team', 'away_team', 'prediction_label',
                    'prob_over', 'confidence', 'recommendation'
                ] if c in future_df.columns]
                st.dataframe(future_df[cols], use_container_width=True, height=500)

    with tab6:
        st.subheader("✅ Validación: Predicción vs Resultado Real")

        st.markdown("Carga resultados reales para comparar contra las predicciones y medir performance.")
        uploaded = st.file_uploader("Sube resultados (CSV o Parquet)", type=['csv', 'parquet', 'pq'])

        if uploaded is None:
            st.info("Sube un archivo con resultados reales para iniciar la validación.")
        else:
            name = getattr(uploaded, 'name', '').lower()
            if name.endswith('.parquet') or name.endswith('.pq'):
                actual_df = pd.read_parquet(uploaded)
            else:
                actual_df = pd.read_csv(uploaded)

            if 'date' in actual_df.columns:
                actual_df['date'] = pd.to_datetime(actual_df['date'], errors='coerce')

            if 'total_goals' not in actual_df.columns:
                if 'home_goals' in actual_df.columns and 'away_goals' in actual_df.columns:
                    actual_df['total_goals'] = actual_df['home_goals'] + actual_df['away_goals']

            if 'over_2.5' not in actual_df.columns and 'total_goals' in actual_df.columns:
                actual_df['over_2.5'] = (actual_df['total_goals'] > 2.5).astype(int)

            preds_keyed = _add_match_key(df)
            actual_keyed = _add_match_key(actual_df)

            join_mode = "match_key"
            if 'fixture_id' in preds_keyed.columns and 'fixture_id' in actual_keyed.columns:
                if preds_keyed['fixture_id'].notna().any() and actual_keyed['fixture_id'].notna().any():
                    join_mode = "fixture_id"

            if join_mode == "fixture_id":
                merged = preds_keyed.merge(
                    actual_keyed,
                    on='fixture_id',
                    how='inner',
                    suffixes=('_pred', '_actual')
                )
            else:
                merged = preds_keyed.merge(
                    actual_keyed,
                    on='match_key',
                    how='inner',
                    suffixes=('_pred', '_actual')
                )

            required = ['prediction', 'prob_over']
            if not all(c in merged.columns for c in required):
                st.error("El dataset de predicciones no contiene columnas necesarias: 'prediction' y 'prob_over'.")
            elif 'over_2.5' not in merged.columns:
                st.error("El dataset de resultados reales debe tener 'over_2.5' o 'total_goals' (o home_goals+away_goals).")
            else:
                eval_df = merged.copy()
                y_true = eval_df['over_2.5'].astype(int).to_numpy()
                y_pred = eval_df['prediction'].astype(int).to_numpy()
                p_over = pd.to_numeric(eval_df['prob_over'], errors='coerce').fillna(0.5).to_numpy()

                accuracy = float((y_true == y_pred).mean()) if len(y_true) else float('nan')
                brier = float(np.mean((p_over - y_true) ** 2)) if len(y_true) else float('nan')
                logloss = _compute_logloss(y_true, p_over) if len(y_true) else float('nan')

                if 'recommendation' in eval_df.columns:
                    bet_mask = eval_df['recommendation'].astype(str).str.startswith('BET')
                    if bet_mask.any():
                        correct_bets = (y_true[bet_mask.to_numpy()] == y_pred[bet_mask.to_numpy()]).sum()
                        bets = int(bet_mask.sum())
                        roi = float(2 * correct_bets - bets)
                    else:
                        roi = 0.0
                else:
                    roi = 0.0

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Partidos validados", len(eval_df))
                with col2:
                    st.metric("Accuracy", f"{accuracy:.1%}" if len(eval_df) else "-")
                with col3:
                    st.metric("Brier", f"{brier:.4f}" if len(eval_df) else "-")
                with col4:
                    st.metric("ROI (simple)", f"{roi:+.1f}" if len(eval_df) else "-")

                st.caption(f"Join usado: {join_mode}")

                display_cols = []
                for c in ['fixture_id', 'date_pred', 'home_team_pred', 'away_team_pred', 'gameweek_pred', 'prediction_label', 'prob_over', 'confidence', 'recommendation', 'home_goals', 'away_goals', 'total_goals', 'over_2.5']:
                    if c in eval_df.columns:
                        display_cols.append(c)

                if display_cols:
                    out = eval_df[display_cols].copy()
                else:
                    out = eval_df.copy()

                if 'prediction' in eval_df.columns and 'over_2.5' in eval_df.columns:
                    out['acierto'] = (eval_df['prediction'].astype(int) == eval_df['over_2.5'].astype(int))

                st.markdown("### 🧾 Resultados por partido")
                st.dataframe(out, use_container_width=True, height=600)

                csv_bytes = out.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar validación (CSV)",
                    data=csv_bytes,
                    file_name="validation_report.csv",
                    mime="text/csv"
                )

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray;'>
        <p>Dashboard generado con datos del {last_update} | Modelo Balanced Poisson GLM</p>
        <p>⚠️ Estas predicciones son estimaciones estadísticas. Apuesta responsablemente.</p>
    </div>
    """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("❌ Archivo de predicciones no encontrado. Ejecuta primero el script de predicción.")
    st.code("python scripts/predict_fixtures.py --input data/predictions/prediction_data_*.parquet")

except Exception as e:
    st.error(f"Error al cargar datos: {e}")
    st.exception(e)
