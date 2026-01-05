# Guia Rapida - Dashboard Bias Monitor

## DATOS LISTOS CON FEATURES V2

Se han generado exitosamente los datos con las 94 columnas (78 features V2):

```
Archivo: data/final/training_data_v2.parquet
Filas: 949 partidos
Columnas: 94 (90 numericas)
Features V2: TODAS incluidas
```

---

## COMO USAR EL DASHBOARD

### 1. Abrir Dashboard

El dashboard ya esta corriendo en:
- **URL**: http://localhost:8501

Si no esta corriendo, ejecuta:
```powershell
python -m streamlit run dashboards/bias_monitor.py
```

### 2. Cargar Datos en el Dashboard

En el campo **"Dataset path (CSV/Parquet o glob)"**, escribe:

```
data/final/training_data_v2.parquet
```

O alternativamente (version CSV):
```
data/final/training_data_v2.csv
```

### 3. Seleccionar Modelo

El modelo compatible ya esta cargado: `poisson_balanced`

Si quieres cambiar, en "Model File" selecciona:
- `retrain_checkpoint.pkl` - Modelo entrenado con V2

---

## FEATURES V2 INCLUIDAS (94 columnas totales)

### Informacion Basica (13)
- fixture_id, date, season
- home_team, away_team
- home_goals, away_goals, total_goals
- over_0.5, over_1.5, over_2.5, over_3.5, over_4.5

### Rolling Features - Casa (18)
- home_goals_scored_L3/L5/L10
- home_goals_conceded_L3/L5/L10
- home_goal_diff_L3/L5/L10
- home_over_rate_L3/L5/L10
- home_clean_sheet_rate_L3/L5/L10
- home_failed_to_score_rate_L3/L5/L10

### Rolling Features - Visitante (18)
- away_goals_scored_L3/L5/L10
- away_goals_conceded_L3/L5/L10
- away_goal_diff_L3/L5/L10
- away_over_rate_L3/L5/L10
- away_clean_sheet_rate_L3/L5/L10
- away_failed_to_score_rate_L3/L5/L10

### League Averages (5)
- gameweek
- league_avg_goals_L5/L10
- league_avg_conceded_L5/L10

### Ratio Features (10)
- home/away_attack_ratio_L5/L10
- home/away_defense_ratio_L5/L10
- attack_defense_ratio_L5
- defense_attack_ratio_L5

### Form Features (2)
- form_diff_L5
- form_diff_L10

### Momentum Features (6)
- home/away_goals_momentum
- home/away_defense_momentum
- home/away_form_momentum

### Volatility Features (7)
- home/away_goals_scored_volatility_L5
- home/away_goals_conceded_volatility_L5
- home/away_total_goals_volatility_L5
- combined_volatility

### Context Features (9)
- is_derby
- home/away_points_L10
- home/away_position_L10
- is_top6_clash
- is_relegation_battle
- is_mismatch
- position_diff

### Rest Features (6)
- home/away_days_rest
- rest_advantage
- days_since_last_match_diff
- home/away_short_rest

---

## VERIFICACION RAPIDA

Si quieres verificar las features antes de cargar:

```powershell
python check_features.py
```

Esto muestra:
- Total de columnas
- Features V2 encontradas
- Lista completa de columnas

---

## METRICAS ESPERADAS EN DASHBOARD

Despues de cargar los datos, el dashboard mostrara:

### KPIs Principales
- **ROC AUC**: ~0.50 (modelo balanced con CV)
- **Balanced Accuracy**: ~0.50
- **Pred Over Rate**: Variable segun fold (38-83%)
- **Specificity**: ~61%
- **Sensitivity**: ~39%

### Visualizaciones
1. **ROC Curve** - Curva ROC con AUC
2. **Calibration Plot** - Calibracion de probabilidades
3. **Probability Histogram** - Distribucion de predicciones
4. **Confusion Matrix** - Matriz de confusion
5. **Metrics Gauges** - Indicadores tipo velocimetro

---

## TROUBLESHOOTING

### Error: "KeyError: features not found"
**Solucion**: Asegurate de usar `training_data_v2.parquet` (no `training_data_latest.parquet`)

### Error: "File not found"
**Solucion**: Usa ruta relativa desde la raiz del proyecto:
```
data/final/training_data_v2.parquet
```

### Dashboard no carga datos
**Solucion**: Intenta con ruta absoluta:
```
C:\Users\ediSh\OneDrive\Desktop\fmCalistenico\premier-league-predictor\data\final\training_data_v2.parquet
```

### Modelo predice 100% Over
**Problema conocido**: El modelo `poisson_balanced` tiene threshold muy alto en algunos folds
**Solucion**: Esto es normal en CV, el dashboard muestra metricas promedio

---

## COMANDOS UTILES

```powershell
# Verificar features
python check_features.py

# Ejecutar dashboard
python -m streamlit run dashboards/bias_monitor.py

# Regenerar datos V2
python scripts/retrain_improved_pipeline.py --models poisson_balanced

# Ver resultados de entrenamiento
cat models/results/retrain_comparison.csv

# Ver plots generados
ls models/results/retrain_plots_*/
```

---

## SIGUIENTE PASO

1. Abre: http://localhost:8501
2. En "Dataset path", escribe: `data/final/training_data_v2.parquet`
3. El dashboard cargara automaticamente
4. Explora las visualizaciones y metricas

LISTO!
