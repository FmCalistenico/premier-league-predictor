# Dashboard Setup Guide

## Problema Actual

El dashboard esta corriendo pero muestra un error:
```
KeyError: "None of [Index(['home_goals_scored_L3', ...]] are in the [columns]"
```

**Causa**: El modelo guardado (`poisson_balanced`) fue entrenado con features V2 (72 features), pero el dashboard necesita datos procesados con esas mismas features.

---

## Solucion 1: Generar Datos para el Dashboard (RECOMENDADO)

El dashboard necesita un archivo de datos procesado con las features V2. Ejecuta:

```powershell
# Opcion A: Ejecutar pipeline completo
python run_complete_pipeline.py

# Opcion B: Solo generar features V2
python scripts/train_model_v2.py --validate-features
```

Esto generara:
- `data/final/training_data_latest.parquet` - Dataset con todas las features
- `models/results/` - Modelos entrenados compatibles

Luego abre el dashboard en tu navegador y **carga el archivo manualmente**:
1. Abre: http://localhost:8501
2. En la barra lateral: "Upload CSV/Parquet"
3. Selecciona: `data/final/training_data_latest.parquet`

---

## Solucion 2: Usar Modelo Simple (Para Testing Rapido)

Si solo quieres probar el dashboard sin entrenar, usa datos de ejemplo:

```powershell
# Crear datos de ejemplo simples
python -c "
import pandas as pd
import numpy as np

# Crear dataset minimo
n = 100
df = pd.DataFrame({
    'home_team': ['Arsenal', 'Liverpool'] * 50,
    'away_team': ['Chelsea', 'Man City'] * 50,
    'home_goals': np.random.randint(0, 4, n),
    'away_goals': np.random.randint(0, 4, n),
    'date': pd.date_range('2024-01-01', periods=n)
})

df['total_goals'] = df['home_goals'] + df['away_goals']
df['over_2.5'] = (df['total_goals'] > 2.5).astype(int)

# Guardar
df.to_csv('data/processed/dashboard_test.csv', index=False)
print('Archivo creado: data/processed/dashboard_test.csv')
"
```

Luego carga `dashboard_test.csv` en el dashboard.

---

## Solucion 3: Entrenar Modelo Compatible

Entrena un modelo nuevo que el dashboard pueda usar:

```powershell
# Entrenar con features V2 y guardar correctamente
python scripts/retrain_improved_pipeline.py
```

Esto creara:
- Modelos en `models/results/retrain_checkpoint.pkl`
- Datos compatibles en `data/final/`
- Plots en `models/results/retrain_plots_*/`

---

## Como Ejecutar el Dashboard

### Metodo 1: Script PowerShell
```powershell
.\run_dashboard.ps1
```

### Metodo 2: Comando Directo
```powershell
python -m streamlit run dashboards/bias_monitor.py
```

### Metodo 3: Agregar al PATH
1. Agrega a PATH: `C:\Users\ediSh\AppData\Roaming\Python\Python314\Scripts`
2. Reinicia PowerShell
3. Ejecuta: `streamlit run dashboards/bias_monitor.py`

---

## Verificar Instalacion

```powershell
# Verificar todas las dependencias
python -c "import streamlit; import plotly; import statsmodels; import sklearn; print('OK')"

# Listar modelos disponibles
ls models/results/

# Ver datos procesados
ls data/final/
```

---

## Uso del Dashboard

Una vez corriendo (http://localhost:8501):

1. **Carga de Datos**:
   - Sidebar -> "Upload CSV/Parquet"
   - O selecciona archivo existente del dropdown

2. **Seleccion de Modelo**:
   - Sidebar -> "Model File"
   - Selecciona: `retrain_checkpoint.pkl` o `production_model.pkl`

3. **Analisis**:
   - ROC Curve
   - Calibration Plot
   - Probability Histogram
   - Confusion Matrix
   - Metrics Gauges

4. **Filtros**:
   - Por equipo
   - Por fecha
   - Por threshold

---

## Troubleshooting

### Error: "No module named 'X'"
```powershell
python -m pip install X
```

### Error: "KeyError: features not in columns"
- Regenera datos con features V2
- O usa modelo compatible con tus datos

### Error: "streamlit not found"
```powershell
# Usa python -m streamlit en lugar de streamlit directamente
python -m streamlit run dashboards/bias_monitor.py
```

### Dashboard se cierra inmediatamente
```powershell
# Verifica logs
cat logs/app.log

# O ejecuta sin background
python -m streamlit run dashboards/bias_monitor.py --server.headless false
```

---

## Proximos Pasos Recomendados

1. **Genera datos compatibles**: `python run_complete_pipeline.py`
2. **Abre dashboard**: http://localhost:8501
3. **Carga datos**: Upload `data/final/training_data_latest.parquet`
4. **Analiza metricas**: Revisa ROC AUC, calibration, etc.
5. **Optimiza threshold**: Usa controles del dashboard para encontrar mejor threshold

---

## Contacto y Ayuda

- Documentacion completa: `README.md`
- Guia rapida: `QUICKSTART.md`
- Soluciones sesgo: `SOLUTIONS_MODEL_BIAS.md`
- Features V1 vs V2: `FEATURE_ENGINEERING_COMPARISON.md`
