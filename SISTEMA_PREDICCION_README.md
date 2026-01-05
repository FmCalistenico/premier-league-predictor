# Sistema Completo de Predicción - Premier League Over/Under 2.5

## 🎯 ¿Qué es Este Sistema?

Un sistema end-to-end de Machine Learning para **predecir resultados Over/Under 2.5 goles** en partidos de la Premier League con **scoring de confiabilidad**.

### Características Principales

✅ **Features V2 Avanzadas** (78 features sin data leakage)
- Rolling stats (L3/L5/L10)
- Ratio features normalizados por liga
- Momentum y volatility
- Context features (derbies, top6 clashes)

✅ **Confidence Scoring** (0-100%)
- Predicciones con nivel de confiabilidad
- Recomendaciones BET/SKIP automáticas
- Filtrado de predicciones inciertas

✅ **Threshold Optimizado**
- Sistema de optimización automática
- Target: Sensitivity ≥55%, Specificity ≥60%
- Pred Over Rate ≈50-55% (balanceado)

✅ **Sistema de Validación**
- Backtest en jornadas pasadas
- Cálculo de ROI simulado
- Monitoreo de data drift

---

## 📦 Archivos del Sistema

### Scripts Principales

| Script | Propósito | Uso |
|--------|-----------|-----|
| `get_upcoming_fixtures.py` | Obtener fixtures futuros | Desde API o manual |
| `build_prediction_dataset.py` | Calcular features V2 | **SIN data leakage** |
| `predict_fixtures.py` | Generar predicciones | Con confidence scoring |
| `optimize_threshold_production.py` | Optimizar threshold | Una vez antes de producción |
| `backtest_predictions.py` | Validar sistema | En jornadas pasadas |

### Scripts de Apoyo

| Script | Propósito |
|--------|-----------|
| `example_complete_workflow.py` | Ejemplo completo funcionando |
| `check_features.py` | Verificar features en datos |
| `retrain_improved_pipeline.py` | Re-entrenar modelo |

### Documentación

| Archivo | Contenido |
|---------|-----------|
| `QUICK_START_PREDICTIONS.md` | **Guía rápida (EMPIEZA AQUÍ)** |
| `PREDICTION_SYSTEM_GUIDE.md` | Guía maestra completa |
| `DASHBOARD_SETUP.md` | Setup del dashboard |
| `SOLUTIONS_MODEL_BIAS.md` | Soluciones para sesgo |

---

## 🚀 Quick Start (3 Opciones)

### Opción 1: Ejemplo Completo (MÁS RÁPIDO)

```powershell
python example_complete_workflow.py
```

Genera predicciones de ejemplo para 4 partidos en **< 1 minuto**.

### Opción 2: Workflow Manual Paso a Paso

```powershell
# 1. Get fixtures
python scripts/get_upcoming_fixtures.py --manual

# 2. Build dataset
python scripts/build_prediction_dataset.py --input data/raw/upcoming_fixtures_*.csv --gameweek 23

# 3. Predict
python scripts/predict_fixtures.py --input data/predictions/prediction_data_*.parquet --threshold 0.62 --min-confidence 70
```

### Opción 3: Desde API-Football

```powershell
# Setup API key
$env:FOOTBALL_DATA_API_KEY = "tu_key_aqui"

# 1. Get fixtures
python scripts/get_upcoming_fixtures.py --from-api --gameweek 23

# 2-3. Same as Opción 2
```

---

## 📊 Output Esperado

### Archivo CSV de Predicciones

```csv
fixture_id,date,home_team,away_team,prob_over,prob_under,prediction_label,confidence,expected_total_goals,recommendation
GW23_ARS_MCI,2026-01-11,Arsenal,Man City,0.72,0.28,Over 2.5,85,3.2,"BET: Over 2.5 (High Confidence)"
GW23_NEW_FUL,2026-01-11,Newcastle,Fulham,0.45,0.55,Under 2.5,82,2.3,"BET: Under 2.5 (High Confidence)"
GW23_TOT_BRE,2026-01-11,Tottenham,Brentford,0.52,0.48,Over 2.5,41,3.0,"SKIP: Low Confidence"
GW23_LIV_CHE,2026-01-12,Liverpool,Chelsea,0.68,0.32,Over 2.5,76,3.1,"BET: Over 2.5 (High Confidence)"
```

### Interpretación

#### BET: High Confidence (≥80%)
```
Arsenal vs Man City | Over 2.5 (72%) | Confidence: 85%
→ Apostar con confianza (stake normal)
```

#### BET: Medium Confidence (65-79%)
```
Liverpool vs Chelsea | Over 2.5 (68%) | Confidence: 76%
→ Apostar con cautela (stake reducido)
```

#### SKIP: Low Confidence (<65%)
```
Tottenham vs Brentford | Over 2.5 (52%) | Confidence: 41%
→ NO apostar (muy incierto)
```

---

## ⚙️ Configuración Inicial (Una Vez)

### 1. Optimizar Threshold

El modelo viene con threshold=0.75 (muy conservador). Optimízalo:

```powershell
python scripts/optimize_threshold_production.py \
    --metric custom \
    --target-sensitivity 0.55 \
    --target-specificity 0.60 \
    --save-model
```

**Resultado esperado:**
- Threshold óptimo: ~0.62-0.65
- Balanced Accuracy: ~0.60
- Pred Over Rate: ~50-55%

### 2. Validar con Backtest

```powershell
python scripts/backtest_predictions.py \
    --start-gameweek 18 \
    --end-gameweek 22 \
    --min-confidence 70
```

**Métricas objetivo:**
- Accuracy: ≥55%
- ROI: Positivo
- Calibration error: <0.10

### 3. Ajustar Min-Confidence

Basado en backtest:

```
Si ROI positivo con confidence ≥70% → Usar 70%
Si ROI negativo con confidence ≥70% → Subir a 75-80%
```

---

## 🔄 Workflow Semanal

### Antes de Fixtures (Lunes/Martes)

```powershell
# 1. Obtener fixtures
python scripts/get_upcoming_fixtures.py --manual

# 2. Build dataset con V2 features
python scripts/build_prediction_dataset.py --input data/raw/upcoming_fixtures_*.csv

# 3. Predecir con confidence
python scripts/predict_fixtures.py --input data/predictions/prediction_data_*.parquet --threshold 0.62 --min-confidence 70

# 4. Revisar CSV y actuar
cat data/predictions/predictions_*.csv
```

### Después de Resultados (Fin de Semana)

- Comparar predicciones vs resultados reales
- Calcular ROI real
- Ajustar thresholds si es necesario

### Mensual

```powershell
# Re-entrenar con datos frescos
python scripts/retrain_improved_pipeline.py --seasons 2425 2526

# Re-optimizar threshold
python scripts/optimize_threshold_production.py --save-model
```

---

## 🛡️ Validación y Seguridad

### ✅ Sin Data Leakage

Todas las features se calculan con `.shift(1)`:

```python
# Ejemplo en build_prediction_dataset.py
grouped['goals_scored'].shift(1).rolling(5).mean()
```

Esto asegura que solo usamos información **disponible antes del partido**.

### ✅ Backtest Realista

El sistema simula predicciones "antes" de que ocurran los partidos:

```python
# Para GW 23, usa solo datos de GW 1-22
historical_before_gw = df[df['gameweek'] < 23]
```

### ✅ Confidence Scoring Multi-Factor

Considera:
- Extremeness de probabilidad
- Distancia de threshold (2.5 goles)
- Data drift de features
- Contexto (derby, top6 clash)
- Volatilidad de equipos

---

## 📈 Métricas Esperadas

### Modelo Actual (después de optimización)

| Métrica | Valor Esperado | Notas |
|---------|----------------|-------|
| **Threshold** | 0.62-0.65 | Optimizado |
| **ROC AUC** | 0.55-0.60 | Modelo balanced |
| **Balanced Accuracy** | 0.57-0.62 | Target |
| **Sensitivity** | 55-65% | Detecta Over |
| **Specificity** | 60-70% | Detecta Under |
| **Pred Over Rate** | 50-55% | Balanceado |
| **Confidence (avg)** | 70-75% | Con V2 features |

### Backtest (GW 18-22)

| Métrica | Target | Realista |
|---------|--------|----------|
| **Accuracy** | ≥60% | 55-65% |
| **ROI** | +10%+ | +5% to +15% |
| **Predictions Made** | 60-70% fixtures | 65% |

**Nota:** Over/Under 2.5 es inherentemente difícil. Accuracy >55% y ROI positivo es **excelente**.

---

## 🔧 Troubleshooting

### Problema: "Feature X not found"
**Solución:**
```powershell
# Verificar que usas datos V2
python check_features.py
```

### Problema: Todas predicciones "Over"
**Solución:**
```powershell
# Threshold demasiado bajo, optimizar:
python scripts/optimize_threshold_production.py --save-model
```

### Problema: Confidence muy bajo (<50%)
**Solución:**
```powershell
# Posible data drift, re-entrenar:
python scripts/retrain_improved_pipeline.py --seasons 2425 2526
```

### Problema: Backtest ROI negativo
**Solución:**
- Subir `--min-confidence` a 75-80%
- Re-entrenar modelo con ventana temporal más corta
- Verificar data drift

---

## 📚 Arquitectura del Sistema

```
INPUT (Fixtures Futuros)
    ↓
[get_upcoming_fixtures.py]
    ↓
CSV con fixtures sin resultados
    ↓
[build_prediction_dataset.py]
    ↓
Dataset con 78 features V2
(calculadas sin data leakage)
    ↓
[predict_fixtures.py]
    ↓
Predicciones + Confidence + Recommendations
    ↓
CSV Actionable
```

### Features V2 (78 totales)

| Categoría | Cantidad | Ejemplos |
|-----------|----------|----------|
| Rolling stats | 30 | goals_scored_L5, clean_sheet_rate_L3 |
| League averages | 5 | league_avg_goals_L5 |
| Ratios | 10 | home_attack_ratio_L5, attack_defense_ratio |
| Momentum | 6 | home_goals_momentum, away_defense_momentum |
| Volatility | 7 | combined_volatility, goals_scored_volatility_L5 |
| Context | 9 | is_derby, is_top6_clash, home_position_L10 |
| Rest | 6 | home_days_rest, home_short_rest |
| Form | 2 | form_diff_L5, form_diff_L10 |

---

## 🎓 Mejores Prácticas

### 1. Empezar Conservador

- Primera semana: Solo 2-3 apuestas con confidence ≥80%
- Stakes: 10-20% de tu capital normal
- Registrar resultados meticulosamente

### 2. Monitorear y Ajustar

```
Después de 3-5 jornadas:
- Calcular ROI real
- Comparar vs backtest
- Ajustar min-confidence si es necesario
```

### 3. Re-entrenar Regularmente

```powershell
# Cada mes o cada 10 jornadas
python scripts/retrain_improved_pipeline.py --seasons 2425 2526
python scripts/optimize_threshold_production.py --save-model
```

### 4. Validar Siempre

```
Antes de cambios importantes:
- Hacer backtest nuevo
- Verificar drift
- Comparar métricas
```

---

## 📞 Soporte

### Documentación

- `QUICK_START_PREDICTIONS.md` - **Empieza aquí**
- `PREDICTION_SYSTEM_GUIDE.md` - Guía completa
- `DASHBOARD_SETUP.md` - Dashboard de monitoreo

### Logs

- `logs/retrain_*.log` - Entrenamiento
- `logs/app.log` - General

### Scripts de Diagnóstico

- `check_features.py` - Verificar features
- `python scripts/backtest_predictions.py` - Validar sistema

---

## ✅ Checklist Pre-Producción

- [ ] ✓ Optimizar threshold
- [ ] ✓ Hacer backtest (GW 18-22)
- [ ] ✓ Verificar accuracy ≥55%
- [ ] ✓ Verificar ROI positivo
- [ ] ✓ Ajustar min-confidence
- [ ] ✓ Probar con example_complete_workflow.py
- [ ] ✓ Empezar con stakes pequeños

---

## 🚀 Comando Para Empezar AHORA

```powershell
python example_complete_workflow.py
```

Tendrás predicciones de ejemplo en **< 1 minuto**.

Luego lee: `QUICK_START_PREDICTIONS.md`

---

**¡Buena suerte con las predicciones!** ⚽📊
