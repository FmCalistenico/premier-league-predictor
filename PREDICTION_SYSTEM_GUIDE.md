#

 Sistema Completo de Predicción - Guía Maestra

## Resumen Ejecutivo

Este sistema te permite predecir resultados Over/Under 2.5 goles para futuras jornadas de la Premier League con **scoring de confiabilidad**.

**Características:**
- ✅ Features V2 (78 features) sin data leakage
- ✅ Confidence scoring (0-100%)
- ✅ Threshold optimizado (0.60-0.65 vs 0.75 anterior)
- ✅ Detección de data drift
- ✅ Sistema de backtest
- ✅ Recomendaciones BET/SKIP automáticas

---

## Flujo Completo de Predicción

```
1. GET FIXTURES        2. BUILD DATASET       3. PREDICT           4. ANALYZE
   ↓                      ↓                      ↓                    ↓
get_upcoming_      build_prediction_    predict_fixtures    [Review CSV]
fixtures.py        dataset.py           .py                 + Dashboard

Input: API/Manual  Input: Fixtures      Input: Dataset      Output: Actionable
Output: CSV        Output: Parquet      Output: CSV         predictions
                   (78 features V2)      (with confidence)
```

---

## 1. OBTENER FIXTURES FUTUROS

### Opción A: Desde API-Football (Recomendado)

```powershell
# Configurar API key
$env:FOOTBALL_DATA_API_KEY = "tu_api_key_aqui"

# Obtener fixtures de GW 23
python scripts/get_upcoming_fixtures.py --from-api --gameweek 23

# Output: data/raw/upcoming_fixtures_GW23_20260104.csv
```

### Opción B: Input Manual (Sin API)

```powershell
python scripts/get_upcoming_fixtures.py --manual
```

Luego ingresa fixtures en formato:
```
> 2026-01-11,Arsenal,Man City,23
> 2026-01-11,Newcastle,Fulham,23
> [Enter dos veces para terminar]
```

### Opción C: Desde CSV Template

```powershell
# Crear template
python scripts/get_upcoming_fixtures.py --create-template data/raw/fixtures_template.csv

# Llenar manualmente el CSV, luego:
python scripts/get_upcoming_fixtures.py --from-csv data/raw/fixtures_template.csv
```

**Output:** `data/raw/upcoming_fixtures_GW{X}_{date}.csv`

---

## 2. CONSTRUIR DATASET DE PREDICCIÓN

Este paso calcula las 78 features V2 usando solo datos históricos (sin data leakage).

```powershell
python scripts/build_prediction_dataset.py \
    --input data/raw/upcoming_fixtures_GW23_20260104.csv \
    --gameweek 23

# Output: data/predictions/prediction_data_GW23_{timestamp}.parquet
```

**Lo que hace internamente:**
1. Carga datos históricos (`training_data_v2.parquet`)
2. Para cada fixture futuro:
   - Calcula rolling stats L3/L5/L10 con partidos anteriores
   - Obtiene posiciones en tabla actuales
   - Calcula días de descanso desde último partido
   - Genera ratios, momentum, volatility, context features
3. Valida que no haya NaN (rellena con promedios de liga si es necesario)
4. Exporta dataset listo para predicción

**Parámetros opcionales:**
```powershell
--historical data/custom/historical.parquet  # Usar otros datos históricos
--output data/custom/prediction_data.parquet # Custom output path
```

---

## 3. GENERAR PREDICCIONES CON CONFIDENCE

```powershell
python scripts/predict_fixtures.py \
    --input data/predictions/prediction_data_GW23_{timestamp}.parquet \
    --threshold 0.62 \
    --min-confidence 70

# Output: data/predictions/predictions_{timestamp}.csv
```

**Parámetros importantes:**
- `--threshold 0.62`: Threshold optimizado (en lugar del 0.75 default del modelo)
- `--min-confidence 70`: Solo recomendar BET si confidence ≥ 70%
- `--model models/custom_model.pkl`: Usar modelo personalizado

**Output CSV:**
```csv
fixture_id,date,home_team,away_team,prob_over,prob_under,prediction_label,confidence,expected_total_goals,recommendation
GW23_ARS_MCI,2026-01-11,Arsenal,Man City,0.68,0.32,Over 2.5,78,3.2,"BET: Over 2.5 (High Confidence)"
GW23_NEW_FUL,2026-01-11,Newcastle,Fulham,0.45,0.55,Under 2.5,82,2.3,"BET: Under 2.5 (High Confidence)"
GW23_TOT_BRE,2026-01-11,Tottenham,Brentford,0.52,0.48,Over 2.5,41,3.0,"SKIP: Low Confidence"
```

---

## 4. OPTIMIZAR THRESHOLD (UNA VEZ)

Ejecuta esto **antes de hacer predicciones de producción** para encontrar el threshold óptimo:

```powershell
python scripts/optimize_threshold_production.py \
    --metric custom \
    --target-sensitivity 0.55 \
    --target-specificity 0.60 \
    --save-model

# Output:
# - models/results/threshold_optimization.png (plots)
# - models/production_model_optimized_{timestamp}.pkl (modelo con threshold actualizado)
```

**Métricas disponibles:**
- `custom`: Encuentra threshold que cumple targets y está cerca de 50% pred over rate
- `balanced_accuracy`: Maximiza balanced accuracy
- `f1`: Maximiza F1 score
- `youden`: Maximiza Youden's J (sensitivity + specificity - 1)

**Resultado esperado:**
```
Optimal Threshold: 0.6250
Sensitivity: 0.571 (57.1%)
Specificity: 0.643 (64.3%)
Pred Over Rate: 52.3%
Balanced Accuracy: 0.607
```

Luego usa el modelo optimizado:
```powershell
python scripts/predict_fixtures.py \
    --input data/predictions/prediction_data_GW23.parquet \
    --model models/production_model_optimized_20260104.pkl
```

---

## 5. BACKTEST / VALIDACIÓN

Valida el sistema con jornadas pasadas (simula predicciones "antes" de que ocurrieran):

```powershell
python scripts/backtest_predictions.py \
    --start-gameweek 18 \
    --end-gameweek 22 \
    --min-confidence 70

# Output: models/results/backtest_report.csv
```

**Qué calcula:**
- Accuracy de predicciones por gameweek
- ROI simulado (asumiendo apuestas iguales)
- Calibration: ¿prob_over=0.7 realmente ocurre 70% de las veces?
- Drift: ¿Features siguen siendo estables?

**Output:**
```csv
gameweek,total_fixtures,predictions_made,correct,accuracy,roi,avg_confidence,calibration_error
18,10,7,5,0.714,+12.5%,76.2,0.08
19,10,8,5,0.625,-5.0%,72.1,0.11
20,10,6,4,0.667,+8.3%,78.5,0.06
```

---

## 6. MONITOREO DE DATA DRIFT

Ejecuta periódicamente para detectar si features están "driftando":

```powershell
python scripts/monitor_drift.py \
    --reference-data data/final/training_data_v2.parquet \
    --new-data data/predictions/prediction_data_GW23.parquet

# Output: models/results/drift_report.csv
```

**Features con drift alto (KS stat > 0.15):**
- `league_avg_goals_L10` (0.412) ← Re-entrenar modelo
- `league_avg_conceded_L10` (0.409)
- `combined_volatility` (0.167)

**Acciones:**
- KS < 0.10: OK, seguir usando modelo
- KS 0.10-0.20: Warning, monitorear
- KS > 0.20: Re-entrenar modelo con datos recientes

---

## Comandos Rápidos de Referencia

```powershell
# Pipeline completo (GW 23 ejemplo)

# 1. Get fixtures (manual)
python scripts/get_upcoming_fixtures.py --manual

# 2. Build dataset
python scripts/build_prediction_dataset.py --input data/raw/upcoming_fixtures_GW23.csv --gameweek 23

# 3. Predict
python scripts/predict_fixtures.py --input data/predictions/prediction_data_GW23_*.parquet --threshold 0.62 --min-confidence 70

# 4. Review
cat data/predictions/predictions_*.csv
```

**One-liner (si tienes fixtures en CSV):**
```powershell
python scripts/build_prediction_dataset.py --input fixtures.csv | python scripts/predict_fixtures.py --input - --threshold 0.62
```

---

## Confidence Scoring - Cómo Funciona

El confidence score (0-100%) se calcula con múltiples factores:

### Factor 1: Extremeness de Probabilidad (50% peso)
- `prob_over = 0.95` → Alta confianza (extremo)
- `prob_over = 0.52` → Baja confianza (cerca del 50%)

### Factor 2: Distancia del Threshold (±30%)
- `expected_goals = 1.8` (lejos de 2.5) → +10%
- `expected_goals = 2.6` (cerca de 2.5) → -30%

### Factor 3: Data Drift (hasta -40%)
- Si las features tienen drift alto → Penalizar confianza

### Factor 4: Contexto del Partido
- **Derby**: +5% (más predecible)
- **Top 6 clash**: -5% (más impredecible)
- **Alta volatilidad**: -15%

### Factor 5: Consistencia de Equipos
- Equipos con `volatility_L5 < 0.5` → +10%
- Equipos con `volatility_L5 > 1.5` → -15%

**Ejemplo:**
```
Arsenal vs Man City
  prob_over = 0.72 → extremeness = 0.44 → base = 72%
  expected_goals = 3.1 → far from 2.5 → +10% → 79.2%
  top6_clash = True → -5% → 75.3%
  avg_volatility = 0.6 → no change → 75.3%

  Final Confidence: 75%
```

---

## Interpretación de Recomendaciones

### BET: High Confidence (Confidence ≥ 80%)
```
Arsenal vs Man City | Over 2.5 (72%) | Confidence: 85% | BET: Over 2.5 (High Confidence)
```
- **Acción**: Apostar con confianza alta
- **Probabilidad modelo**: 72% Over
- **Confiabilidad**: 85%

### BET: Medium Confidence (65% ≤ Confidence < 80%)
```
Newcastle vs Fulham | Under 2.5 (58%) | Confidence: 72% | BET: Under 2.5 (Medium Confidence)
```
- **Acción**: Apostar con cautela (menor stake)
- **Probabilidad modelo**: 58% Under
- **Confiabilidad**: 72%

### SKIP: Low Confidence (Confidence < 65%)
```
Tottenham vs Brentford | Over 2.5 (52%) | Confidence: 41% | SKIP: Low Confidence
```
- **Acción**: NO apostar
- **Razón**: Probabilidad muy cercana al 50%, alta incertidumbre

---

## Troubleshooting

### Error: "Feature X not found"
**Causa**: El modelo espera features V2 pero el dataset tiene V1

**Solución**:
```powershell
# Verificar que usas training_data_v2.parquet (no training_data_latest.parquet)
python check_features.py
```

### Error: "NaN values in features"
**Causa**: Equipo con historial insuficiente (< 10 partidos)

**Solución**: El script rellena automáticamente con promedios de liga. Revisa:
```powershell
# Ver warnings en logs
cat logs/retrain_*.log | grep NaN
```

### Predicciones todas "Over" o todas "Under"
**Causa**: Threshold mal configurado

**Solución**:
```powershell
# Re-optimizar threshold
python scripts/optimize_threshold_production.py --save-model

# Usar el nuevo modelo
python scripts/predict_fixtures.py --model models/production_model_optimized_*.pkl
```

### Confidence scores muy bajos
**Causa**: Data drift alto o features muy volátiles

**Solución**:
```powershell
# 1. Monitorear drift
python scripts/monitor_drift.py

# 2. Si drift > 0.20, re-entrenar con datos recientes:
python scripts/retrain_improved_pipeline.py --seasons 2425 2526
```

---

## Best Practices

### 1. Re-entrenar Cada 5-10 Jornadas
```powershell
# Cada mes aproximadamente
python scripts/retrain_improved_pipeline.py --seasons 2425 2526
python scripts/optimize_threshold_production.py --save-model
```

### 2. Validar con Backtest Antes de Producción
```powershell
# Simular últimas 5 jornadas
python scripts/backtest_predictions.py --start-gameweek 17 --end-gameweek 22

# Si accuracy < 55% → Revisar modelo
```

### 3. Ajustar min-confidence Según Resultados
```
Si backtest muestra:
  - ROI positivo con confidence ≥ 70% → Usar 70%
  - ROI negativo con confidence ≥ 70% → Subir a 75-80%
```

### 4. Monitorear Drift Mensualmente
```powershell
python scripts/monitor_drift.py
# Si > 5 features con KS > 0.20 → Re-entrenar urgente
```

### 5. Mantener Log de Predicciones
```
Guardar todas las predicciones CSV para análisis posterior:
  data/predictions/archive/predictions_GW23_20260111.csv
  data/predictions/archive/predictions_GW24_20260118.csv
```

---

## Próximos Pasos Recomendados

### Semana 1: Setup y Validación
- [x] Instalar sistema
- [x] Optimizar threshold
- [ ] Hacer backtest de últimas 5 jornadas
- [ ] Ajustar min-confidence basado en resultados

### Semana 2: Primera Predicción Real
- [ ] Obtener fixtures de próxima jornada
- [ ] Generar predicciones
- [ ] Hacer 2-3 apuestas de prueba (stakes bajos)
- [ ] Registrar resultados

### Semana 3-4: Monitoreo
- [ ] Comparar predicciones vs resultados reales
- [ ] Calcular ROI real
- [ ] Ajustar confidence thresholds si es necesario
- [ ] Monitorear drift

### Mes 2: Optimización
- [ ] Re-entrenar modelo con datos frescos
- [ ] Analizar features que mejor predicen
- [ ] Considerar ensemble con otros modelos
- [ ] Implementar dashboard de monitoreo

---

## Contacto y Ayuda

**Documentación completa:**
- `README.md` - Visión general del proyecto
- `DASHBOARD_SETUP.md` - Setup del dashboard
- `SOLUTIONS_MODEL_BIAS.md` - Soluciones para sesgo
- `FEATURE_ENGINEERING_COMPARISON.md` - V1 vs V2 features

**Logs:**
- `logs/retrain_*.log` - Logs de entrenamiento
- `logs/app.log` - Logs generales

**Scripts útiles:**
- `check_features.py` - Verificar features en datos
- `run_complete_pipeline.py` - Pipeline completo de entrenamiento

---

**¡Buena suerte con las predicciones!** ⚽📊
