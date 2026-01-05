# Quick Start - Sistema de Predicción

## 🚀 Usa el Sistema AHORA (5 minutos)

### Opción 1: Workflow Completo Automático

```powershell
# Ejecuta el ejemplo completo
python example_complete_workflow.py
```

Esto generará predicciones de ejemplo para 4 partidos y te mostrará:
- ✅ Predicciones con confidence scoring
- ✅ Recomendaciones BET/SKIP
- ✅ Expected goals
- ✅ Resumen de insights accionables

**Output:** `data/predictions/example_predictions_GW23.csv`

---

### Opción 2: Predicción Manual Paso a Paso

#### Paso 1: Crear Fixtures (Input Manual)

```powershell
python scripts/get_upcoming_fixtures.py --manual
```

Ingresa tus fixtures:
```
> 2026-01-11,Arsenal,Man City,23
> 2026-01-11,Newcastle,Fulham,23
> [Enter dos veces]
```

**Output:** `data/raw/upcoming_fixtures_GW23_20260104.csv`

#### Paso 2: Construir Dataset con Features V2

```powershell
python scripts/build_prediction_dataset.py \
    --input data/raw/upcoming_fixtures_GW23_20260104.csv \
    --gameweek 23
```

**Output:** `data/predictions/prediction_data_GW23_{timestamp}.parquet`

#### Paso 3: Generar Predicciones

```powershell
python scripts/predict_fixtures.py \
    --input data/predictions/prediction_data_GW23_*.parquet \
    --threshold 0.62 \
    --min-confidence 70
```

**Output:** `data/predictions/predictions_{timestamp}.csv`

#### Paso 4: Ver Resultados

```powershell
cat data/predictions/predictions_*.csv
```

O abre el CSV en Excel/Google Sheets.

---

## ⚡ Comandos One-Liner

### Predicción Rápida (Manual Input)

```powershell
python scripts/get_upcoming_fixtures.py --manual && python scripts/build_prediction_dataset.py --input data/raw/upcoming_fixtures_*.csv --gameweek 23 && python scripts/predict_fixtures.py --input data/predictions/prediction_data_GW23_*.parquet --threshold 0.62
```

### Backtest Rápido (Validar Sistema)

```powershell
# Validar últimas 5 jornadas
python scripts/backtest_predictions.py --start-gameweek 18 --end-gameweek 22 --min-confidence 70
```

Esto te dirá:
- ✅ Accuracy real del sistema
- ✅ ROI simulado
- ✅ Si el modelo funciona bien

---

## 📊 Interpretar Resultados

### Archivo CSV de Predicciones

```csv
fixture_id,date,home_team,away_team,prob_over,prediction_label,confidence,recommendation
GW23_ARS_MCI,2026-01-11,Arsenal,Man City,0.72,Over 2.5,85,"BET: Over 2.5 (High Confidence)"
```

**Columnas importantes:**
- `prob_over`: Probabilidad de Over 2.5 según el modelo
- `confidence`: Confiabilidad 0-100%
- `recommendation`: Acción sugerida

### Cómo Actuar

#### Confidence ≥ 80% (Very High)
```
Arsenal vs Man City | Over 2.5 (72%) | Confidence: 85%
→ ACCIÓN: Apostar con confianza (stake normal)
```

#### Confidence 65-79% (High/Medium)
```
Newcastle vs Fulham | Under 2.5 (58%) | Confidence: 72%
→ ACCIÓN: Apostar con cautela (stake reducido)
```

#### Confidence < 65% (Low)
```
Tottenham vs Brentford | Over 2.5 (52%) | Confidence: 41%
→ ACCIÓN: NO apostar (muy incierto)
```

---

## 🔧 Antes de Usar en Producción

### 1. Optimizar Threshold (IMPORTANTE)

El threshold default del modelo (0.75) es muy conservador. Optimízalo:

```powershell
python scripts/optimize_threshold_production.py \
    --metric custom \
    --target-sensitivity 0.55 \
    --target-specificity 0.60 \
    --save-model
```

**Output:**
- Threshold óptimo: ~0.62-0.65
- Modelo guardado: `models/production_model_optimized_*.pkl`

Luego usa el modelo optimizado:
```powershell
python scripts/predict_fixtures.py \
    --model models/production_model_optimized_*.pkl \
    --input data/predictions/prediction_data_GW23.parquet
```

### 2. Validar con Backtest

```powershell
# Simular predicciones de jornadas pasadas
python scripts/backtest_predictions.py --start-gameweek 18 --end-gameweek 22

# Output: models/results/backtest_report_*.csv
```

**Métricas esperadas:**
- Accuracy: 55-65%
- ROI: Positivo (esperanza: +5% to +15%)
- Calibration error: < 0.10

**Si backtest falla** (accuracy < 50% o ROI negativo):
- ❌ NO uses el sistema aún
- ✅ Re-entrena modelo con datos más recientes
- ✅ Ajusta min-confidence más alto (75-80%)

### 3. Ajustar Confidence Threshold

Basado en backtest:

```
Si backtest muestra:
  ROI positivo con confidence ≥ 70% → Usar --min-confidence 70
  ROI negativo con confidence ≥ 70% → Subir a --min-confidence 75
  ROI positivo solo con confidence ≥ 80% → Usar --min-confidence 80
```

---

## ❓ Troubleshooting Rápido

### Error: "Feature X not found"
```powershell
# Solución: Verificar que usas datos V2
python check_features.py
```

### Predicciones todas "Over" o todas "Under"
```powershell
# Solución: Optimizar threshold
python scripts/optimize_threshold_production.py --save-model
```

### Confidence scores muy bajos (<50%)
```powershell
# Posible data drift alto
# Solución: Re-entrenar modelo
python scripts/retrain_improved_pipeline.py --seasons 2425 2526
```

### Error: "Model not found"
```powershell
# Solución: Verificar ruta del modelo
ls models/results/

# Si no existe, re-entrenar:
python scripts/retrain_improved_pipeline.py --models poisson_balanced
```

---

## 📁 Archivos Importantes

### Inputs
- `data/raw/upcoming_fixtures_*.csv` - Fixtures futuros
- `data/final/training_data_v2.parquet` - Datos históricos con V2 features

### Outputs
- `data/predictions/prediction_data_*.parquet` - Dataset con features
- `data/predictions/predictions_*.csv` - Predicciones finales (ABRIR ESTO)

### Models
- `models/results/retrain_checkpoint.pkl` - Modelo entrenado
- `models/production_model_optimized_*.pkl` - Modelo con threshold optimizado

### Logs
- `logs/retrain_*.log` - Logs de entrenamiento
- `logs/app.log` - Logs generales

---

## 🎯 Workflow Recomendado (Semanal)

### Lunes/Martes (Antes de Fixtures)
```powershell
# 1. Obtener fixtures de próxima jornada
python scripts/get_upcoming_fixtures.py --manual  # O desde API

# 2. Generar dataset
python scripts/build_prediction_dataset.py --input data/raw/upcoming_fixtures_*.csv

# 3. Predecir
python scripts/predict_fixtures.py --input data/predictions/prediction_data_*.parquet --threshold 0.62 --min-confidence 70

# 4. Revisar CSV y hacer apuestas
cat data/predictions/predictions_*.csv
```

### Fin de Semana (Después de Resultados)
```powershell
# 1. Comparar predicciones vs resultados reales
# 2. Calcular ROI real
# 3. Ajustar confidence thresholds si es necesario
```

### Mensual
```powershell
# Re-entrenar modelo con datos frescos
python scripts/retrain_improved_pipeline.py --seasons 2425 2526

# Re-optimizar threshold
python scripts/optimize_threshold_production.py --save-model
```

---

## 📚 Documentación Completa

- `PREDICTION_SYSTEM_GUIDE.md` - Guía maestra completa
- `DASHBOARD_SETUP.md` - Setup del dashboard
- `README.md` - Visión general del proyecto

---

## ✅ Checklist Antes de Primera Predicción Real

- [ ] ✓ Optimizar threshold (`optimize_threshold_production.py`)
- [ ] ✓ Hacer backtest (`backtest_predictions.py --start-gameweek 18 --end-gameweek 22`)
- [ ] ✓ Verificar backtest accuracy ≥ 55%
- [ ] ✓ Verificar backtest ROI positivo
- [ ] ✓ Ajustar min-confidence basado en backtest
- [ ] ✓ Hacer predicción de prueba con `example_complete_workflow.py`
- [ ] ✓ Empezar con stakes pequeños (10-20% de capital normal)
- [ ] ✓ Hacer solo 2-3 apuestas la primera semana
- [ ] ✓ Registrar resultados para análisis posterior

---

**¡Listo para empezar!** 🚀⚽

Ejecuta:
```powershell
python example_complete_workflow.py
```

Y tendrás tus primeras predicciones en menos de 1 minuto.
