# Troubleshooting Guide

Soluciones a problemas comunes en el proyecto Premier League Predictor.

---

## 🔧 Problemas Comunes

### 1. Error: "['home_team'] not in index"

**Síntoma:**
```
KeyError: "['home_team'] not in index"
```

**Causa:**
El DataFrame tiene columnas con nombres diferentes (`home_team_name` en lugar de `home_team`).

**Solución:**

El `FeatureEngineerV2` ahora normaliza automáticamente los nombres de columnas. Si encuentras este error:

```python
# El FeatureEngineerV2 normaliza automáticamente
from src.features import FeatureEngineerV2

engineer = FeatureEngineerV2()
df_features = engineer.engineer_features(df)  # Normaliza home_team_name -> home_team
```

Si usas datos raw, asegúrate de tener:
- `home_team` o `home_team_name`
- `away_team` o `away_team_name`
- `home_goals`
- `away_goals`
- `date`

---

### 2. Error: "training_data_latest.parquet not found"

**Síntoma:**
```
FileNotFoundError: training_data_latest.parquet not found
```

**Causa:**
No has ejecutado el pipeline de datos.

**Solución:**
```bash
# Opción 1: Pipeline completo
python run_complete_pipeline.py

# Opción 2: Solo datos
python run_complete_pipeline.py --skip-cv

# Opción 3: Modo rápido
python run_complete_pipeline.py --quick
```

---

### 3. Error: "No module named 'src'"

**Síntoma:**
```
ModuleNotFoundError: No module named 'src'
```

**Causa:**
Python no encuentra el módulo `src` en el path.

**Solución:**
```bash
# Opción 1: Instalar en modo desarrollo
pip install -e .

# Opción 2: Ejecutar desde la raíz del proyecto
cd premier-league-predictor
python scripts/train_model_v2.py

# Opción 3: Agregar al PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/premier-league-predictor"
```

---

### 4. Modelo predice 94% Over

**Síntoma:**
El modelo predice "Over 2.5" en el 94% de casos.

**Causa:**
- Threshold por defecto (0.5) no ajustado al base rate (59.5%)
- Features problemáticas con correlación circular
- Clase imbalance no tratado

**Solución:**

Ver [SOLUTIONS_MODEL_BIAS.md](SOLUTIONS_MODEL_BIAS.md) para soluciones completas.

**Quick fix:**
```bash
# Optimizar threshold
python scripts/optimize_threshold.py --metric balanced_accuracy
```

**Solución completa:**
```bash
# Usar FeatureEngineerV2
python scripts/train_model_v2.py --validate-features
```

---

### 5. ROC AUC cercano a 0.5 (random)

**Síntoma:**
ROC AUC entre 0.48-0.52 en cross-validation.

**Causa:**
- Features con alta varianza entre folds
- Data leakage en features
- Features muy correlacionadas

**Solución:**
```bash
# Analizar features problemáticas
python scripts/analyze_features.py

# Ver features con VIF > 10
cat models/analysis/feature_vif.csv

# Migrar a V2
python scripts/migrate_to_v2.py
```

---

### 6. Error al calcular VIF

**Síntoma:**
```
LinAlgError: Singular matrix
```

**Causa:**
Multicolinealidad perfecta entre features.

**Solución:**
```python
# FeatureEngineerV2 ya filtra features problemáticas
from src.features import FeatureEngineerV2

engineer = FeatureEngineerV2()
df_features = engineer.engineer_features(df)

# Validar VIF
validation = engineer.validate_features(df_features)
print(f"Features con VIF > 10: {validation['vif_issues']}")
```

---

### 7. Warning: "Found X invalid predictions (NaN/inf)"

**Síntoma:**
```
WARNING: Found 15 invalid predictions (NaN/inf), replacing with 0.5
```

**Causa:**
El modelo Poisson produce valores NaN o infinitos en algunos casos.

**Solución:**
Ya está manejado automáticamente en `model_pipeline.py`:
```python
# Se limpia automáticamente
y_pred_proba = np.clip(y_pred_proba, 0.0, 1.0)
```

Si persiste:
```bash
# Verificar datos de entrada
python scripts/analyze_features.py
```

---

### 8. Cross-validation muy lento

**Síntoma:**
CV con 5 folds tarda >10 minutos.

**Causa:**
Dataset grande o muchas features.

**Solución:**
```bash
# Reducir número de folds
python scripts/train_model_v2.py --cv-splits 3

# Usar modo rápido
python run_complete_pipeline.py --quick --skip-cv
```

---

### 9. Features con alta correlación

**Síntoma:**
Validación muestra muchas features con correlación > 0.7 con target.

**Causa:**
Posible data leakage o features circulares.

**Solución:**
```bash
# Usar V2 que ya filtra features problemáticas
python scripts/train_model_v2.py --validate-features

# Ver reporte de validación
cat models/analysis/feature_validation_v2.csv
```

---

### 10. Error: "Classification metrics can't handle a mix of binary and multilabel-indicator targets"

**Síntoma:**
```
ValueError: Classification metrics can't handle a mix of binary and multilabel-indicator targets
```

**Causa:**
`predict_proba()` retorna array 2D `[[prob_under, prob_over], ...]` pero el código espera solo las probabilidades de la clase positiva.

**Solución:**
```python
# Extraer solo columna de clase positiva (Over = 1)
y_pred_proba = model.predict_proba(X_test)

if y_pred_proba.ndim == 2:
    y_pred_proba = y_pred_proba[:, 1]  # Columna de "Over"
```

Ya está corregido en `migrate_to_v2.py`.

---

### 11. Error: "NaN, inf or invalid value detected in weights"

**Síntoma:**
```
ValueError: NaN, inf or invalid value detected in weights, estimation infeasible.
RuntimeWarning: invalid value encountered in divide
RuntimeWarning: divide by zero encountered in divide
```

**Causa:**
Features V2 contienen valores extremos, NaN, infinitos o varianza cero que causan problemas numéricos en Poisson GLM.

**Solución:**
```bash
# Diagnosticar qué features causan problemas
python scripts/diagnose_features.py

# Ver diagnóstico
cat models/analysis/feature_diagnostics_v2.txt
```

Ya está corregido en `PoissonGoalsModel.fit()` con:
- Clipping de valores extremos (>1e6)
- Eliminación de features con varianza cero
- Manejo robusto de NaN/Inf

Si persiste, el modelo ahora elimina automáticamente features problemáticas.

---

### 12. Error: "Min samples must be > 1"

**Síntoma:**
```
ValueError: min_samples must be > 1
```

**Causa:**
Dataset demasiado pequeño para rolling features.

**Solución:**
```bash
# Usar más temporadas
python run_complete_pipeline.py --seasons 2223 2324 2425

# O usar datos de API
python run_complete_pipeline.py --source api
```

---

## 🧪 Tests y Diagnóstico

### Verificar instalación
```bash
# Test rápido
python scripts/test_v2_quick.py

# Tests unitarios
pytest tests/ -v

# Tests específicos V2
pytest tests/test_engineering_v2.py -v
```

### Diagnóstico de datos
```bash
# Ver estadísticas de datos procesados
python -c "
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path('data/processed').glob('transformed_*.csv').__next__())
print(f'Matches: {len(df)}')
print(f'Columns: {df.columns.tolist()}')
print(f'Date range: {df.date.min()} to {df.date.max()}')
print(f'Over rate: {(df.total_goals > 2.5).mean():.1%}')
"
```

### Diagnóstico de modelo
```bash
# Comparar V1 vs V2
python scripts/compare_models.py

# Analizar features
python scripts/analyze_features.py

# Optimizar threshold
python scripts/optimize_threshold.py --metric balanced_accuracy
```

---

## 📚 Recursos

### Documentación
- [QUICKSTART.md](QUICKSTART.md) - Inicio rápido
- [SOLUTIONS_MODEL_BIAS.md](SOLUTIONS_MODEL_BIAS.md) - Soluciones al sesgo
- [FEATURE_ENGINEERING_COMPARISON.md](FEATURE_ENGINEERING_COMPARISON.md) - V1 vs V2

### Scripts útiles
- `scripts/test_v2_quick.py` - Test rápido de V2
- `scripts/migrate_to_v2.py` - Migración V1 a V2
- `scripts/compare_models.py` - Comparación de modelos
- `scripts/analyze_features.py` - Análisis de features
- `scripts/optimize_threshold.py` - Optimización de threshold

---

## 🆘 Soporte

Si ninguna solución funciona:

1. **Verificar logs:**
   ```bash
   ls -la logs/
   tail -n 50 logs/latest.log
   ```

2. **Limpiar y reiniciar:**
   ```bash
   # Limpiar datos procesados
   rm -rf data/processed/*
   rm -rf data/final/*

   # Re-ejecutar pipeline
   python run_complete_pipeline.py
   ```

3. **Reinstalar dependencias:**
   ```bash
   pip install -r requirements.txt --upgrade
   pip install -e .
   ```

4. **Reportar issue:**
   - Incluir: error completo, logs, comando ejecutado
   - Ubicación: GitHub Issues

---

**Última actualización:** 2026-01-02
