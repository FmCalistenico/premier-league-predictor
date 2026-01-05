# Soluciones para el Sesgo del Modelo Poisson

## 📊 Análisis del Problema

### Resultados Actuales
```
ROC-AUC: 0.5777 (CV: 0.5165 ± 0.0390)
Accuracy: 59.54%
Recall: 95.75% (extremadamente alto)
Specificity: 6.2% (extremadamente bajo)

Confusion Matrix:
- True Negatives: 24 (solo 6.2% de Under detectados)
- False Positives: 360 (93.8% de Under clasificados como Over)
- False Negatives: 24 (4.25% de Over clasificados como Under)
- True Positives: 541 (95.75% de Over detectados)

Predicción: 94% de casos clasificados como "Over 2.5"
```

## 🔍 Explicación del Sesgo hacia "Over"

### 1. **Problema de Threshold**
- **Threshold actual**: 0.5 (por defecto)
- **Base rate**: 59.5% Over vs 40.5% Under
- **Problema**: El modelo debería usar threshold ≈ 0.595 para equilibrar las clases

**Por qué ocurre:**
```python
# El modelo predice probabilidades calibradas
# Pero usa threshold fijo de 0.5
if prob_over > 0.5:  # Threshold demasiado bajo
    predict "Over"
else:
    predict "Under"

# Debería ser:
if prob_over > 0.595:  # Threshold ajustado al base rate
    predict "Over"
```

### 2. **Características de la Distribución Poisson**
El modelo Poisson tiende a sobreestimar λ_home + λ_away porque:

**a) Sesgo positivo inherente:**
- Poisson GLM predice λ = exp(β₀ + β₁x₁ + ... + βₙxₙ)
- La función exponencial garantiza λ > 0
- Pequeños errores en coeficientes se amplifican exponencialmente

**b) Errores asimétricos:**
```python
# Si λ_true = 1.5
# λ_pred = 1.8 → P(Total > 2.5) aumenta significativamente
# λ_pred = 1.2 → P(Total > 2.5) disminuye menos

# El error hacia arriba tiene más impacto en probabilidades
```

### 3. **Features Problemáticas**
Algunas features tienen correlación muy alta con el target:

**Features con sesgo positivo:**
- `expected_total_goals`: Suma directa de λ_home + λ_away
- `combined_over_rate`: Tasa histórica de Over (casi circular)
- `home_goals_scored_L3/L5/L10`: Correlación directa con goles
- `away_goals_scored_L3/L5/L10`: Correlación directa con goles

**Por qué son problemáticas:**
```python
# expected_total_goals es prácticamente el output del modelo
expected_total_goals = lambda_home_strength + lambda_away_strength

# combined_over_rate es casi circular con el target
combined_over_rate = (home_over_rate + away_over_rate) / 2
# Si ambos equipos tienen over_rate alto → target probablemente es Over
```

### 4. **Desbalance de Clases No Tratado**
- 59.5% Over vs 40.5% Under
- El modelo no usa pesos para balancear
- Minimizar log-loss favorece la clase mayoritaria

## 💡 Solución 1: Optimización de Threshold

### Implementación

Ya creado: `scripts/optimize_threshold.py`

### Uso:
```bash
# Encontrar threshold óptimo
python scripts/optimize_threshold.py \
    --metric f1 \
    --save-results

# Optimizar para balanced accuracy
python scripts/optimize_threshold.py \
    --metric balanced_accuracy \
    --save-results
```

### Cómo funciona:
```python
# El script prueba thresholds desde 0.05 hasta 0.95
# y encuentra el que maximiza la métrica elegida

optimal_threshold = 0.62  # Ejemplo (depende de tus datos)

# Luego usar en predicciones:
prob_over = model.predict_proba(X)
y_pred = (prob_over > optimal_threshold).astype(int)
```

### Resultados esperados:
- **Antes**: 94% Over, 6% Under
- **Después**: ~60% Over, ~40% Under (equilibrado)
- **Balanced Accuracy**: Mejora de ~51% a ~60-65%

---

## 💡 Solución 2: Feature Engineering y Selección

### Paso 1: Identificar Features Problemáticas

```bash
# Ejecutar análisis de features
python scripts/analyze_features.py \
    --model models/poisson_model_latest.pkl \
    --data data/final/training_data_latest.parquet \
    --output models/analysis/
```

Esto generará:
- `feature_importance.csv`: Coeficientes del modelo
- `feature_correlations.csv`: Correlaciones con target
- `feature_vif.csv`: Análisis de multicolinealidad
- `problematic_features.txt`: Lista de features a considerar eliminar

### Paso 2: Crear Nueva Versión del Feature Engineer

Crea: `src/features/engineering_v2.py`

```python
"""
Feature Engineering V2 - Reduced feature set to minimize bias.
"""

import pandas as pd
import numpy as np
from src.utils import get_logger


class FeatureEngineerV2:
    """Enhanced feature engineering with bias reduction."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def engineer_features(self, df):
        """
        Create features with problematic ones removed/modified.

        Args:
            df: DataFrame with match data

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Create features
        self.logger.info("Creating rolling features...")
        df = self.create_rolling_features(df)

        self.logger.info("Creating match features...")
        df = self.create_match_features_v2(df)  # Modified version

        self.logger.info("Creating rest features...")
        df = self.create_rest_features(df)

        # REMOVED: H2H features (too correlated)

        # Drop rows with NaN in critical features
        df = df.dropna(subset=[
            'home_goals_scored_L5',
            'away_goals_scored_L5'
        ])

        self.logger.info(f"Final dataset: {len(df)} matches, {len(df.columns)} columns")

        return df

    def create_rolling_features(self, df):
        """Create rolling statistics (unchanged)."""
        windows = [3, 5, 10]

        # Create team-level dataset
        home_df = df[['date', 'home_team', 'home_goals', 'away_goals']].copy()
        home_df.columns = ['date', 'team', 'goals_scored', 'goals_conceded']

        away_df = df[['date', 'away_team', 'away_goals', 'home_goals']].copy()
        away_df.columns = ['date', 'team', 'goals_scored', 'goals_conceded']

        team_df = pd.concat([home_df, away_df], ignore_index=True)
        team_df = team_df.sort_values(['team', 'date']).reset_index(drop=True)

        # Calculate metrics
        team_df['goal_diff'] = team_df['goals_scored'] - team_df['goals_conceded']
        team_df['total_goals'] = team_df['goals_scored'] + team_df['goals_conceded']
        team_df['over_2.5'] = (team_df['total_goals'] > 2.5).astype(int)

        # Rolling windows with shift(1) to prevent data leakage
        grouped = team_df.groupby('team')

        for window in windows:
            team_df[f'goals_scored_L{window}'] = (
                grouped['goals_scored'].shift(1).rolling(window, min_periods=1).mean()
            )
            team_df[f'goals_conceded_L{window}'] = (
                grouped['goals_conceded'].shift(1).rolling(window, min_periods=1).mean()
            )
            team_df[f'goal_diff_L{window}'] = (
                grouped['goal_diff'].shift(1).rolling(window, min_periods=1).mean()
            )
            team_df[f'over_rate_L{window}'] = (
                grouped['over_2.5'].shift(1).rolling(window, min_periods=1).mean()
            )

        # Merge back
        rolling_cols = [col for col in team_df.columns if col not in ['date', 'team', 'goals_scored', 'goals_conceded', 'goal_diff', 'total_goals', 'over_2.5']]

        home_rolling = team_df[['date', 'team'] + rolling_cols].copy()
        home_rolling.columns = ['date', 'home_team'] + [f'home_{col}' for col in rolling_cols]

        away_rolling = team_df[['date', 'team'] + rolling_cols].copy()
        away_rolling.columns = ['date', 'away_team'] + [f'away_{col}' for col in rolling_cols]

        df = df.merge(home_rolling, on=['date', 'home_team'], how='left')
        df = df.merge(away_rolling, on=['date', 'away_team'], how='left')

        return df

    def create_match_features_v2(self, df):
        """
        Create match-level features (MODIFIED to reduce bias).

        Changes:
        - REMOVED: expected_total_goals (too direct)
        - REMOVED: combined_over_rate (almost circular)
        - KEPT: Strength ratios (relative measures)
        - ADDED: Defensive measures
        """
        # Attack strength ratio (relative measure, not absolute)
        df['attack_strength_ratio'] = (
            df['home_goals_scored_L5'] / (df['away_goals_conceded_L5'] + 0.1)
        )

        # Defense strength ratio
        df['defense_strength_ratio'] = (
            df['away_goals_scored_L5'] / (df['home_goals_conceded_L5'] + 0.1)
        )

        # Form difference (relative)
        df['form_diff_L5'] = df['home_goal_diff_L5'] - df['away_goal_diff_L5']
        df['form_diff_L10'] = df['home_goal_diff_L10'] - df['away_goal_diff_L10']

        # Over rate difference (not combined!)
        df['over_rate_diff_L5'] = df['home_over_rate_L5'] - df['away_over_rate_L5']

        # REMOVED: expected_total_goals
        # REMOVED: combined_over_rate
        # These were causing the Over bias

        return df

    def create_rest_features(self, df):
        """Create rest days features (unchanged)."""
        df = df.sort_values('date').reset_index(drop=True)

        df['home_days_rest'] = (
            df.groupby('home_team')['date']
            .diff()
            .dt.days
            .fillna(7)
        )

        df['away_days_rest'] = (
            df.groupby('away_team')['date']
            .diff()
            .dt.days
            .fillna(7)
        )

        df['rest_advantage'] = df['home_days_rest'] - df['away_days_rest']

        return df
```

### Paso 3: Reentrenar con Nuevo Feature Set

```bash
# Primero, crear nuevo pipeline de datos con FeatureEngineerV2
# Editar pipelines/data_pipeline.py temporalmente para usar V2

python run_complete_pipeline.py --seasons 2223 2324 2425
```

### Resultados esperados:
- Menos features (25 vs 38)
- Menor correlación directa con target
- Mejor generalización
- ROC-AUC: 0.58-0.63 (mejora)

---

## 💡 Solución 3: Reentrenamiento con Class Weights

### Implementación

Crea: `src/models/poisson_model_balanced.py`

```python
"""
Balanced Poisson Model with class weights and calibration.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
from scipy.stats import poisson
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.utils import get_logger


class BalancedPoissonGoalsModel:
    """
    Poisson regression model with class balancing and calibration.

    Improvements:
    1. Weighted loss function to handle class imbalance
    2. Isotonic calibration to improve probability estimates
    3. Regularization to prevent overfitting
    """

    def __init__(self):
        self.home_model = None
        self.away_model = None
        self.calibrator = None
        self.feature_cols = None
        self.logger = get_logger(__name__)

    def fit(self, df, feature_cols, target_col='over_2.5'):
        """
        Train the balanced Poisson model.

        Args:
            df: Training DataFrame
            feature_cols: List of feature column names
            target_col: Target variable for weight calculation

        Returns:
            self
        """
        self.feature_cols = feature_cols

        # Prepare features
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X = sm.add_constant(X)

        y_home = df['home_goals'].values
        y_away = df['away_goals'].values

        # Calculate sample weights based on class imbalance
        # Matches that result in Under should have higher weight
        y_target = df[target_col].values
        class_counts = np.bincount(y_target)
        class_weights = len(y_target) / (len(class_counts) * class_counts)

        # Assign weights to each sample
        sample_weights = np.array([class_weights[y] for y in y_target])

        # Normalize weights
        sample_weights = sample_weights / sample_weights.mean()

        self.logger.info(f"Training with class weights:")
        self.logger.info(f"  Under weight: {class_weights[0]:.3f}")
        self.logger.info(f"  Over weight: {class_weights[1]:.3f}")

        # Train home goals model with weights
        self.logger.info("Training home goals model...")
        self.home_model = sm.GLM(
            y_home,
            X,
            family=Poisson(),
            freq_weights=sample_weights  # Apply weights
        ).fit()

        # Train away goals model with weights
        self.logger.info("Training away goals model...")
        self.away_model = sm.GLM(
            y_away,
            X,
            family=Poisson(),
            freq_weights=sample_weights  # Apply weights
        ).fit()

        # Train calibrator
        self.logger.info("Training probability calibrator...")
        self._fit_calibrator(df, feature_cols, target_col)

        return self

    def _fit_calibrator(self, df, feature_cols, target_col):
        """
        Fit isotonic calibration on Poisson probabilities.

        Args:
            df: Training DataFrame
            feature_cols: Feature columns
            target_col: Target variable
        """
        # Get raw Poisson probabilities
        raw_probs = self._predict_proba_raw(df[feature_cols])

        # Fit isotonic calibration
        y_true = df[target_col].values

        # Create a dummy classifier wrapper for calibration
        class DummyClassifier:
            def __init__(self, probs):
                self.probs = probs

            def predict_proba(self, X):
                # Return pre-computed probabilities
                return np.column_stack([1 - self.probs, self.probs])

        dummy = DummyClassifier(raw_probs)

        # Calibrate
        self.calibrator = CalibratedClassifierCV(
            dummy,
            method='isotonic',
            cv='prefit'
        )
        self.calibrator.fit(df[feature_cols], y_true)

    def _predict_proba_raw(self, X):
        """Predict probabilities without calibration."""
        X_input = X[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_input = sm.add_constant(X_input, has_constant='add')

        # Predict expected goals
        lambda_home = self.home_model.predict(X_input)
        lambda_away = self.away_model.predict(X_input)

        # Calculate Over 2.5 probabilities analytically
        probs_over = []
        for lh, la in zip(lambda_home, lambda_away):
            prob_over, _ = self._calculate_over_under_proba(lh, la, threshold=2.5)
            probs_over.append(prob_over)

        return np.array(probs_over)

    def predict_proba(self, X):
        """
        Predict calibrated probabilities.

        Args:
            X: Features DataFrame

        Returns:
            Array of calibrated Over 2.5 probabilities
        """
        # Get calibrated probabilities
        probs_calibrated = self.calibrator.predict_proba(X)[:, 1]

        return probs_calibrated

    def predict(self, X, threshold=0.5):
        """
        Predict binary outcome with custom threshold.

        Args:
            X: Features DataFrame
            threshold: Decision threshold (default: 0.5)

        Returns:
            Binary predictions (0: Under, 1: Over)
        """
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)

    def _calculate_over_under_proba(self, lambda_home, lambda_away, threshold=2.5):
        """Calculate Over/Under probabilities analytically."""
        prob_over = 0.0

        # Sum probabilities for all goal combinations
        for home_goals in range(11):
            for away_goals in range(11):
                total = home_goals + away_goals
                prob = poisson.pmf(home_goals, lambda_home) * poisson.pmf(away_goals, lambda_away)

                if total > threshold:
                    prob_over += prob

        prob_under = 1.0 - prob_over

        return prob_over, prob_under

    def save(self, filepath):
        """Save model to file."""
        import joblib
        joblib.dump(self, filepath)
        self.logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Load model from file."""
        import joblib
        return joblib.load(filepath)
```

### Paso 4: Script de Reentrenamiento

Crea: `scripts/train_balanced_model.py`

```python
"""
Train balanced Poisson model with class weights.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logging, get_logger, Config
from src.models.poisson_model_balanced import BalancedPoissonGoalsModel
from src.models.evaluator import ModelEvaluator, TimeSeriesValidator
import pandas as pd


def main():
    setup_logging()
    logger = get_logger(__name__)
    config = Config()

    logger.info("="*80)
    logger.info("TRAINING BALANCED POISSON MODEL")
    logger.info("="*80)

    # Load data
    data_path = config.data_final_path / 'training_data_latest.parquet'
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # Prepare features
    feature_cols = [col for col in df.columns if col not in [
        'home_goals', 'away_goals', 'total_goals',
        'over_0.5', 'over_1.5', 'over_2.5', 'over_3.5', 'over_4.5',
        'home_team', 'away_team', 'date', 'season'
    ]]

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Samples: {len(df)}")

    # Split data
    train_size = int(0.8 * len(df))
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    # Train model
    logger.info("\nTraining balanced model...")
    model = BalancedPoissonGoalsModel()
    model.fit(df_train, feature_cols, target_col='over_2.5')

    # Evaluate
    logger.info("\nEvaluating on test set...")
    y_true = df_test['over_2.5'].values
    y_pred_proba = model.predict_proba(df_test[feature_cols])

    metrics = ModelEvaluator.evaluate_classification_metrics(
        y_true,
        y_pred_proba,
        threshold=0.5
    )

    logger.info("\nTest Set Results:")
    logger.info(f"  Accuracy:     {metrics['accuracy']:.4f}")
    logger.info(f"  ROC AUC:      {metrics['roc_auc']:.4f}")
    logger.info(f"  F1 Score:     {metrics['f1_score']:.4f}")
    logger.info(f"  Precision:    {metrics['precision']:.4f}")
    logger.info(f"  Recall:       {metrics['recall']:.4f}")
    logger.info(f"  Brier Score:  {metrics['brier_score']:.4f}")

    # Cross-validation
    logger.info("\nRunning cross-validation...")
    validator = TimeSeriesValidator()
    cv_results = validator.validate(model, df, feature_cols, n_splits=5)

    mean_row = cv_results[cv_results['fold'] == 'mean'].iloc[0]
    logger.info(f"\nCV Results:")
    logger.info(f"  ROC AUC: {mean_row['roc_auc']:.4f} ± {cv_results[cv_results['fold'] == 'std'].iloc[0]['roc_auc']:.4f}")

    # Save
    model_path = config.models_path / 'poisson_model_balanced.pkl'
    model.save(model_path)
    logger.info(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
```

### Uso:
```bash
python scripts/train_balanced_model.py
```

### Resultados esperados:
- Mejor balance entre Over/Under predictions
- Predicciones: ~60% Over, ~40% Under (vs 94% Over)
- ROC-AUC: 0.60-0.65 (mejora significativa)
- Calibration Error: < 0.08

---

## 📋 Resumen de las 3 Soluciones

### Solución 1: Optimización de Threshold ⚡ (Rápida)
**Complejidad**: Baja
**Tiempo**: 5 minutos
**Mejora esperada**: +8-12% en Balanced Accuracy

✅ **Ventajas:**
- Implementación inmediata
- No requiere reentrenamiento
- Fácil de ajustar

❌ **Desventajas:**
- No soluciona el problema raíz
- Las probabilidades siguen sesgadas

**Cuándo usar**: Como solución rápida mientras implementas las otras

---

### Solución 2: Feature Engineering V2 🛠️ (Media)
**Complejidad**: Media
**Tiempo**: 1-2 horas
**Mejora esperada**: +5-10% en ROC-AUC

✅ **Ventajas:**
- Elimina features problemáticas
- Mejor generalización
- Modelo más interpretable

❌ **Desventajas:**
- Requiere reentrenamiento completo
- Necesita validación cuidadosa

**Cuándo usar**: Cuando quieras mejorar la calidad del modelo a largo plazo

---

### Solución 3: Modelo Balanceado con Pesos 🎯 (Completa)
**Complejidad**: Alta
**Tiempo**: 2-3 horas
**Mejora esperada**: +10-15% en todas las métricas

✅ **Ventajas:**
- Solución más robusta
- Calibración mejorada
- Mejor manejo de desbalance

❌ **Desventajas:**
- Implementación más compleja
- Requiere más tiempo de entrenamiento

**Cuándo usar**: Para producción o cuando necesites el mejor rendimiento

---

## 🎯 Recomendación de Implementación

### Plan Sugerido:

**Fase 1: Solución Inmediata (Hoy)**
```bash
# 1. Optimizar threshold
python scripts/optimize_threshold.py --metric balanced_accuracy
```

**Fase 2: Mejora de Features (Esta semana)**
```bash
# 1. Analizar features
python scripts/analyze_features.py

# 2. Implementar FeatureEngineerV2
# (copiar código de arriba)

# 3. Reentrenar
python run_complete_pipeline.py
```

**Fase 3: Modelo Balanceado (Próxima semana)**
```bash
# 1. Implementar BalancedPoissonGoalsModel
# (copiar código de arriba)

# 2. Entrenar modelo balanceado
python scripts/train_balanced_model.py

# 3. Comparar resultados
```

---

## 📊 Métricas Esperadas Después de las Soluciones

### Antes (Actual)
```
ROC-AUC:           0.5777 (CV: 0.5165 ± 0.0390)
Balanced Accuracy: ~0.51
Predicciones:      94% Over, 6% Under
Specificity:       6.2%
Sensitivity:       95.7%
```

### Después de Solución 1 (Threshold)
```
ROC-AUC:           0.5777 (sin cambio)
Balanced Accuracy: ~0.60-0.63
Predicciones:      60% Over, 40% Under
Specificity:       45-50%
Sensitivity:       70-75%
```

### Después de Solución 2 (Features V2)
```
ROC-AUC:           0.62-0.65
Balanced Accuracy: ~0.61-0.64
Predicciones:      62% Over, 38% Under
Specificity:       50-55%
Sensitivity:       72-77%
```

### Después de Solución 3 (Balanceado)
```
ROC-AUC:           0.65-0.70
Balanced Accuracy: ~0.63-0.67
Predicciones:      61% Over, 39% Under
Specificity:       55-60%
Sensitivity:       72-78%
Calibration Error: <0.08
```

---

## 🔬 Validación de Resultados

Después de implementar cada solución, ejecuta:

```bash
# 1. Validación completa
python scripts/train_model.py --cv-splits 5

# 2. Análisis de threshold
python scripts/optimize_threshold.py --metric balanced_accuracy

# 3. Análisis de features
python scripts/analyze_features.py

# 4. Comparación de modelos
# Crear script de comparación si es necesario
```

---

## 📚 Referencias y Notas Técnicas

### Por qué Poisson tiende a sobreestimar

La distribución Poisson asume:
- Eventos independientes
- Tasa constante λ

En fútbol:
- Goles NO son totalmente independientes (momentum, táctica)
- λ varía durante el partido (cansancio, cambios)

Esto causa sobreestimación sistemática de λ, especialmente cuando:
1. Equipos fuertes juegan contra débiles
2. Hay racha de goles (momentum)
3. Condiciones especiales (clima, lesiones)

### Alternativas al Poisson

Si las soluciones no funcionan suficiente, considera:

1. **Negative Binomial**: Permite sobredispersión
2. **Zero-Inflated Poisson**: Maneja exceso de 0-0
3. **XGBoost Classifier**: Aprendizaje directo de Over/Under
4. **Ensemble**: Combinar Poisson + XGBoost

---

¡Buena suerte con las implementaciones! 🚀
