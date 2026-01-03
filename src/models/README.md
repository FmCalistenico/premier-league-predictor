# Models Module

This module implements machine learning models for Premier League match prediction.

## Poisson Goals Model

The `PoissonGoalsModel` uses dual independent Poisson regression to predict match outcomes.

### Mathematical Foundation

**Model Structure:**
- Two independent Generalized Linear Models (GLM) with Poisson family
- Home model: `log(λ_home) = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ`
- Away model: `log(λ_away) = γ₀ + γ₁X₁ + γ₂X₂ + ... + γₙXₙ`

Where:
- `λ_home`: Expected home team goals (Poisson parameter)
- `λ_away`: Expected away team goals (Poisson parameter)
- `X₁, X₂, ..., Xₙ`: Feature vectors

**Probability Calculation:**

Since home and away goals are independent:
```
P(home=i, away=j) = P(home=i) × P(away=j)
                  = Poisson(i; λ_home) × Poisson(j; λ_away)
```

**Over/Under Probability:**
```
P(Total > 2.5) = Σ P(home=i, away=j) for all i+j > 2.5
               = Σ Poisson(i; λ_home) × Poisson(j; λ_away)
```

This is calculated analytically (not via simulation) for exact probabilities.

## Usage

### Basic Training

```python
from src.models import PoissonGoalsModel

# Initialize model
model = PoissonGoalsModel(threshold=2.5)

# Train on data
model.fit(
    df_train,
    feature_cols=['expected_total_goals', 'form_difference', ...]
)

# Model summary
print(model)
# PoissonGoalsModel(threshold=2.5, n_features=38, home_AIC=2341.23, away_AIC=2398.45)
```

### Making Predictions

```python
# Predict expected goals
lambda_home, lambda_away = model.predict_expected_goals(df_test)

# Predict over/under probabilities
prob_over, prob_under, expected_total = model.predict_over_under(df_test)

# Binary predictions (0=under, 1=over)
predictions = model.predict(df_test)

# Probability predictions (sklearn-compatible)
proba = model.predict_proba(df_test)
# Returns: [[prob_under, prob_over], ...]
```

### Analytical Calculations

Calculate probabilities for specific λ values:

```python
# Example: Expected 1.5 home goals, 1.2 away goals
prob_over, prob_under = model.predict_over_under_analytical(
    lambda_home=1.5,
    lambda_away=1.2,
    threshold=2.5
)

print(f"P(Over 2.5): {prob_over:.3f}")  # 0.456
print(f"P(Under 2.5): {prob_under:.3f}")  # 0.544
```

### Model Inspection

```python
# Get coefficients with significance
coef_df = model.get_coefficients()
print(coef_df[['feature', 'home_coef', 'home_p_value', 'away_coef', 'away_p_value']])

# Get model summary statistics
summary = model.get_model_summary()
print(f"Home AIC: {summary['home_model']['aic']:.2f}")
print(f"Away AIC: {summary['away_model']['aic']:.2f}")
```

### Save and Load

```python
# Save trained model
model.save('models/poisson_model.pkl')

# Load later
loaded_model = PoissonGoalsModel.load('models/poisson_model.pkl')

# Use loaded model
predictions = loaded_model.predict(df_new)
```

## Complete Example

```python
from src.data import FootballDataCSVClient, DataTransformer
from src.features import FeatureEngineer
from src.models import PoissonGoalsModel
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Prepare data
csv_client = FootballDataCSVClient()
df_raw = csv_client.get_multiple_seasons(['2223', '2324', '2425'])

# 2. Transform
transformer = DataTransformer()
df_transformed, _ = transformer.transform(df_raw, source='csv')

# 3. Engineer features
engineer = FeatureEngineer()
df_features = engineer.engineer_features(df_transformed)
feature_cols = engineer.get_feature_names()

# 4. Split data (time-based)
train_df = df_features[df_features['date'] < '2024-08-01']
test_df = df_features[df_features['date'] >= '2024-08-01']

# 5. Train model
model = PoissonGoalsModel(threshold=2.5)
model.fit(train_df, feature_cols)

# 6. Evaluate
y_true = test_df['over_2.5'].values
y_pred = model.predict(test_df)
y_proba = model.predict_proba(test_df)[:, 1]

print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
print(f"ROC AUC: {roc_auc_score(y_true, y_proba):.3f}")

# 7. Save model
model.save('models/poisson_over25.pkl')
```

## Model Components

### Attributes

- `home_model`: GLM for home goals
- `away_model`: GLM for away goals
- `threshold`: Goals threshold for over/under (default: 2.5)
- `feature_cols`: List of feature names used

### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `fit(df, feature_cols)` | Train both models | Self |
| `predict_expected_goals(df)` | Get λ values | (λ_home, λ_away) |
| `predict_over_under_analytical(λh, λa, t)` | Calculate P(Over) | (prob_over, prob_under) |
| `predict_over_under(df, threshold)` | Predict for DataFrame | (prob_over, prob_under, expected) |
| `predict_proba(df)` | Sklearn-compatible probabilities | Array (n, 2) |
| `predict(df)` | Binary predictions | Array (n,) |
| `get_coefficients()` | Get model coefficients | DataFrame |
| `get_model_summary()` | Get model statistics | Dict |
| `save(filepath)` | Save model | None |
| `load(filepath)` | Load model (classmethod) | PoissonGoalsModel |

## Model Statistics

After training, access model statistics:

```python
summary = model.get_model_summary()

# Available statistics:
# - AIC (Akaike Information Criterion)
# - BIC (Bayesian Information Criterion)
# - Deviance
# - Pearson Chi-squared
# - Convergence status
# - Number of iterations
```

**Lower AIC/BIC indicates better model fit.**

## Advantages of Poisson Model

1. **Interpretability**: Coefficients have clear meaning
2. **Probabilistic**: Provides proper probabilities, not just classifications
3. **Analytical**: Exact calculations (no simulation needed)
4. **Statistically sound**: Based on well-established GLM theory
5. **Efficient**: Fast training and prediction

## Limitations

1. **Independence assumption**: Assumes home/away goals are independent
2. **Poisson assumption**: Assumes goals follow Poisson distribution
3. **Linear log-link**: May not capture complex non-linear relationships
4. **No correlation modeling**: Doesn't model correlation between teams

## Feature Importance

The most important features for Poisson models are typically:

**High importance:**
- `expected_total_goals`: Direct indicator of scoring
- `combined_over_rate`: Historical over/under rate
- `goals_scored_L5`, `goals_scored_L10`: Recent scoring form
- `attack_strength_diff`: Relative attacking ability

**Medium importance:**
- `form_difference`: Overall team form
- `h2h_avg_goals`: Head-to-head history
- `defense_strength_diff`: Defensive capabilities

**Lower importance:**
- Very short-term features (L3)
- Individual team strengths (less predictive than differences)

## Performance Expectations

Typical performance on Premier League data:

- **Accuracy**: 55-65% (over/under 2.5)
- **ROC AUC**: 0.60-0.70
- **Calibration**: Generally well-calibrated probabilities

**Note**: Over/under prediction is inherently difficult. Even 60% accuracy is considered good.

## Comparison with Other Models

| Model | Pros | Cons |
|-------|------|------|
| **Poisson** | Interpretable, probabilistic, fast | Independence assumption |
| Random Forest | No assumptions, captures non-linearity | Black box, slower |
| XGBoost | High accuracy, handles interactions | Black box, requires tuning |
| Neural Network | Very flexible | Needs more data, hard to interpret |

## Best Practices

1. **Time-based splits**: Always use temporal validation (no random splits)
2. **Feature engineering**: Quality features matter more than model complexity
3. **Calibration**: Check probability calibration on validation set
4. **Threshold tuning**: Experiment with different thresholds (2.5, 3.5, etc.)
5. **Ensemble**: Consider combining with other models

## Examples

See [examples/poisson_model_example.py](../../examples/poisson_model_example.py) for:
- Basic training
- Making predictions
- Analytical calculations
- Model evaluation
- Getting model summary
- Saving and loading

## Dependencies

- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `scipy`: Poisson probability calculations
- `statsmodels`: GLM implementation
- `scikit-learn`: Evaluation metrics (optional)

## Testing

Run examples:
```bash
python examples/poisson_model_example.py
```

## References

1. Maher, M.J. (1982). "Modelling association football scores". Statistica Neerlandica, 36(3), 109-118.
2. Dixon, M.J., & Coles, S.G. (1997). "Modelling Association Football Scores and Inefficiencies in the Football Betting Market". Journal of the Royal Statistical Society, 46(2), 265-280.
3. Karlis, D., & Ntzoufras, I. (2003). "Analysis of sports data by using bivariate Poisson models". Journal of the Royal Statistical Society, 52(3), 381-393.
