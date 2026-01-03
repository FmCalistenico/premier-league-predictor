# Premier League Predictor - Quick Start Guide

Get up and running in 5 minutes!

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

```bash
cp .env.example .env
# Edit .env if you want to use API data
```

## ⚡ Quick Execution

### Option 1: Complete Pipeline (Recommended)

Run everything in one command:

```bash
python run_complete_pipeline.py
```

This will:
1. ✅ Extract data from CSV (last 3 seasons)
2. ✅ Transform and clean data
3. ✅ Engineer 38 features
4. ✅ Train Poisson model
5. ✅ Run 5-fold cross-validation
6. ✅ Save model and generate plots

**Duration**: ~2-3 minutes

### Option 2: Quick Mode (Fast)

For testing or quick iterations:

```bash
python run_complete_pipeline.py --quick
```

- Uses only 2 seasons
- Skips cross-validation
- **Duration**: ~30 seconds

### Option 3: Use Existing Data

If you already have training data:

```bash
python run_complete_pipeline.py --skip-data
```

## 📊 View Results

After running the pipeline:

### 1. Model Performance

Check the console output for:
- Accuracy
- ROC AUC
- F1 Score
- Cross-validation results

### 2. Visualizations

Open these plots:
```
models/plots/
├── calibration_curve.png    # Probability calibration
├── roc_curve.png            # ROC curve with AUC
└── confusion_matrix.png     # Prediction matrix
```

### 3. Detailed Results

```bash
# Cross-validation results
cat models/results/cv_results.csv

# Model metadata
cat models/poisson_model_latest_metadata.json
```

## 🎯 Make Predictions

### Load Trained Model

```python
from src.models import PoissonGoalsModel

# Load model
model = PoissonGoalsModel.load('models/poisson_model_latest.pkl')

# Make predictions
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)
```

## 🔧 Advanced Usage

### Custom Seasons

```bash
python run_complete_pipeline.py --seasons 2324 2425
```

### Skip Cross-Validation

```bash
python run_complete_pipeline.py --skip-cv
```

### Verbose Logging

```bash
python run_complete_pipeline.py --verbose
```

### Use API Data

```bash
# First, add your API key to .env
python run_complete_pipeline.py --source api
```

## 📝 Step-by-Step Execution

If you prefer to run each stage separately:

### Stage 1: Data Pipeline

```bash
python scripts/extract_data.py --sources csv --seasons 2223 2324 2425
```

### Stage 2: Transform Data

```bash
python scripts/transform_data.py \
    --input data/raw/csv/csv_data_*.csv \
    --source csv \
    --output data/processed/transformed.csv
```

### Stage 3: Train Model

```bash
python scripts/train_model.py \
    --data data/final/training_data_latest.parquet \
    --cv-splits 5
```

## 🐍 Python API

### Complete Pipeline

```python
from pipelines import DataPipeline, ModelPipeline

# 1. Data Pipeline
data_pipeline = DataPipeline()
df_final = data_pipeline.run_full_pipeline(
    source='csv',
    csv_seasons=['2223', '2324', '2425']
)

# 2. Model Pipeline
model_pipeline = ModelPipeline()
model, metrics, cv_results = model_pipeline.run_full_training(
    cross_validate=True,
    save_model=True
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"ROC AUC: {metrics['roc_auc']:.3f}")
```

### Quick Functions

```python
# Quick data pipeline (2 seasons only)
from pipelines import run_quick_pipeline
df = run_quick_pipeline(source='csv')

# Quick training (no CV)
from pipelines import run_quick_training
model, metrics = run_quick_training()
```

## 📂 Output Structure

After running the pipeline:

```
premier-league-predictor/
├── data/
│   ├── raw/           # Raw extracted data
│   ├── processed/     # Transformed data
│   └── final/         # Training datasets
│       ├── training_data_latest.parquet
│       └── training_data_latest.csv
│
├── models/
│   ├── poisson_model_latest.pkl        # Trained model
│   ├── poisson_model_latest_metadata.json
│   ├── plots/                          # Evaluation plots
│   │   ├── calibration_curve.png
│   │   ├── roc_curve.png
│   │   └── confusion_matrix.png
│   └── results/
│       └── cv_results.csv             # Cross-validation results
│
└── logs/              # Execution logs
```

## 🎓 Understanding the Model

### Features Used (38 total)

1. **Rolling Features** (24): Team form over 3, 5, 10 matches
   - Goals scored/conceded
   - Goal difference
   - Over 2.5 rate

2. **Match Features** (9): Comparative team strengths
   - Expected total goals
   - Attack/defense strength
   - Form difference

3. **Rest Features** (3): Fixture congestion
   - Days rest for each team
   - Rest advantage

4. **H2H Features** (2): Historical matchups
   - Average goals in H2H
   - Over rate in H2H

### Model Type

**Dual Poisson Regression**:
- Predicts home and away goals independently
- Calculates over/under probabilities analytically
- Well-calibrated probability estimates

### Performance Expectations

Typical results on Premier League data:
- **Accuracy**: 60-65%
- **ROC AUC**: 0.65-0.70
- **Calibration Error**: < 0.10

Note: Over/under prediction is inherently difficult. 60%+ accuracy is good!

## ❓ Troubleshooting

### "No module named 'src'"

```bash
# Make sure you're in the project root directory
cd premier-league-predictor

# Or install the package
pip install -e .
```

### "training_data_latest.parquet not found"

```bash
# Run data pipeline first
python run_complete_pipeline.py
```

### API Key Issues

```bash
# For CSV data, API key is not needed
python run_complete_pipeline.py --source csv
```

### Memory Errors

```bash
# Use quick mode with fewer seasons
python run_complete_pipeline.py --quick
```

## 📚 Next Steps

1. **Explore Examples**: Check `examples/` directory for detailed usage
2. **Read Documentation**: See module READMEs in `src/`
3. **Customize**: Modify `config/config.yaml` for your needs
4. **Add Models**: Extend with XGBoost, Random Forest, etc.
5. **Build API**: Create FastAPI endpoint for predictions
6. **Dashboard**: Build Streamlit dashboard

## 🆘 Get Help

- **Documentation**: Check module READMEs
- **Examples**: Run scripts in `examples/`
- **Logs**: Check `logs/` for detailed execution logs
- **Issues**: Report bugs on GitHub

## ✨ Pro Tips

1. **Always use time-based splits** for validation (not random!)
2. **Check calibration plots** to ensure probabilities are meaningful
3. **Monitor data leakage** - all rolling features use `.shift(1)`
4. **Save models with metadata** for reproducibility
5. **Use cross-validation** to get reliable performance estimates

---

Happy Predicting! ⚽📈
