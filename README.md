# Premier League Predictor

A professional Data Science project for predicting Premier League match outcomes using machine learning.

## Project Structure

```
premier-league-predictor/
├── data/
│   ├── raw/              # Raw data from sources
│   ├── processed/        # Cleaned and preprocessed data
│   └── final/            # Final datasets ready for modeling
├── src/
│   ├── data/             # Data loading and processing scripts
│   ├── features/         # Feature engineering modules
│   ├── models/           # Model training and evaluation
│   └── utils/            # Utility functions
├── pipelines/            # End-to-end ML pipelines
├── notebooks/            # Jupyter notebooks for exploration
├── tests/                # Unit and integration tests
├── config/               # Configuration files
├── logs/                 # Application logs
├── docs/                 # Documentation
├── dashboards/           # Dashboard applications
├── models/               # Trained model artifacts
├── requirements.txt      # Python dependencies
├── setup.py              # Package installation
└── README.md             # This file
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd premier-league-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

5. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Training Models
```bash
python pipelines/train_pipeline.py
```

### Making Predictions
```bash
python pipelines/predict_pipeline.py
```

### Running Tests
```bash
pytest tests/
```

## Features

- ✅ Data collection from multiple sources (CSV, API)
- ✅ Advanced feature engineering (V1 and V2)
  - **V2 Enhanced**: Bias reduction, ratio features, momentum, volatility
- ✅ Poisson regression model with analytical probability calculation
- ✅ Time series cross-validation (respects temporal order)
- ✅ Model evaluation with calibration metrics
- ✅ Automatic feature validation (VIF, correlations)
- ✅ Threshold optimization for balanced predictions
- ✅ Complete ML pipelines (data + model)

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for detailed quick start guide.

### Train Model with Enhanced Features (V2)

```bash
# Complete pipeline with V2 features
python scripts/train_model_v2.py --validate-features

# Migrate from V1 to V2
python scripts/migrate_to_v2.py

# Compare V1 vs V2 results
python scripts/compare_models.py
```

### Analyze and Fix Model Bias

```bash
# Optimize prediction threshold
python scripts/optimize_threshold.py --metric balanced_accuracy

# Analyze feature importance and correlations
python scripts/analyze_features.py

# View solutions for model bias
cat SOLUTIONS_MODEL_BIAS.md
```

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[SOLUTIONS_MODEL_BIAS.md](SOLUTIONS_MODEL_BIAS.md)** - Fix Over/Under prediction bias
- **[FEATURE_ENGINEERING_COMPARISON.md](FEATURE_ENGINEERING_COMPARISON.md)** - V1 vs V2 features
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions
- **Module READMEs** - See individual src/ module documentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
