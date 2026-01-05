"""
Simple Prediction Example (Without Feature Engineering)
========================================================

This example shows how to make predictions using pre-calculated features
from the existing training dataset.

Usage:
    python example_simple_prediction.py
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("SIMPLE PREDICTION EXAMPLE")
print("=" * 80)

# =========================================================================
# STEP 1: Load Pre-calculated Data
# =========================================================================
print("\n[STEP 1] Loading pre-calculated features from training data...")

data_path = Path('data/final/training_data_v2.parquet')

if not data_path.exists():
    print(f"ERROR: {data_path} not found")
    print("Please ensure you have the V2 training data.")
    exit(1)

df = pd.read_parquet(data_path)
print(f"[OK] Loaded {len(df)} matches with features")

# =========================================================================
# STEP 2: Select Recent Matches as "Test" Fixtures
# =========================================================================
print("\n[STEP 2] Selecting recent matches to simulate predictions...")

# Use last 10 matches as our "upcoming fixtures"
test_fixtures = df.tail(10).copy()

print(f"[OK] Selected {len(test_fixtures)} recent matches for testing")
print("\nTest Fixtures:")
for _, row in test_fixtures.head(5).iterrows():
    date = row['date'] if 'date' in row else 'N/A'
    home = row.get('home_team', row.get('home_team_name', 'Unknown'))
    away = row.get('away_team', row.get('away_team_name', 'Unknown'))
    print(f"  {date} | {home:20s} vs {away:20s}")

# =========================================================================
# STEP 3: Load Model
# =========================================================================
print("\n[STEP 3] Loading trained model...")

model_path = Path('models/production_model_balanced_20260104.pkl')

if not model_path.exists():
    print(f"ERROR: {model_path} not found")
    print("Using default production model...")
    model_path = Path('models/production/best_model.pkl')

    if not model_path.exists():
        print("Please train a model first using:")
        print("  python scripts/retrain_improved_pipeline.py")
        exit(1)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f"[OK] Model loaded: {type(model).__name__}")
print(f"  Balanced threshold: {getattr(model, 'threshold', 0.5)}")

# =========================================================================
# STEP 4: Prepare Features
# =========================================================================
print("\n[STEP 4] Preparing features for prediction...")

# Get feature columns
feature_cols = [col for col in test_fixtures.columns
                if col not in ['fixture_id', 'date', 'season', 'home_team', 'away_team',
                               'home_team_name', 'away_team_name',
                               'gameweek', 'home_goals', 'away_goals', 'total_goals',
                               'over_0.5', 'over_1.5', 'over_2.5', 'over_3.5', 'over_4.5']]

X = test_fixtures[feature_cols].copy()

# Handle any NaN values
X = X.fillna(X.median())

print(f"[OK] Using {len(feature_cols)} features")

# =========================================================================
# STEP 5: Make Predictions
# =========================================================================
print("\n[STEP 5] Making predictions...")

# Get probabilities
try:
    probabilities = model.predict_proba(X)
    prob_under = probabilities[:, 0]
    prob_over = probabilities[:, 1]
except Exception as e:
    print(f"Error getting probabilities: {e}")
    print("Using predict instead...")
    predictions = model.predict(X)
    prob_over = np.full(len(X), 0.5)
    prob_under = np.full(len(X), 0.5)
else:
    # Apply balanced threshold to probabilities
    # Model default uses 0.5 (prob_over > prob_under)
    # We use optimized threshold 0.5778 for better balance
    probability_threshold = getattr(model, 'threshold', 0.5778)
    if probability_threshold == 2.5:  # It's the goals threshold, use our optimized one
        probability_threshold = 0.5778

    predictions = (prob_over >= probability_threshold).astype(int)
    print(f"  Applied probability threshold: {probability_threshold:.4f}")

print(f"[OK] Predictions completed for {len(predictions)} fixtures")

# =========================================================================
# STEP 6: Calculate Simple Confidence
# =========================================================================
print("\n[STEP 6] Calculating confidence scores...")

confidence_scores = []

for prob in prob_over:
    # Simple confidence based on probability extremeness
    extremeness = abs(prob - 0.5) * 2  # 0-1 scale
    confidence = 50 + (extremeness * 50)  # 50-100% scale
    confidence_scores.append(confidence)

confidence_scores = np.array(confidence_scores)

print(f"[OK] Confidence scores calculated")
print(f"  Average confidence: {confidence_scores.mean():.1f}%")

# =========================================================================
# STEP 7: Create Results DataFrame
# =========================================================================
print("\n[STEP 7] Organizing results...")

results = pd.DataFrame({
    'date': test_fixtures['date'].values if 'date' in test_fixtures else ['N/A'] * len(test_fixtures),
    'home_team': test_fixtures.get('home_team', test_fixtures.get('home_team_name', 'Unknown')).values,
    'away_team': test_fixtures.get('away_team', test_fixtures.get('away_team_name', 'Unknown')).values,
    'prob_over': prob_over,
    'prob_under': prob_under,
    'prediction': predictions,
    'confidence': confidence_scores
})

# Add prediction labels
results['prediction_label'] = results['prediction'].map({0: 'Under 2.5', 1: 'Over 2.5'})

# Add recommendations
def get_recommendation(row):
    if row['confidence'] >= 75:
        return f"BET: {row['prediction_label']} (High Confidence)"
    elif row['confidence'] >= 65:
        return f"BET: {row['prediction_label']} (Medium Confidence)"
    else:
        return "SKIP: Low Confidence"

results['recommendation'] = results.apply(get_recommendation, axis=1)

# Add actual results if available
if 'over_2.5' in test_fixtures.columns:
    results['actual_over_2.5'] = test_fixtures['over_2.5'].values
    results['correct'] = (results['prediction'] == results['actual_over_2.5']).astype(int)

print(f"[OK] Results organized")

# =========================================================================
# STEP 8: Display Results
# =========================================================================
print("\n" + "=" * 100)
print("PREDICTION RESULTS")
print("=" * 100)

for idx, row in results.iterrows():
    print(f"\n{row['home_team']} vs {row['away_team']}")
    print(f"  Date: {row['date']}")
    print(f"  Prediction: {row['prediction_label']}")
    print(f"  Probability: Over={row['prob_over']:.1%}, Under={row['prob_under']:.1%}")
    print(f"  Confidence: {row['confidence']:.1f}%")
    print(f"  >> {row['recommendation']}")

    if 'actual_over_2.5' in results.columns:
        actual = 'Over 2.5' if row['actual_over_2.5'] == 1 else 'Under 2.5'
        status = '[CORRECT]' if row['correct'] == 1 else '[WRONG]'
        print(f"  Actual Result: {actual} ({status})")

# =========================================================================
# STEP 9: Save Results
# =========================================================================
print("\n[STEP 9] Saving predictions...")

output_path = Path('data/predictions/simple_predictions_example.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(output_path, index=False)
print(f"[OK] Saved predictions: {output_path}")

# =========================================================================
# STEP 10: Summary Statistics
# =========================================================================
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

bet_count = results['recommendation'].str.startswith('BET').sum()
skip_count = len(results) - bet_count

print(f"\nRecommendations:")
print(f"  BET:  {bet_count} fixtures")
print(f"  SKIP: {skip_count} fixtures")

print(f"\nPrediction Distribution:")
print(f"  Over 2.5:  {(results['prediction'] == 1).sum()} ({100 * (results['prediction'] == 1).mean():.1f}%)")
print(f"  Under 2.5: {(results['prediction'] == 0).sum()} ({100 * (results['prediction'] == 0).mean():.1f}%)")

print(f"\nConfidence:")
print(f"  Average: {results['confidence'].mean():.1f}%")
print(f"  Min: {results['confidence'].min():.1f}%")
print(f"  Max: {results['confidence'].max():.1f}%")

if 'correct' in results.columns:
    # Calculate accuracy only for BET recommendations
    bet_results = results[results['recommendation'].str.startswith('BET')]

    if len(bet_results) > 0:
        accuracy = bet_results['correct'].mean()
        print(f"\nAccuracy (BET recommendations only):")
        print(f"  {bet_results['correct'].sum()}/{len(bet_results)} correct ({accuracy:.1%})")

print("\n" + "=" * 100)
print("[COMPLETE] EXAMPLE FINISHED")
print("=" * 100)
print(f"\nResults saved to: {output_path}")
print("\nThis example shows the prediction workflow using existing features.")
print("For real predictions on future fixtures, use:")
print("  python scripts/get_upcoming_fixtures.py --manual")
print("  python scripts/build_prediction_dataset.py --input ...")
print("  python scripts/predict_fixtures.py --input ...")
