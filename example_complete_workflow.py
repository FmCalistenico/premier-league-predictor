"""
Example: Complete Prediction Workflow
======================================

Demonstrates the full prediction pipeline from scratch.

This example shows how to:
1. Create upcoming fixtures (manual input)
2. Build prediction dataset with V2 features
3. Make predictions with confidence scoring
4. Analyze and act on recommendations

Usage:
    python example_complete_workflow.py
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

# Import our custom modules
from scripts.build_prediction_dataset import PredictionDatasetBuilder
from scripts.predict_fixtures import FixturePredictor

def main():
    print("=" * 80)
    print("COMPLETE PREDICTION WORKFLOW EXAMPLE")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Create Upcoming Fixtures
    # =========================================================================
    print("\n[STEP 1] Creating upcoming fixtures...")

    # In production, you would get these from API or user input
    # For this example, we'll create fixtures from a recent completed gameweek
    # to simulate "predicting" matches we can validate
    upcoming_fixtures = pd.DataFrame([
        {
            'date': '2024-12-21',  # Using recent GW for testing
            'home_team': 'Arsenal',
            'away_team': 'Crystal Palace',
            'gameweek': 17
        },
        {
            'date': '2024-12-21',
            'home_team': 'Newcastle',
            'away_team': 'Aston Villa',
            'gameweek': 17
        },
        {
            'date': '2024-12-22',
            'home_team': 'Tottenham',
            'away_team': 'Liverpool',
            'gameweek': 17
        },
        {
            'date': '2024-12-22',
            'home_team': 'Man City',
            'away_team': 'Man Utd',
            'gameweek': 17
        }
    ])

    print(f"✓ Created {len(upcoming_fixtures)} fixtures for prediction")
    print("\nFixtures:")
    for _, row in upcoming_fixtures.iterrows():
        print(f"  {row['date']} | {row['home_team']:15s} vs {row['away_team']:15s}")

    # Save fixtures (optional)
    fixtures_path = 'data/raw/example_fixtures_GW23.csv'
    Path(fixtures_path).parent.mkdir(parents=True, exist_ok=True)
    upcoming_fixtures.to_csv(fixtures_path, index=False)
    print(f"\n✓ Saved fixtures: {fixtures_path}")

    # =========================================================================
    # STEP 2: Build Prediction Dataset
    # =========================================================================
    print("\n[STEP 2] Building prediction dataset with V2 features...")

    builder = PredictionDatasetBuilder(
        historical_data_path='data/final/training_data_v2.parquet'
    )

    # Build dataset (this calculates all 78 V2 features)
    prediction_df = builder.build_upcoming_fixtures(upcoming_fixtures)

    print(f"✓ Dataset built: {len(prediction_df)} rows, {len(prediction_df.columns)} columns")
    print(f"  Features: {len([c for c in prediction_df.columns if c not in ['fixture_id', 'date', 'home_team', 'away_team', 'gameweek']])}")

    # Save prediction dataset (optional)
    dataset_path = 'data/predictions/example_prediction_data_GW23.parquet'
    builder.save_prediction_dataset(prediction_df, dataset_path)

    # =========================================================================
    # STEP 3: Make Predictions
    # =========================================================================
    print("\n[STEP 3] Making predictions with confidence scoring...")

    predictor = FixturePredictor(
        model_path='models/results/retrain_checkpoint.pkl',
        optimized_threshold=0.62  # Use optimized threshold instead of default 0.75
    )

    # Make predictions
    results = predictor.predict(prediction_df)

    # Calculate confidence scores
    results = predictor.calculate_confidence(results, prediction_df)

    # Generate recommendations (min confidence = 70%)
    results = predictor.generate_recommendations(results, min_confidence=70)

    print(f"✓ Predictions completed for {len(results)} fixtures")

    # =========================================================================
    # STEP 4: Analyze Results
    # =========================================================================
    print("\n[STEP 4] Analyzing predictions...")

    # Count predictions by recommendation
    bet_count = (results['recommendation'].str.startswith('BET')).sum()
    skip_count = len(results) - bet_count

    print(f"\nRecommendation Summary:")
    print(f"  BET:  {bet_count} fixtures (confidence ≥ 70%)")
    print(f"  SKIP: {skip_count} fixtures (confidence < 70%)")

    # Show detailed predictions
    print(f"\n{'=' * 100}")
    print(f"DETAILED PREDICTIONS")
    print(f"{'=' * 100}")

    for _, row in results.iterrows():
        print(f"\n{row['home_team']} vs {row['away_team']}")
        print(f"  Date: {row['date']}")
        print(f"  Prediction: {row['prediction_label']}")
        print(f"  Probability: Over={row['prob_over']:.1%}, Under={row['prob_under']:.1%}")
        print(f"  Expected Goals: {row['expected_total_goals']:.2f} (Home: {row['expected_home_goals']:.2f}, Away: {row['expected_away_goals']:.2f})")
        print(f"  Confidence: {row['confidence']:.1f}% ({row['confidence_tier']})")
        print(f"  → {row['recommendation']}")

    # =========================================================================
    # STEP 5: Save Predictions
    # =========================================================================
    print(f"\n[STEP 5] Saving predictions...")

    predictions_path = 'data/predictions/example_predictions_GW23.csv'
    predictor.save_predictions(results, predictions_path)

    # =========================================================================
    # STEP 6: Actionable Insights
    # =========================================================================
    print(f"\n{'=' * 100}")
    print(f"ACTIONABLE INSIGHTS")
    print(f"{'=' * 100}")

    # High confidence bets
    high_conf_bets = results[
        (results['recommendation'].str.startswith('BET')) &
        (results['confidence'] >= 75)
    ]

    if not high_conf_bets.empty:
        print(f"\n🎯 HIGH CONFIDENCE BETS (≥75%):")
        for _, row in high_conf_bets.iterrows():
            print(f"  ✓ {row['home_team']:15s} vs {row['away_team']:15s} | "
                  f"{row['prediction_label']:12s} ({row['prob_over']:.1%}) | "
                  f"Confidence: {row['confidence']:.0f}%")
    else:
        print(f"\n⚠️  No high confidence bets found for this gameweek")

    # Medium confidence bets
    medium_conf_bets = results[
        (results['recommendation'].str.startswith('BET')) &
        (results['confidence'] >= 65) &
        (results['confidence'] < 75)
    ]

    if not medium_conf_bets.empty:
        print(f"\n⚖️  MEDIUM CONFIDENCE BETS (65-75%):")
        for _, row in medium_conf_bets.iterrows():
            print(f"  • {row['home_team']:15s} vs {row['away_team']:15s} | "
                  f"{row['prediction_label']:12s} ({row['prob_over']:.1%}) | "
                  f"Confidence: {row['confidence']:.0f}%")

    # Skipped predictions
    skipped = results[results['recommendation'].str.startswith('SKIP')]

    if not skipped.empty:
        print(f"\n❌ SKIPPED (Low Confidence <65%):")
        for _, row in skipped.iterrows():
            print(f"  - {row['home_team']:15s} vs {row['away_team']:15s} | "
                  f"{row['prediction_label']:12s} ({row['prob_over']:.1%}) | "
                  f"Confidence: {row['confidence']:.0f}%")

    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print(f"\n{'=' * 100}")
    print(f"SUMMARY STATISTICS")
    print(f"{'=' * 100}")
    print(f"Total Fixtures: {len(results)}")
    print(f"Average Confidence: {results['confidence'].mean():.1f}%")
    print(f"Predicted Over: {(results['prediction'] == 1).sum()} ({100 * (results['prediction'] == 1).mean():.1f}%)")
    print(f"Predicted Under: {(results['prediction'] == 0).sum()} ({100 * (results['prediction'] == 0).mean():.1f}%)")
    print(f"Avg Prob(Over): {results['prob_over'].mean():.1%}")
    print(f"Avg Expected Goals: {results['expected_total_goals'].mean():.2f}")

    print(f"\n{'=' * 100}")
    print(f"✓ WORKFLOW COMPLETE")
    print(f"{'=' * 100}")
    print(f"\nPredictions saved to: {predictions_path}")
    print(f"\nNext steps:")
    print(f"  1. Review high confidence predictions")
    print(f"  2. Consider placing bets (start with small stakes)")
    print(f"  3. Track results after the gameweek completes")
    print(f"  4. Calculate actual ROI and refine confidence thresholds")

    return results


if __name__ == "__main__":
    main()
