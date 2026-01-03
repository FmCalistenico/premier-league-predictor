# Features Module

This module handles feature engineering for Premier League match prediction with **strict data leakage prevention**.

## Critical: Data Leakage Prevention

⚠️ **ALL rolling features use `.shift(1)` before `.rolling()`** to ensure we only use information available BEFORE the match we're predicting.

```python
# CORRECT - No data leakage
team_data['goals_scored_L5'] = (
    grouped['goals_scored']
    .shift(1)  # Skip current match
    .rolling(window=5, min_periods=1)
    .mean()
)

# WRONG - Data leakage!
team_data['goals_scored_L5'] = (
    grouped['goals_scored']
    .rolling(window=5, min_periods=1)  # Includes current match
    .mean()
)
```

## Feature Categories

### 1. Rolling Features
Created for each team (home/away) and window size:

**Windows**: 3, 5, 10 matches

**Features per window**:
- `goals_scored_L{window}`: Average goals scored in last N matches
- `goals_conceded_L{window}`: Average goals conceded in last N matches
- `over_rate_L{window}`: Rate of over 2.5 goals in last N matches
- `goal_diff_L{window}`: Average goal difference in last N matches

**Total**: 4 features × 3 windows × 2 teams = **24 rolling features**

### 2. Match-Level Features
Comparative features between teams:

- `expected_total_goals`: Combined average goals scored
- `form_difference`: Difference in goal difference form
- `home_attack_strength`: Normalized attack strength vs league average
- `away_attack_strength`: Normalized attack strength vs league average
- `attack_strength_diff`: Difference in attack strength
- `home_defense_strength`: Normalized defense strength vs league average
- `away_defense_strength`: Normalized defense strength vs league average
- `defense_strength_diff`: Difference in defense strength
- `combined_over_rate`: Combined rate of over 2.5 goals

**Total**: **9 match features**

### 3. Rest Days Features
Fixture congestion indicators:

- `home_days_rest`: Days since last match for home team
- `away_days_rest`: Days since last match for away team
- `rest_advantage`: Difference in rest days (home - away)

**Total**: **3 rest features**

### 4. Head-to-Head Features
Historical matchup statistics:

- `h2h_avg_goals`: Average total goals in last N H2H matches (default: 5)
- `h2h_over_rate`: Rate of over 2.5 goals in last N H2H matches

**Total**: **2 H2H features**

## Total Features Created

**38 features** = 24 rolling + 9 match + 3 rest + 2 H2H

## Usage

### Basic Usage

```python
from src.features import FeatureEngineer

# Initialize
engineer = FeatureEngineer()

# Create all features
df_features = engineer.engineer_features(
    df_transformed,
    rolling_windows=[3, 5, 10],
    h2h_window=5,
    min_matches_required=5
)

# Get feature names
feature_names = engineer.get_feature_names()
print(f"Created {len(feature_names)} features")
```

### Step-by-Step Usage

```python
from src.features import FeatureEngineer

engineer = FeatureEngineer()

# Step 1: Rolling features
df = engineer.create_rolling_features(df, windows=[3, 5, 10])

# Step 2: Match features
df = engineer.create_match_features(df)

# Step 3: Rest days features
df = engineer.create_rest_days_features(df)

# Step 4: Head-to-head features
df = engineer.create_head_to_head_features(df, h2h_window=5)

# Get created features
features = engineer.get_feature_names()
```

### Complete Pipeline with Data Preparation

```python
from src.data import FootballDataCSVClient, DataTransformer
from src.features import FeatureEngineer

# 1. Extract data
csv_client = FootballDataCSVClient()
df_raw = csv_client.get_multiple_seasons(['2223', '2324', '2425'])

# 2. Transform data
transformer = DataTransformer()
df_transformed, metadata = transformer.transform(df_raw, source='csv')

# 3. Engineer features
engineer = FeatureEngineer()
df_features = engineer.engineer_features(df_transformed)

# 4. Ready for modeling
print(f"Final dataset: {len(df_features)} matches, {len(df_features.columns)} columns")
```

## Feature Engineering Pipeline

The `engineer_features()` method runs the complete pipeline:

1. **Create rolling features** - Team form and statistics
2. **Create match features** - Comparative team strengths
3. **Create rest days features** - Fixture congestion
4. **Create head-to-head features** - Historical matchup data
5. **Filter insufficient history** - Remove matches without enough data

## Data Leakage Prevention Verification

Example verification that no leakage exists:

```python
from src.features import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_rolling_features(df, windows=[3])

# Check a specific match
sample = df_features.iloc[10]
print(f"Match: {sample['home_team_name']} vs {sample['away_team_name']}")
print(f"Date: {sample['date']}")
print(f"Home goals scored L3: {sample['home_goals_scored_L3']}")

# This feature was calculated using only matches BEFORE this date
# The current match goals are NOT included
```

## Feature Importance Considerations

**Expected high importance features**:
- `expected_total_goals`: Strong indicator of match outcome
- `combined_over_rate`: Direct predictor of over/under
- `form_difference`: Team form is critical
- `goals_scored_L5`, `goals_scored_L10`: Recent scoring form

**Expected moderate importance**:
- `h2h_avg_goals`: Historical matchup patterns
- `attack_strength_diff`: Relative team strengths
- `rest_advantage`: Fixture congestion effects

**Expected lower importance**:
- Individual defense/attack strengths (less predictive than combined)
- Very short windows (L3) may be noisy

## Configuration

Default parameters:
```python
rolling_windows = [3, 5, 10]  # Match windows for rolling stats
h2h_window = 5                # Previous H2H matches to consider
min_matches_required = 5       # Minimum matches needed per team
```

Adjust based on your needs:
```python
# More granular windows
df_features = engineer.engineer_features(
    df,
    rolling_windows=[3, 5, 7, 10, 15],
    h2h_window=10,
    min_matches_required=10
)
```

## Performance Considerations

- **Memory**: Rolling features require groupby operations on large datasets
- **Speed**: H2H features are slowest (quadratic in matches per matchup)
- **Optimization**: Pre-sort data by date for faster processing

**Approximate processing time** (1000 matches):
- Rolling features: ~2-3 seconds
- Match features: <1 second
- Rest days: ~1 second
- H2H features: ~5-10 seconds
- **Total**: ~10-15 seconds

## Data Requirements

**Minimum required columns**:
- `date`: Match date
- `home_team_name`: Home team name
- `away_team_name`: Away team name
- `home_goals`: Home team goals
- `away_goals`: Away team goals
- `total_goals`: Total goals in match
- `over_2.5`: Binary target (optional but recommended)

**Recommended minimum data**:
- At least 1 full season (380 matches for Premier League)
- At least 10 matches per team for reliable rolling features

## Examples

See [examples/feature_engineering_example.py](../../examples/feature_engineering_example.py) for:
- Individual feature creation examples
- Complete pipeline example
- Data leakage verification example
- Feature statistics and analysis

## Logging

All operations are logged with detailed information:
```
INFO: Creating rolling features with windows: [3, 5, 10]
INFO: Created 24 rolling features
INFO: Creating match-level features
INFO: Created 9 match-level features
INFO: Creating rest days features
INFO: Created 3 rest days features
INFO: Creating head-to-head features (window=5)
INFO: Created 2 head-to-head features
INFO: Final dataset: 950 matches with 38 features
```

## Testing

Verify feature engineering:
```bash
python examples/feature_engineering_example.py
```

## Notes

1. **Always check for data leakage** - Verify rolling features use `.shift(1)`
2. **Handle missing values** - Early matches may not have full rolling windows
3. **Normalize features** - Consider scaling before modeling
4. **Feature selection** - Not all 38 features may be needed
5. **Cross-validation** - Use time-based splits to respect temporal nature
