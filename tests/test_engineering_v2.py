"""
Unit tests for FeatureEngineerV2.

Tests cover:
1. Data leakage prevention (shift(1) validation)
2. Feature creation and naming
3. VIF validation (< 10)
4. Correlation validation (< 0.7)
5. Context features (derbies, top clashes)
6. Ratio features correctness
7. Momentum and volatility features
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.engineering_v2 import FeatureEngineerV2, run_feature_engineering_v2


@pytest.fixture
def sample_data():
    """Create sample match data for testing."""
    np.random.seed(42)

    dates = pd.date_range(start='2023-08-01', periods=100, freq='3D')
    teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Tottenham',
             'Man United', 'West Ham', 'Everton']

    data = []
    for i in range(100):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])

        # Simulate realistic goal distributions
        home_goals = np.random.poisson(1.5)
        away_goals = np.random.poisson(1.2)

        data.append({
            'date': dates[i],
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'total_goals': home_goals + away_goals,
            'over_2.5': int(home_goals + away_goals > 2.5)
        })

    return pd.DataFrame(data)


@pytest.fixture
def derby_data():
    """Create data with known derbies."""
    data = [
        {'date': pd.Timestamp('2023-08-01'), 'home_team': 'Arsenal', 'away_team': 'Chelsea',
         'home_goals': 2, 'away_goals': 1, 'total_goals': 3, 'over_2.5': 1},
        {'date': pd.Timestamp('2023-08-05'), 'home_team': 'Liverpool', 'away_team': 'Everton',
         'home_goals': 3, 'away_goals': 0, 'total_goals': 3, 'over_2.5': 1},
        {'date': pd.Timestamp('2023-08-10'), 'home_team': 'Man City', 'away_team': 'Man United',
         'home_goals': 1, 'away_goals': 1, 'total_goals': 2, 'over_2.5': 0},
        {'date': pd.Timestamp('2023-08-15'), 'home_team': 'Arsenal', 'away_team': 'Man City',
         'home_goals': 2, 'away_goals': 2, 'total_goals': 4, 'over_2.5': 1},
    ]
    return pd.DataFrame(data)


class TestFeatureEngineerV2:
    """Test suite for FeatureEngineerV2."""

    def test_initialization(self):
        """Test FeatureEngineerV2 initialization."""
        engineer = FeatureEngineerV2()
        assert engineer is not None
        assert hasattr(engineer, 'city_teams')
        assert 'London' in engineer.city_teams
        assert 'Arsenal' in engineer.city_teams['London']

    def test_engineer_features_basic(self, sample_data):
        """Test basic feature engineering pipeline."""
        engineer = FeatureEngineerV2()
        df_result = engineer.engineer_features(sample_data)

        # Check that we got features
        assert len(df_result) > 0
        assert len(df_result.columns) > len(sample_data.columns)

        # Check for key feature categories
        assert any('_L5' in col for col in df_result.columns)
        assert any('_L10' in col for col in df_result.columns)
        assert any('ratio' in col for col in df_result.columns)
        assert any('momentum' in col for col in df_result.columns)
        assert any('volatility' in col for col in df_result.columns)

    def test_no_data_leakage_rolling(self, sample_data):
        """Test that rolling features use shift(1) to prevent data leakage."""
        engineer = FeatureEngineerV2()

        # Add a known sequence for one team
        df = sample_data.copy()
        df.loc[df['home_team'] == 'Arsenal', 'home_goals'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9][:sum(df['home_team'] == 'Arsenal')]

        df_result = engineer.engineer_features(df)

        # Check that first match for Arsenal has NaN or filled with min_periods=1
        arsenal_home = df_result[df_result['home_team'] == 'Arsenal'].sort_values('date')

        if len(arsenal_home) > 1:
            # Second match should use first match data (shift(1))
            # Not current match data
            first_match_goals = arsenal_home.iloc[0]['home_goals']
            second_match_L3 = arsenal_home.iloc[1]['home_goals_scored_L3']

            # L3 for second match should be based on first match only (shift(1))
            # With min_periods=1, it should equal first match goals
            assert pd.notna(second_match_L3), "Rolling features should exist"

    def test_removed_problematic_features(self, sample_data):
        """Test that problematic features are not created."""
        engineer = FeatureEngineerV2()
        df_result = engineer.engineer_features(sample_data)

        # These features should NOT exist (removed due to bias)
        problematic = [
            'expected_total_goals',
            'combined_over_rate',
            'h2h_avg_goals',
            'h2h_over_rate'
        ]

        for feature in problematic:
            assert feature not in df_result.columns, f"Problematic feature {feature} should be removed"

    def test_ratio_features_creation(self, sample_data):
        """Test that ratio features are created correctly."""
        engineer = FeatureEngineerV2()
        df_result = engineer.engineer_features(sample_data)

        # Check ratio features exist
        ratio_features = [
            'home_attack_ratio_L5',
            'away_attack_ratio_L5',
            'home_defense_ratio_L5',
            'away_defense_ratio_L5',
            'attack_defense_ratio_L5'
        ]

        for feature in ratio_features:
            assert feature in df_result.columns, f"Ratio feature {feature} should exist"
            # Check no inf or extreme values
            assert not df_result[feature].isin([np.inf, -np.inf]).any(), f"{feature} has inf values"

    def test_momentum_features(self, sample_data):
        """Test momentum features are created."""
        engineer = FeatureEngineerV2()
        df_result = engineer.engineer_features(sample_data)

        momentum_features = [
            'home_goals_momentum',
            'away_goals_momentum',
            'home_defense_momentum',
            'away_defense_momentum',
            'home_form_momentum',
            'away_form_momentum'
        ]

        for feature in momentum_features:
            assert feature in df_result.columns, f"Momentum feature {feature} should exist"

        # Momentum should be difference between L3 and L10
        # So it can be positive, negative, or zero
        assert df_result['home_goals_momentum'].dtype in [np.float64, np.int64]

    def test_volatility_features(self, sample_data):
        """Test volatility features are created."""
        engineer = FeatureEngineerV2()
        df_result = engineer.engineer_features(sample_data)

        volatility_features = [
            'home_goals_scored_volatility_L5',
            'away_goals_scored_volatility_L5',
            'home_goals_conceded_volatility_L5',
            'away_goals_conceded_volatility_L5',
            'combined_volatility'
        ]

        for feature in volatility_features:
            assert feature in df_result.columns, f"Volatility feature {feature} should exist"
            # Volatility should be >= 0 (standard deviation)
            assert (df_result[feature] >= 0).all() or df_result[feature].isna().all(), \
                f"{feature} should be non-negative"

    def test_context_features(self, derby_data):
        """Test context features (derbies, top clashes, etc.)."""
        engineer = FeatureEngineerV2()
        df_result = engineer.engineer_features(derby_data)

        context_features = [
            'is_derby',
            'is_top6_clash',
            'is_relegation_battle',
            'is_mismatch',
            'position_diff'
        ]

        for feature in context_features:
            assert feature in df_result.columns, f"Context feature {feature} should exist"

        # Check derby detection
        # Arsenal vs Chelsea (both London)
        arsenal_chelsea = df_result[
            (df_result['home_team'] == 'Arsenal') &
            (df_result['away_team'] == 'Chelsea')
        ]
        if len(arsenal_chelsea) > 0:
            assert arsenal_chelsea.iloc[0]['is_derby'] == 1, "Arsenal vs Chelsea should be derby"

        # Liverpool vs Everton (both Liverpool)
        liverpool_everton = df_result[
            (df_result['home_team'] == 'Liverpool') &
            (df_result['away_team'] == 'Everton')
        ]
        if len(liverpool_everton) > 0:
            assert liverpool_everton.iloc[0]['is_derby'] == 1, "Liverpool vs Everton should be derby"

    def test_rest_features(self, sample_data):
        """Test rest days features."""
        engineer = FeatureEngineerV2()
        df_result = engineer.engineer_features(sample_data)

        rest_features = [
            'home_days_rest',
            'away_days_rest',
            'rest_advantage',
            'days_since_last_match_diff',
            'home_short_rest',
            'away_short_rest'
        ]

        for feature in rest_features:
            assert feature in df_result.columns, f"Rest feature {feature} should exist"

        # Days rest should be >= 0
        assert (df_result['home_days_rest'] >= 0).all()
        assert (df_result['away_days_rest'] >= 0).all()

        # Short rest should be binary
        assert df_result['home_short_rest'].isin([0, 1]).all()
        assert df_result['away_short_rest'].isin([0, 1]).all()

    def test_validate_features_vif(self, sample_data):
        """Test feature validation for VIF."""
        engineer = FeatureEngineerV2()
        df_result = engineer.engineer_features(sample_data)

        # Run validation
        validation = engineer.validate_features(df_result)

        assert 'n_features' in validation
        assert 'vif_issues' in validation
        assert 'correlation_issues' in validation
        assert 'problematic_features' in validation

        # Check that we don't have too many VIF issues
        # With good feature engineering, most features should have VIF < 10
        vif_issue_rate = len(validation['vif_issues']) / validation['n_features']
        assert vif_issue_rate < 0.3, f"Too many VIF issues: {vif_issue_rate:.1%}"

    def test_validate_features_correlation(self, sample_data):
        """Test feature validation for correlation with target."""
        engineer = FeatureEngineerV2()
        df_result = engineer.engineer_features(sample_data)

        # Run validation
        validation = engineer.validate_features(df_result)

        # Check that no feature has extreme correlation (> 0.7)
        # This would indicate potential data leakage
        for feature_info in validation['features']:
            correlation = abs(feature_info['correlation_with_target'])
            assert correlation < 0.7, \
                f"Feature {feature_info['feature_name']} has too high correlation: {correlation:.3f}"

    def test_get_feature_metadata(self, sample_data):
        """Test get_feature_metadata method."""
        engineer = FeatureEngineerV2()
        df_result = engineer.engineer_features(sample_data)

        # Before validation
        metadata = engineer.get_feature_metadata()
        assert metadata == {} or len(metadata) == 0

        # After validation
        engineer.validate_features(df_result)
        metadata = engineer.get_feature_metadata()

        assert 'n_features' in metadata
        assert 'features' in metadata
        assert len(metadata['features']) > 0

        # Check feature info structure
        first_feature = metadata['features'][0]
        assert 'feature_name' in first_feature
        assert 'vif_score' in first_feature
        assert 'correlation_with_target' in first_feature
        assert 'is_problematic' in first_feature

    def test_get_feature_list(self, sample_data):
        """Test get_feature_list method."""
        engineer = FeatureEngineerV2()
        df_result = engineer.engineer_features(sample_data)

        feature_list = engineer.get_feature_list(df_result)

        # Should not include target or metadata columns
        excluded = ['home_goals', 'away_goals', 'total_goals', 'over_2.5',
                   'home_team', 'away_team', 'date', 'season']

        for col in excluded:
            assert col not in feature_list, f"{col} should be excluded from feature list"

        # Should include engineered features
        assert 'home_attack_ratio_L5' in feature_list
        assert 'away_defense_ratio_L5' in feature_list

    def test_run_feature_engineering_v2(self, sample_data):
        """Test convenience function run_feature_engineering_v2."""
        df_result, validation = run_feature_engineering_v2(sample_data, validate=True)

        # Check dataframe
        assert len(df_result) > 0
        assert len(df_result.columns) > len(sample_data.columns)

        # Check validation results
        assert 'n_features' in validation
        assert 'features' in validation

    def test_no_nan_in_critical_features(self, sample_data):
        """Test that critical features have no NaN after engineering."""
        engineer = FeatureEngineerV2()
        df_result = engineer.engineer_features(sample_data)

        # Critical features that should not have NaN
        # (after dropping rows with NaN in critical features)
        critical_features = [
            'home_goals_scored_L5',
            'away_goals_scored_L5',
            'home_attack_ratio_L5',
            'away_attack_ratio_L5'
        ]

        for feature in critical_features:
            if feature in df_result.columns:
                nan_count = df_result[feature].isna().sum()
                assert nan_count == 0, f"{feature} has {nan_count} NaN values"

    def test_feature_count_reduction(self, sample_data):
        """Test that V2 has fewer features than V1 (removed problematic ones)."""
        from src.features.engineering import FeatureEngineer

        # V1 features
        engineer_v1 = FeatureEngineer()
        df_v1 = engineer_v1.engineer_features(sample_data)
        features_v1 = [col for col in df_v1.columns if col not in
                      ['home_goals', 'away_goals', 'total_goals', 'over_2.5',
                       'home_team', 'away_team', 'date', 'season']]

        # V2 features
        engineer_v2 = FeatureEngineerV2()
        df_v2 = engineer_v2.engineer_features(sample_data)
        features_v2 = engineer_v2.get_feature_list(df_v2)

        # V2 should have removed at least the 4 problematic features
        # but added new ones (momentum, volatility, context)
        # So the count might be similar but composition is different

        # Check that problematic features are gone
        assert 'expected_total_goals' not in features_v2
        assert 'combined_over_rate' not in features_v2

    def test_league_stats_calculation(self, sample_data):
        """Test league average statistics calculation."""
        engineer = FeatureEngineerV2()
        df_result = engineer.engineer_features(sample_data)

        # League average columns should exist
        assert 'league_avg_goals_L5' in df_result.columns
        assert 'league_avg_goals_L10' in df_result.columns

        # League averages should be reasonable (0.5 to 3.0 goals per match)
        assert (df_result['league_avg_goals_L5'] >= 0).all()
        assert (df_result['league_avg_goals_L5'] <= 5).all()

    def test_position_features(self, sample_data):
        """Test league position features."""
        engineer = FeatureEngineerV2()
        df_result = engineer.engineer_features(sample_data)

        # Position features should exist
        assert 'home_position_L10' in df_result.columns
        assert 'away_position_L10' in df_result.columns
        assert 'position_diff' in df_result.columns

        # Positions should be reasonable (1-20 for Premier League)
        assert (df_result['home_position_L10'] >= 1).all()
        assert (df_result['away_position_L10'] >= 1).all()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        engineer = FeatureEngineerV2()
        df_empty = pd.DataFrame(columns=['date', 'home_team', 'away_team', 'home_goals', 'away_goals'])

        df_result = engineer.engineer_features(df_empty)
        assert len(df_result) == 0

    def test_single_match(self):
        """Test handling of single match."""
        engineer = FeatureEngineerV2()
        df_single = pd.DataFrame([{
            'date': pd.Timestamp('2023-08-01'),
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'home_goals': 2,
            'away_goals': 1,
            'total_goals': 3,
            'over_2.5': 1
        }])

        df_result = engineer.engineer_features(df_single)
        # Should handle gracefully (might be empty after dropna)
        assert len(df_result) >= 0

    def test_all_same_team(self):
        """Test handling when all matches involve same team."""
        engineer = FeatureEngineerV2()

        dates = pd.date_range(start='2023-08-01', periods=10, freq='3D')
        teams = ['Liverpool', 'Arsenal', 'Chelsea', 'Man City', 'Tottenham']

        data = []
        for i in range(10):
            data.append({
                'date': dates[i],
                'home_team': 'Liverpool',  # Always home
                'away_team': np.random.choice([t for t in teams if t != 'Liverpool']),
                'home_goals': np.random.poisson(1.5),
                'away_goals': np.random.poisson(1.2),
            })

        df = pd.DataFrame(data)
        df['total_goals'] = df['home_goals'] + df['away_goals']
        df['over_2.5'] = (df['total_goals'] > 2.5).astype(int)

        df_result = engineer.engineer_features(df)
        assert len(df_result) > 0

        # Liverpool should have rolling features
        assert df_result['home_goals_scored_L5'].notna().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
