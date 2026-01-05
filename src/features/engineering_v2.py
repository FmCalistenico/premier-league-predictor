"""
Feature Engineering V2 - Enhanced version with bias reduction.

This version addresses the Over 2.5 prediction bias by:
1. Removing highly correlated features (expected_total_goals, combined_over_rate)
2. Adding relative/ratio features instead of absolute values
3. Including context-aware features (derbies, top clashes, etc.)
4. Validating VIF and correlations
5. Adding momentum and volatility features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.utils import get_logger


class FeatureEngineerV2:
    """
    Enhanced feature engineering with bias reduction and improved features.

    Key improvements:
    - No circular features (expected_total_goals, combined_over_rate removed)
    - Ratio-based features (relative to league average)
    - Context features (derbies, top clashes, relegation battles)
    - Momentum and volatility features
    - VIF and correlation validation
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.league_stats = {}
        self.feature_metadata = {}

        # Premier League city mapping for derby detection
        self.city_teams = {
            'London': ['Arsenal', 'Chelsea', 'Tottenham', 'West Ham', 'Crystal Palace',
                      'Fulham', 'Brentford', 'QPR', 'Watford', 'Charlton'],
            'Manchester': ['Man United', 'Man City', 'Manchester Utd', 'Manchester City'],
            'Liverpool': ['Liverpool', 'Everton'],
            'Birmingham': ['Aston Villa', 'Birmingham'],
            'Sheffield': ['Sheffield United', 'Sheffield Weds'],
            'Nottingham': ['Nottingham Forest', 'Notts County'],
            'Bristol': ['Bristol City', 'Bristol Rovers']
        }

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to expected format.

        Handles variations like:
        - home_team_name -> home_team
        - away_team_name -> away_team

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with normalized column names
        """
        # Create column mapping
        column_mapping = {}

        # Team name columns
        if 'home_team_name' in df.columns and 'home_team' not in df.columns:
            column_mapping['home_team_name'] = 'home_team'
        if 'away_team_name' in df.columns and 'away_team' not in df.columns:
            column_mapping['away_team_name'] = 'away_team'

        # Apply mapping
        if column_mapping:
            df = df.rename(columns=column_mapping)
            self.logger.info(f"Normalized column names: {column_mapping}")

        # Verify required columns exist
        required_cols = ['date', 'home_team', 'away_team', 'home_goals', 'away_goals']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available: {df.columns.tolist()}")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced feature set with bias reduction.

        Args:
            df: DataFrame with match data (date, home_team, away_team, home_goals, away_goals)

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        self.logger.info("Starting Feature Engineering V2...")

        # Normalize column names (handle home_team_name vs home_team)
        df = self._normalize_column_names(df)

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Stage 1: Rolling features (with shift(1))
        self.logger.info("Stage 1: Creating rolling features...")
        df = self.create_rolling_features(df)

        # Stage 2: Calculate league averages
        self.logger.info("Stage 2: Calculating league statistics...")
        df = self.calculate_league_stats(df)

        # Stage 3: Ratio features (relative to league average)
        self.logger.info("Stage 3: Creating ratio features...")
        df = self.create_ratio_features(df)

        # Stage 4: Momentum features
        self.logger.info("Stage 4: Creating momentum features...")
        df = self.create_momentum_features(df)

        # Stage 5: Volatility features
        self.logger.info("Stage 5: Creating volatility features...")
        df = self.create_volatility_features(df)

        # Stage 6: Context features
        self.logger.info("Stage 6: Creating context features...")
        df = self.create_context_features(df)

        # Stage 7: Rest features
        self.logger.info("Stage 7: Creating rest features...")
        df = self.create_rest_features(df)

        # Drop rows with NaN in critical features
        critical_features = [
            'home_goals_scored_L5', 'away_goals_scored_L5',
            'home_attack_ratio_L5', 'away_attack_ratio_L5'
        ]
        df = df.dropna(subset=critical_features)

        self.logger.info(f"✓ Feature Engineering V2 completed: {len(df)} matches, {len(df.columns)} features")

        return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling statistics with strict data leakage prevention.
        Uses shift(1) before rolling to ensure only past data is used.

        Args:
            df: DataFrame with match data

        Returns:
            DataFrame with rolling features
        """
        windows = [3, 5, 10]

        # Create team-level dataset
        home_df = df[['date', 'home_team', 'home_goals', 'away_goals']].copy()
        home_df.columns = ['date', 'team', 'goals_scored', 'goals_conceded']

        away_df = df[['date', 'away_team', 'away_goals', 'home_goals']].copy()
        away_df.columns = ['date', 'team', 'goals_scored', 'goals_conceded']

        team_df = pd.concat([home_df, away_df], ignore_index=True)
        team_df = team_df.sort_values(['team', 'date']).reset_index(drop=True)

        # Calculate basic metrics
        team_df['goal_diff'] = team_df['goals_scored'] - team_df['goals_conceded']
        team_df['total_goals'] = team_df['goals_scored'] + team_df['goals_conceded']
        team_df['over_2.5'] = (team_df['total_goals'] > 2.5).astype(int)
        team_df['clean_sheet'] = (team_df['goals_conceded'] == 0).astype(int)
        team_df['failed_to_score'] = (team_df['goals_scored'] == 0).astype(int)

        # Rolling windows with shift(1) - CRITICAL for preventing data leakage
        grouped = team_df.groupby('team')

        for window in windows:
            # Goals scored/conceded
            team_df[f'goals_scored_L{window}'] = (
                grouped['goals_scored'].shift(1).rolling(window, min_periods=1).mean()
            )
            team_df[f'goals_conceded_L{window}'] = (
                grouped['goals_conceded'].shift(1).rolling(window, min_periods=1).mean()
            )

            # Goal difference
            team_df[f'goal_diff_L{window}'] = (
                grouped['goal_diff'].shift(1).rolling(window, min_periods=1).mean()
            )

            # Over 2.5 rate (will NOT be used directly in match features)
            team_df[f'over_rate_L{window}'] = (
                grouped['over_2.5'].shift(1).rolling(window, min_periods=1).mean()
            )

            # Clean sheets and failures
            team_df[f'clean_sheet_rate_L{window}'] = (
                grouped['clean_sheet'].shift(1).rolling(window, min_periods=1).mean()
            )
            team_df[f'failed_to_score_rate_L{window}'] = (
                grouped['failed_to_score'].shift(1).rolling(window, min_periods=1).mean()
            )

        # Merge back to original dataframe
        rolling_cols = [col for col in team_df.columns if col not in
                       ['date', 'team', 'goals_scored', 'goals_conceded',
                        'goal_diff', 'total_goals', 'over_2.5', 'clean_sheet', 'failed_to_score']]

        # Home team rolling features
        home_rolling = team_df[['date', 'team'] + rolling_cols].copy()
        home_rolling.columns = ['date', 'home_team'] + [f'home_{col}' for col in rolling_cols]

        # Away team rolling features
        away_rolling = team_df[['date', 'team'] + rolling_cols].copy()
        away_rolling.columns = ['date', 'away_team'] + [f'away_{col}' for col in rolling_cols]

        df = df.merge(home_rolling, on=['date', 'home_team'], how='left')
        df = df.merge(away_rolling, on=['date', 'away_team'], how='left')

        return df

    def calculate_league_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate league-wide average statistics for normalization.

        Args:
            df: DataFrame with rolling features

        Returns:
            DataFrame with league average columns
        """
        # Calculate league averages for each gameweek
        df['gameweek'] = df.groupby(df['date'].dt.to_period('W')).ngroup()

        for window in [5, 10]:
            # League average goals scored
            df[f'league_avg_goals_L{window}'] = (
                df.groupby('gameweek')[f'home_goals_scored_L{window}']
                .transform(lambda x: x.median())
            )

            # League average goals conceded
            df[f'league_avg_conceded_L{window}'] = (
                df.groupby('gameweek')[f'home_goals_conceded_L{window}']
                .transform(lambda x: x.median())
            )

        return df

    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio features relative to league average.
        These are more robust than absolute values.

        Args:
            df: DataFrame with rolling features and league stats

        Returns:
            DataFrame with ratio features
        """
        # Attack strength ratios (relative to league average)
        df['home_attack_ratio_L5'] = (
            df['home_goals_scored_L5'] / (df['league_avg_goals_L5'] + 0.01)
        )
        df['away_attack_ratio_L5'] = (
            df['away_goals_scored_L5'] / (df['league_avg_goals_L5'] + 0.01)
        )
        df['home_attack_ratio_L10'] = (
            df['home_goals_scored_L10'] / (df['league_avg_goals_L10'] + 0.01)
        )
        df['away_attack_ratio_L10'] = (
            df['away_goals_scored_L10'] / (df['league_avg_goals_L10'] + 0.01)
        )

        # Defense strength ratios
        df['home_defense_ratio_L5'] = (
            df['home_goals_conceded_L5'] / (df['league_avg_conceded_L5'] + 0.01)
        )
        df['away_defense_ratio_L5'] = (
            df['away_goals_conceded_L5'] / (df['league_avg_conceded_L5'] + 0.01)
        )
        df['home_defense_ratio_L10'] = (
            df['home_goals_conceded_L10'] / (df['league_avg_conceded_L10'] + 0.01)
        )
        df['away_defense_ratio_L10'] = (
            df['away_goals_conceded_L10'] / (df['league_avg_conceded_L10'] + 0.01)
        )

        # Attack vs Defense matchup ratios
        df['attack_defense_ratio_L5'] = (
            df['home_attack_ratio_L5'] / (df['away_defense_ratio_L5'] + 0.1)
        )
        df['defense_attack_ratio_L5'] = (
            df['away_attack_ratio_L5'] / (df['home_defense_ratio_L5'] + 0.1)
        )

        # Form difference (relative measure)
        df['form_diff_L5'] = df['home_goal_diff_L5'] - df['away_goal_diff_L5']
        df['form_diff_L10'] = df['home_goal_diff_L10'] - df['away_goal_diff_L10']

        # REMOVED: expected_total_goals (too correlated)
        # REMOVED: combined_over_rate (circular)

        return df

    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create momentum features showing trend (improving vs declining).

        Args:
            df: DataFrame with rolling features

        Returns:
            DataFrame with momentum features
        """
        # Goals scored momentum (recent vs longer term)
        df['home_goals_momentum'] = df['home_goals_scored_L3'] - df['home_goals_scored_L10']
        df['away_goals_momentum'] = df['away_goals_scored_L3'] - df['away_goals_scored_L10']

        # Defense momentum
        df['home_defense_momentum'] = df['home_goals_conceded_L10'] - df['home_goals_conceded_L3']
        df['away_defense_momentum'] = df['away_goals_conceded_L10'] - df['away_goals_conceded_L3']
        # Note: Inverted so positive = improving defense

        # Form momentum
        df['home_form_momentum'] = df['home_goal_diff_L3'] - df['home_goal_diff_L10']
        df['away_form_momentum'] = df['away_goal_diff_L3'] - df['away_goal_diff_L10']

        return df

    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility features (standard deviation of recent performance).

        Args:
            df: DataFrame with match data

        Returns:
            DataFrame with volatility features
        """
        # Create team-level dataset for volatility calculation
        home_df = df[['date', 'home_team', 'home_goals', 'away_goals']].copy()
        home_df.columns = ['date', 'team', 'goals_scored', 'goals_conceded']

        away_df = df[['date', 'away_team', 'away_goals', 'home_goals']].copy()
        away_df.columns = ['date', 'team', 'goals_scored', 'goals_conceded']

        team_df = pd.concat([home_df, away_df], ignore_index=True)
        team_df = team_df.sort_values(['team', 'date']).reset_index(drop=True)

        team_df['total_goals'] = team_df['goals_scored'] + team_df['goals_conceded']

        # Calculate rolling standard deviations with shift(1)
        grouped = team_df.groupby('team')

        team_df['goals_scored_volatility_L5'] = (
            grouped['goals_scored'].shift(1).rolling(5, min_periods=2).std().fillna(0)
        )
        team_df['goals_conceded_volatility_L5'] = (
            grouped['goals_conceded'].shift(1).rolling(5, min_periods=2).std().fillna(0)
        )
        team_df['total_goals_volatility_L5'] = (
            grouped['total_goals'].shift(1).rolling(5, min_periods=2).std().fillna(0)
        )

        # Merge back
        vol_cols = ['goals_scored_volatility_L5', 'goals_conceded_volatility_L5', 'total_goals_volatility_L5']

        home_vol = team_df[['date', 'team'] + vol_cols].copy()
        home_vol.columns = ['date', 'home_team'] + [f'home_{col}' for col in vol_cols]

        away_vol = team_df[['date', 'team'] + vol_cols].copy()
        away_vol.columns = ['date', 'away_team'] + [f'away_{col}' for col in vol_cols]

        df = df.merge(home_vol, on=['date', 'home_team'], how='left')
        df = df.merge(away_vol, on=['date', 'away_team'], how='left')

        # Combined volatility
        df['combined_volatility'] = (
            df['home_total_goals_volatility_L5'] + df['away_total_goals_volatility_L5']
        ) / 2

        return df

    def create_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create context-aware features (derbies, top clashes, relegation battles).

        Args:
            df: DataFrame with match data

        Returns:
            DataFrame with context features
        """
        # Derby detection
        df['is_derby'] = df.apply(self._is_derby, axis=1).astype(int)

        # Calculate league positions (rolling)
        df = self._calculate_league_positions(df)

        # Top 6 clash
        df['is_top6_clash'] = (
            (df['home_position_L10'] <= 6) & (df['away_position_L10'] <= 6)
        ).astype(int)

        # Relegation battle (both teams in bottom 5)
        df['is_relegation_battle'] = (
            (df['home_position_L10'] >= 16) & (df['away_position_L10'] >= 16)
        ).astype(int)

        # Top vs Bottom (position difference > 10)
        df['is_mismatch'] = (
            abs(df['home_position_L10'] - df['away_position_L10']) > 10
        ).astype(int)

        # Position difference
        df['position_diff'] = df['home_position_L10'] - df['away_position_L10']

        return df

    def create_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rest days features.

        Args:
            df: DataFrame with match data

        Returns:
            DataFrame with rest features
        """
        df = df.sort_values('date').reset_index(drop=True)

        # Days since last match
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

        # Rest advantage
        df['rest_advantage'] = df['home_days_rest'] - df['away_days_rest']

        # Rest difference (absolute)
        df['days_since_last_match_diff'] = abs(df['rest_advantage'])

        # Short rest flag (< 4 days)
        df['home_short_rest'] = (df['home_days_rest'] < 4).astype(int)
        df['away_short_rest'] = (df['away_days_rest'] < 4).astype(int)

        return df

    def _is_derby(self, row) -> bool:
        """Check if match is a derby (same city)."""
        home_team = row['home_team']
        away_team = row['away_team']

        for city, teams in self.city_teams.items():
            if home_team in teams and away_team in teams:
                return True
        return False

    def _calculate_league_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling league positions based on form.

        Args:
            df: DataFrame with match data

        Returns:
            DataFrame with position columns
        """
        # Create team points dataset
        home_df = df[['date', 'home_team', 'home_goals', 'away_goals']].copy()
        home_df.columns = ['date', 'team', 'goals_for', 'goals_against']
        home_df['points'] = home_df.apply(
            lambda x: 3 if x['goals_for'] > x['goals_against']
            else (1 if x['goals_for'] == x['goals_against'] else 0), axis=1
        )

        away_df = df[['date', 'away_team', 'away_goals', 'home_goals']].copy()
        away_df.columns = ['date', 'team', 'goals_for', 'goals_against']
        away_df['points'] = away_df.apply(
            lambda x: 3 if x['goals_for'] > x['goals_against']
            else (1 if x['goals_for'] == x['goals_against'] else 0), axis=1
        )

        team_df = pd.concat([home_df, away_df], ignore_index=True)
        team_df = team_df.sort_values(['team', 'date']).reset_index(drop=True)

        # Calculate rolling points with shift(1)
        grouped = team_df.groupby('team')
        team_df['points_L10'] = (
            grouped['points'].shift(1).rolling(10, min_periods=1).sum()
        )

        # Merge back and calculate positions
        # Remove duplicates before merging (keep first occurrence for each team-date combination)
        home_pos = team_df[['date', 'team', 'points_L10']].drop_duplicates(subset=['date', 'team'], keep='first').copy()
        home_pos.columns = ['date', 'home_team', 'home_points_L10']

        away_pos = team_df[['date', 'team', 'points_L10']].drop_duplicates(subset=['date', 'team'], keep='first').copy()
        away_pos.columns = ['date', 'away_team', 'away_points_L10']

        df = df.merge(home_pos, on=['date', 'home_team'], how='left')
        df = df.merge(away_pos, on=['date', 'away_team'], how='left')

        # Calculate positions (rank within gameweek)
        df['home_position_L10'] = (
            df.groupby('gameweek')['home_points_L10']
            .rank(method='min', ascending=False)
            .fillna(10)
        )
        df['away_position_L10'] = (
            df.groupby('gameweek')['away_points_L10']
            .rank(method='min', ascending=False)
            .fillna(10)
        )

        return df

    def validate_features(self, df: pd.DataFrame, target_col: str = 'over_2.5') -> Dict:
        """
        Validate features for VIF and correlation issues.

        Args:
            df: DataFrame with features
            target_col: Target variable column name

        Returns:
            Dictionary with validation results
        """
        from scipy.stats import spearmanr
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in [
            'home_goals', 'away_goals', 'total_goals',
            'over_0.5', 'over_1.5', 'over_2.5', 'over_3.5', 'over_4.5',
            'home_team', 'away_team', 'date', 'season', 'gameweek'
        ]]

        # Prepare data
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = df[target_col].values

        validation_results = {
            'n_features': len(feature_cols),
            'features': [],
            'problematic_features': [],
            'vif_issues': [],
            'correlation_issues': []
        }

        self.logger.info("Validating features...")

        # Calculate VIF and correlations
        for i, feature in enumerate(feature_cols):
            try:
                # VIF
                vif = variance_inflation_factor(X.values, i)

                # Spearman correlation with target
                corr, pval = spearmanr(X[feature], y, nan_policy='omit')

                feature_info = {
                    'feature_name': feature,
                    'vif_score': float(vif) if np.isfinite(vif) else 999.0,
                    'correlation_with_target': float(corr),
                    'p_value': float(pval),
                    'is_problematic': False
                }

                # Check if problematic
                if vif > 10:
                    feature_info['is_problematic'] = True
                    feature_info['issue'] = f'High VIF: {vif:.2f}'
                    validation_results['vif_issues'].append(feature)

                if abs(corr) > 0.7:
                    feature_info['is_problematic'] = True
                    feature_info['issue'] = f'High correlation: {corr:.3f}'
                    validation_results['correlation_issues'].append(feature)

                validation_results['features'].append(feature_info)

                if feature_info['is_problematic']:
                    validation_results['problematic_features'].append(feature)

            except Exception as e:
                self.logger.warning(f"Could not validate {feature}: {e}")

        # Store metadata
        self.feature_metadata = validation_results

        # Log summary
        self.logger.info(f"Validation complete:")
        self.logger.info(f"  Total features: {validation_results['n_features']}")
        self.logger.info(f"  Problematic features: {len(validation_results['problematic_features'])}")
        self.logger.info(f"  VIF issues: {len(validation_results['vif_issues'])}")
        self.logger.info(f"  Correlation issues: {len(validation_results['correlation_issues'])}")

        return validation_results

    def get_feature_metadata(self) -> Dict:
        """
        Get feature metadata from last validation.

        Returns:
            Dictionary with feature metadata including VIF and correlations
        """
        if not self.feature_metadata:
            self.logger.warning("No feature metadata available. Run validate_features() first.")
            return {}

        return self.feature_metadata

    def get_feature_list(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of engineered features (excluding target and metadata columns).

        Args:
            df: DataFrame with features

        Returns:
            List of feature column names
        """
        exclude_cols = [
            'home_goals', 'away_goals', 'total_goals',
            'over_0.5', 'over_1.5', 'over_2.5', 'over_3.5', 'over_4.5',
            'home_team', 'away_team', 'date', 'season', 'gameweek',
            'home_points_L10', 'away_points_L10'  # Intermediate columns
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        return feature_cols


def run_feature_engineering_v2(df: pd.DataFrame, validate: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to run complete feature engineering V2 pipeline.

    Args:
        df: Input DataFrame with match data
        validate: Whether to validate features after creation

    Returns:
        Tuple of (engineered DataFrame, validation results)
    """
    engineer = FeatureEngineerV2()
    df_engineered = engineer.engineer_features(df)

    validation_results = {}
    if validate and 'over_2.5' in df_engineered.columns:
        validation_results = engineer.validate_features(df_engineered)

    return df_engineered, validation_results
