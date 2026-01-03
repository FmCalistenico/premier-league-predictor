"""
Feature engineering module.
Creates predictive features from match data while preventing data leakage.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from datetime import timedelta

from ..utils import LoggerMixin


class FeatureEngineer(LoggerMixin):
    """
    Creates features for Premier League match prediction.

    CRITICAL: All rolling features use .shift(1) to prevent data leakage.
    This ensures we only use information available BEFORE the match.

    Features created:
    - Rolling statistics (goals, form, over rates)
    - Match-level features (expected goals, strength indices)
    - Rest days features
    - Head-to-head statistics
    """

    def __init__(self):
        """Initialize feature engineer."""
        self.feature_columns = []
        self.logger.info("FeatureEngineer initialized")

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [3, 5, 10]
    ) -> pd.DataFrame:
        """
        Create rolling statistics features for each team.

        CRITICAL: Uses .shift(1) BEFORE .rolling() to prevent data leakage.
        This ensures we only use matches that happened BEFORE the current match.

        Args:
            df: DataFrame with match data
            windows: List of rolling window sizes (default: [3, 5, 10])

        Returns:
            DataFrame with rolling features added

        Features created for each window:
            - goals_scored_L{window}: Avg goals scored in last N matches
            - goals_conceded_L{window}: Avg goals conceded in last N matches
            - over_rate_L{window}: Rate of over 2.5 goals in last N matches
            - goal_diff_L{window}: Avg goal difference in last N matches
        """
        self.logger.info(f"Creating rolling features with windows: {windows}")

        df_features = df.copy()

        # Ensure data is sorted by date
        df_features = df_features.sort_values('date').reset_index(drop=True)

        # Create team-level data for home teams
        home_data = df_features[['date', 'home_team_name', 'home_goals', 'away_goals', 'over_2.5']].copy()
        home_data.columns = ['date', 'team_name', 'goals_scored', 'goals_conceded', 'over_2.5']

        # Create team-level data for away teams
        away_data = df_features[['date', 'away_team_name', 'away_goals', 'home_goals', 'over_2.5']].copy()
        away_data.columns = ['date', 'team_name', 'goals_scored', 'goals_conceded', 'over_2.5']

        # Combine home and away data
        team_data = pd.concat([home_data, away_data], ignore_index=True)
        team_data = team_data.sort_values(['team_name', 'date']).reset_index(drop=True)

        # Calculate goal difference
        team_data['goal_diff'] = team_data['goals_scored'] - team_data['goals_conceded']

        # CRITICAL: Create rolling features with .shift(1) to prevent data leakage
        for window in windows:
            self.logger.debug(f"Creating rolling features for window={window}")

            # Group by team and calculate rolling stats
            grouped = team_data.groupby('team_name', group_keys=False)

            # IMPORTANT: shift(1) means we skip the current match and use only previous matches
            team_data[f'goals_scored_L{window}'] = (
                grouped['goals_scored']
                .shift(1)  # Skip current match
                .rolling(window=window, min_periods=1)
                .mean()
            )

            team_data[f'goals_conceded_L{window}'] = (
                grouped['goals_conceded']
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

            team_data[f'over_rate_L{window}'] = (
                grouped['over_2.5']
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

            team_data[f'goal_diff_L{window}'] = (
                grouped['goal_diff']
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

        # Create feature columns list
        feature_cols = []
        for window in windows:
            feature_cols.extend([
                f'goals_scored_L{window}',
                f'goals_conceded_L{window}',
                f'over_rate_L{window}',
                f'goal_diff_L{window}'
            ])

        # Merge features back to original dataframe
        # For home teams
        home_features = team_data[['date', 'team_name'] + feature_cols].copy()
        home_features.columns = ['date', 'home_team_name'] + [f'home_{col}' for col in feature_cols]

        df_features = df_features.merge(
            home_features,
            on=['date', 'home_team_name'],
            how='left'
        )

        # For away teams
        away_features = team_data[['date', 'team_name'] + feature_cols].copy()
        away_features.columns = ['date', 'away_team_name'] + [f'away_{col}' for col in feature_cols]

        df_features = df_features.merge(
            away_features,
            on=['date', 'away_team_name'],
            how='left'
        )

        # Update feature columns list
        self.feature_columns.extend([f'home_{col}' for col in feature_cols])
        self.feature_columns.extend([f'away_{col}' for col in feature_cols])

        self.logger.info(f"Created {len(feature_cols) * 2} rolling features")

        return df_features

    def create_match_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create match-level comparative features.

        Args:
            df: DataFrame with rolling features

        Returns:
            DataFrame with match features added

        Features created:
            - expected_total_goals: Combined average goals scored
            - form_difference: Difference in goal difference
            - attack_strength: Normalized attack strength vs league average
            - defense_strength: Normalized defense strength vs league average
            - combined_over_rate: Combined rate of over 2.5 goals
        """
        self.logger.info("Creating match-level features")

        df_features = df.copy()

        # Use the largest window for match features (typically 10)
        # Check which windows are available
        available_windows = []
        for col in df_features.columns:
            if 'goals_scored_L' in col:
                window = int(col.split('_L')[-1])
                if window not in available_windows:
                    available_windows.append(window)

        if not available_windows:
            self.logger.warning("No rolling features found. Run create_rolling_features first.")
            return df_features

        # Use the largest window
        window = max(available_windows)
        self.logger.debug(f"Using window={window} for match features")

        # Expected total goals (combined attack strength)
        df_features['expected_total_goals'] = (
            df_features[f'home_goals_scored_L{window}'].fillna(0) +
            df_features[f'away_goals_scored_L{window}'].fillna(0)
        )

        # Form difference (goal difference)
        df_features['form_difference'] = (
            df_features[f'home_goal_diff_L{window}'].fillna(0) -
            df_features[f'away_goal_diff_L{window}'].fillna(0)
        )

        # Attack strength (normalized vs league average)
        league_avg_attack = df_features[f'home_goals_scored_L{window}'].mean()

        df_features['home_attack_strength'] = (
            df_features[f'home_goals_scored_L{window}'] / league_avg_attack
            if league_avg_attack > 0 else 1
        )

        df_features['away_attack_strength'] = (
            df_features[f'away_goals_scored_L{window}'] / league_avg_attack
            if league_avg_attack > 0 else 1
        )

        df_features['attack_strength_diff'] = (
            df_features['home_attack_strength'] - df_features['away_attack_strength']
        )

        # Defense strength (normalized vs league average)
        # Lower is better for defense
        league_avg_defense = df_features[f'home_goals_conceded_L{window}'].mean()

        df_features['home_defense_strength'] = (
            df_features[f'home_goals_conceded_L{window}'] / league_avg_defense
            if league_avg_defense > 0 else 1
        )

        df_features['away_defense_strength'] = (
            df_features[f'away_goals_conceded_L{window}'] / league_avg_defense
            if league_avg_defense > 0 else 1
        )

        df_features['defense_strength_diff'] = (
            df_features['away_defense_strength'] - df_features['home_defense_strength']
        )

        # Combined over rate
        df_features['combined_over_rate'] = (
            df_features[f'home_over_rate_L{window}'].fillna(0.5) +
            df_features[f'away_over_rate_L{window}'].fillna(0.5)
        ) / 2

        # Add to feature columns
        match_features = [
            'expected_total_goals',
            'form_difference',
            'home_attack_strength',
            'away_attack_strength',
            'attack_strength_diff',
            'home_defense_strength',
            'away_defense_strength',
            'defense_strength_diff',
            'combined_over_rate'
        ]

        self.feature_columns.extend(match_features)

        self.logger.info(f"Created {len(match_features)} match-level features")

        return df_features

    def create_rest_days_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rest days features.

        Args:
            df: DataFrame with match data

        Returns:
            DataFrame with rest days features

        Features created:
            - home_days_rest: Days since last match for home team
            - away_days_rest: Days since last match for away team
            - rest_advantage: Difference in rest days (home - away)
        """
        self.logger.info("Creating rest days features")

        df_features = df.copy()

        # Ensure data is sorted
        df_features = df_features.sort_values('date').reset_index(drop=True)

        # Create team-level data
        home_data = df_features[['date', 'home_team_name']].copy()
        home_data.columns = ['date', 'team_name']

        away_data = df_features[['date', 'away_team_name']].copy()
        away_data.columns = ['date', 'team_name']

        # Combine and sort
        team_data = pd.concat([home_data, away_data], ignore_index=True)
        team_data = team_data.sort_values(['team_name', 'date']).reset_index(drop=True)

        # Calculate days since last match
        team_data['prev_date'] = team_data.groupby('team_name')['date'].shift(1)
        team_data['days_rest'] = (team_data['date'] - team_data['prev_date']).dt.days

        # For home teams
        home_rest = team_data[['date', 'team_name', 'days_rest']].copy()
        home_rest.columns = ['date', 'home_team_name', 'home_days_rest']

        df_features = df_features.merge(
            home_rest,
            on=['date', 'home_team_name'],
            how='left'
        )

        # For away teams
        away_rest = team_data[['date', 'team_name', 'days_rest']].copy()
        away_rest.columns = ['date', 'away_team_name', 'away_days_rest']

        df_features = df_features.merge(
            away_rest,
            on=['date', 'away_team_name'],
            how='left'
        )

        # Rest advantage (positive means home team has more rest)
        df_features['rest_advantage'] = (
            df_features['home_days_rest'].fillna(7) -
            df_features['away_days_rest'].fillna(7)
        )

        # Add to feature columns
        rest_features = ['home_days_rest', 'away_days_rest', 'rest_advantage']
        self.feature_columns.extend(rest_features)

        self.logger.info(f"Created {len(rest_features)} rest days features")

        return df_features

    def create_head_to_head_features(
        self,
        df: pd.DataFrame,
        h2h_window: int = 5
    ) -> pd.DataFrame:
        """
        Create head-to-head features between teams.

        Args:
            df: DataFrame with match data
            h2h_window: Number of previous H2H matches to consider

        Returns:
            DataFrame with H2H features

        Features created:
            - h2h_avg_goals: Average total goals in last N H2H matches
            - h2h_over_rate: Rate of over 2.5 goals in last N H2H matches
        """
        self.logger.info(f"Creating head-to-head features (window={h2h_window})")

        df_features = df.copy()

        # Ensure data is sorted
        df_features = df_features.sort_values('date').reset_index(drop=True)

        # Create matchup identifier (order-independent)
        df_features['matchup'] = df_features.apply(
            lambda row: tuple(sorted([row['home_team_name'], row['away_team_name']])),
            axis=1
        )

        # Initialize H2H features
        df_features['h2h_avg_goals'] = np.nan
        df_features['h2h_over_rate'] = np.nan

        # Calculate H2H features for each match
        for idx, row in df_features.iterrows():
            matchup = row['matchup']
            current_date = row['date']

            # Get previous matches between these teams
            h2h_matches = df_features[
                (df_features['matchup'] == matchup) &
                (df_features['date'] < current_date)
            ].tail(h2h_window)

            if len(h2h_matches) > 0:
                # Average goals in H2H matches
                df_features.loc[idx, 'h2h_avg_goals'] = h2h_matches['total_goals'].mean()

                # Over 2.5 rate in H2H matches
                if 'over_2.5' in h2h_matches.columns:
                    df_features.loc[idx, 'h2h_over_rate'] = h2h_matches['over_2.5'].mean()

        # Fill NaN values for teams with no H2H history
        df_features['h2h_avg_goals'] = df_features['h2h_avg_goals'].fillna(
            df_features['total_goals'].mean() if 'total_goals' in df_features.columns else 2.5
        )
        df_features['h2h_over_rate'] = df_features['h2h_over_rate'].fillna(0.5)

        # Drop temporary matchup column
        df_features = df_features.drop('matchup', axis=1)

        # Add to feature columns
        h2h_features = ['h2h_avg_goals', 'h2h_over_rate']
        self.feature_columns.extend(h2h_features)

        self.logger.info(f"Created {len(h2h_features)} head-to-head features")

        return df_features

    def engineer_features(
        self,
        df: pd.DataFrame,
        rolling_windows: List[int] = [3, 5, 10],
        h2h_window: int = 5,
        min_matches_required: int = 5
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.

        Pipeline steps:
        1. Create rolling features
        2. Create match-level features
        3. Create rest days features
        4. Create head-to-head features
        5. Filter matches without sufficient history

        Args:
            df: DataFrame with transformed match data
            rolling_windows: Windows for rolling features
            h2h_window: Window for H2H features
            min_matches_required: Minimum matches needed for a team

        Returns:
            DataFrame ready for modeling with all features

        Example:
            >>> engineer = FeatureEngineer()
            >>> df_features = engineer.engineer_features(df_transformed)
        """
        self.logger.info("Starting complete feature engineering pipeline")
        self.logger.info(f"Input data: {len(df)} matches")

        # Reset feature columns list
        self.feature_columns = []

        df_features = df.copy()
        initial_rows = len(df_features)

        # Ensure required columns exist
        required_cols = [
            'date', 'home_team_name', 'away_team_name',
            'home_goals', 'away_goals', 'total_goals'
        ]
        missing_cols = [col for col in required_cols if col not in df_features.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Step 1: Rolling features
        self.logger.info("Step 1/4: Creating rolling features")
        df_features = self.create_rolling_features(df_features, windows=rolling_windows)

        # Step 2: Match features
        self.logger.info("Step 2/4: Creating match-level features")
        df_features = self.create_match_features(df_features)

        # Step 3: Rest days features
        self.logger.info("Step 3/4: Creating rest days features")
        df_features = self.create_rest_days_features(df_features)

        # Step 4: Head-to-head features
        self.logger.info("Step 4/4: Creating head-to-head features")
        df_features = self.create_head_to_head_features(df_features, h2h_window=h2h_window)

        # Step 5: Filter matches without sufficient history
        self.logger.info(f"Filtering matches with less than {min_matches_required} team matches")

        # Check for missing values in critical features
        critical_features = [f'home_goals_scored_L{rolling_windows[0]}', f'away_goals_scored_L{rolling_windows[0]}']

        # Keep only matches where both teams have sufficient history
        df_features = df_features.dropna(subset=critical_features)

        final_rows = len(df_features)
        removed_rows = initial_rows - final_rows

        self.logger.info(f"Removed {removed_rows} matches without sufficient history")
        self.logger.info(f"Final dataset: {final_rows} matches with {len(self.feature_columns)} features")

        # Log feature summary
        self.logger.info("\nFeature Summary:")
        self.logger.info(f"  - Rolling features: {sum(1 for f in self.feature_columns if 'L' in f and ('home_' in f or 'away_' in f))}")
        self.logger.info(f"  - Match features: {sum(1 for f in self.feature_columns if any(x in f for x in ['expected', 'form', 'strength', 'combined']))}")
        self.logger.info(f"  - Rest features: {sum(1 for f in self.feature_columns if 'rest' in f)}")
        self.logger.info(f"  - H2H features: {sum(1 for f in self.feature_columns if 'h2h' in f)}")

        return df_features

    def get_feature_names(self) -> List[str]:
        """
        Get list of all created feature names.

        Returns:
            List of feature column names

        Example:
            >>> engineer = FeatureEngineer()
            >>> df_features = engineer.engineer_features(df)
            >>> features = engineer.get_feature_names()
        """
        return self.feature_columns.copy()
