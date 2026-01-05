"""
Script to analyze feature importance and identify problematic features.
Analyzes Poisson model coefficients and feature correlations.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import PoissonGoalsModel
from src.utils import setup_logging, get_logger, Config


def analyze_feature_importance(model, feature_cols, top_n=15):
    """
    Analyze feature importance from Poisson model coefficients.

    Args:
        model: Trained PoissonGoalsModel
        feature_cols: List of feature column names
        top_n: Number of top features to display

    Returns:
        DataFrame with feature importance analysis
    """
    logger = get_logger(__name__)

    # Get coefficients from both models
    home_coefs = model.home_model.params[1:]  # Skip intercept
    away_coefs = model.away_model.params[1:]  # Skip intercept

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'home_coef': home_coefs.values,
        'away_coef': away_coefs.values,
        'home_abs_coef': np.abs(home_coefs.values),
        'away_abs_coef': np.abs(away_coefs.values),
        'avg_abs_coef': (np.abs(home_coefs.values) + np.abs(away_coefs.values)) / 2
    })

    # Sort by average absolute coefficient
    importance_df = importance_df.sort_values('avg_abs_coef', ascending=False)

    logger.info(f"\n{'='*80}")
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info(f"{'='*80}\n")

    logger.info(f"Top {top_n} Most Important Features:\n")
    for idx, row in importance_df.head(top_n).iterrows():
        logger.info(f"{row['feature']:40s} | Home: {row['home_coef']:7.4f} | Away: {row['away_coef']:7.4f}")

    return importance_df


def analyze_feature_correlations(df, feature_cols, target_col='over_2.5', top_n=15):
    """
    Analyze correlations between features and target variable.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target variable column name
        top_n: Number of top correlations to display

    Returns:
        DataFrame with correlation analysis
    """
    logger = get_logger(__name__)

    # Calculate correlations with target
    correlations = []
    for feature in feature_cols:
        # Use Spearman correlation (robust to non-linear relationships)
        corr, pval = spearmanr(df[feature].fillna(0), df[target_col])
        correlations.append({
            'feature': feature,
            'spearman_corr': corr,
            'p_value': pval,
            'abs_corr': abs(corr)
        })

    corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)

    logger.info(f"\n{'='*80}")
    logger.info("FEATURE-TARGET CORRELATIONS")
    logger.info(f"{'='*80}\n")

    logger.info(f"Top {top_n} Features Most Correlated with Target:\n")
    for idx, row in corr_df.head(top_n).iterrows():
        significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        logger.info(f"{row['feature']:40s} | Corr: {row['spearman_corr']:7.4f} {significance}")

    return corr_df


def analyze_multicollinearity(df, feature_cols, threshold=10.0):
    """
    Analyze multicollinearity using Variance Inflation Factor (VIF).

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        threshold: VIF threshold for problematic features

    Returns:
        DataFrame with VIF analysis
    """
    logger = get_logger(__name__)
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Prepare data
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    # Calculate VIF for each feature
    vif_data = []
    for i, feature in enumerate(feature_cols):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({
                'feature': feature,
                'VIF': vif,
                'problematic': vif > threshold
            })
        except Exception as e:
            logger.warning(f"Could not calculate VIF for {feature}: {e}")
            vif_data.append({
                'feature': feature,
                'VIF': np.nan,
                'problematic': False
            })

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

    logger.info(f"\n{'='*80}")
    logger.info("MULTICOLLINEARITY ANALYSIS (VIF)")
    logger.info(f"{'='*80}\n")

    problematic = vif_df[vif_df['problematic']]
    if len(problematic) > 0:
        logger.warning(f"Features with VIF > {threshold} (highly collinear):\n")
        for idx, row in problematic.iterrows():
            logger.warning(f"{row['feature']:40s} | VIF: {row['VIF']:7.2f}")
    else:
        logger.info(f"No features with VIF > {threshold}")

    return vif_df


def plot_feature_analysis(importance_df, corr_df, output_dir):
    """
    Create visualizations for feature analysis.

    Args:
        importance_df: Feature importance DataFrame
        corr_df: Feature correlation DataFrame
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Top feature importance (coefficient magnitude)
    top_features = importance_df.head(15)
    ax = axes[0, 0]
    x = np.arange(len(top_features))
    width = 0.35
    ax.barh(x - width/2, top_features['home_coef'], width, label='Home Goals', alpha=0.8)
    ax.barh(x + width/2, top_features['away_coef'], width, label='Away Goals', alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(top_features['feature'], fontsize=8)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Top 15 Features by Coefficient Magnitude')
    ax.legend()
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    # 2. Feature-target correlations
    top_corr = corr_df.head(15)
    ax = axes[0, 1]
    colors = ['red' if x < 0 else 'green' for x in top_corr['spearman_corr']]
    ax.barh(range(len(top_corr)), top_corr['spearman_corr'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_corr)))
    ax.set_yticklabels(top_corr['feature'], fontsize=8)
    ax.set_xlabel('Spearman Correlation with Target')
    ax.set_title('Top 15 Features by Correlation with Over 2.5')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    # 3. Coefficient comparison (home vs away)
    ax = axes[1, 0]
    ax.scatter(importance_df['home_coef'], importance_df['away_coef'], alpha=0.6)
    ax.set_xlabel('Home Goals Coefficient')
    ax.set_ylabel('Away Goals Coefficient')
    ax.set_title('Coefficient Comparison: Home vs Away Models')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.plot([-2, 2], [-2, 2], 'r--', alpha=0.5, label='Equal coefficients')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Distribution of absolute coefficients
    ax = axes[1, 1]
    ax.hist(importance_df['avg_abs_coef'], bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Average Absolute Coefficient')
    ax.set_ylabel('Number of Features')
    ax.set_title('Distribution of Feature Importance')
    ax.axvline(x=importance_df['avg_abs_coef'].median(), color='red',
               linestyle='--', label=f"Median: {importance_df['avg_abs_coef'].median():.3f}")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_dir / 'feature_analysis.png'}")


def identify_problematic_features(importance_df, corr_df, vif_df):
    """
    Identify features that may be causing model bias.

    Args:
        importance_df: Feature importance DataFrame
        corr_df: Feature correlation DataFrame
        vif_df: VIF analysis DataFrame

    Returns:
        Dictionary with categorized problematic features
    """
    logger = get_logger(__name__)

    problematic = {
        'high_correlation': [],
        'high_vif': [],
        'strong_positive_bias': [],
        'recommended_to_remove': []
    }

    # Features with very high correlation to target (might be data leakage indicators)
    high_corr = corr_df[corr_df['abs_corr'] > 0.3]
    problematic['high_correlation'] = high_corr['feature'].tolist()

    # Features with high VIF (multicollinearity)
    high_vif = vif_df[vif_df['VIF'] > 10]
    problematic['high_vif'] = high_vif['feature'].tolist()

    # Features with strong positive coefficients in both models (bias toward Over)
    strong_positive = importance_df[
        (importance_df['home_coef'] > 0.2) &
        (importance_df['away_coef'] > 0.2)
    ]
    problematic['strong_positive_bias'] = strong_positive['feature'].tolist()

    # Combine all for removal recommendation
    all_problematic = set(
        problematic['high_correlation'] +
        problematic['high_vif'] +
        problematic['strong_positive_bias']
    )
    problematic['recommended_to_remove'] = list(all_problematic)

    logger.info(f"\n{'='*80}")
    logger.info("PROBLEMATIC FEATURES IDENTIFICATION")
    logger.info(f"{'='*80}\n")

    logger.warning(f"High Correlation with Target (>0.3): {len(problematic['high_correlation'])} features")
    for feat in problematic['high_correlation']:
        logger.warning(f"  - {feat}")

    logger.warning(f"\nHigh VIF (>10): {len(problematic['high_vif'])} features")
    for feat in problematic['high_vif']:
        logger.warning(f"  - {feat}")

    logger.warning(f"\nStrong Positive Bias: {len(problematic['strong_positive_bias'])} features")
    for feat in problematic['strong_positive_bias']:
        logger.warning(f"  - {feat}")

    logger.info(f"\n{'='*80}")
    logger.info(f"RECOMMENDATION: Consider removing {len(problematic['recommended_to_remove'])} features")
    logger.info(f"{'='*80}")

    return problematic


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description='Analyze features and identify problems in Poisson model'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model (default: models/poisson_model_latest.pkl)'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to training data (default: data/final/training_data_latest.parquet)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results (default: models/analysis/)'
    )

    args = parser.parse_args()

    # Setup
    setup_logging()
    logger = get_logger(__name__)
    config = Config()

    # Paths
    model_path = args.model or config.models_path / 'poisson_model_latest.pkl'
    data_path = args.data or config.data_final_path / 'training_data_latest.parquet'
    output_dir = args.output or config.models_path / 'analysis'

    logger.info("="*80)
    logger.info("FEATURE ANALYSIS FOR POISSON MODEL")
    logger.info("="*80)
    logger.info(f"\nModel: {model_path}")
    logger.info(f"Data:  {data_path}")
    logger.info(f"Output: {output_dir}")

    try:
        # Load model
        logger.info("\nLoading trained model...")
        model = PoissonGoalsModel.load(model_path)

        # Load data
        logger.info("Loading training data...")
        df = pd.read_parquet(data_path)

        feature_cols = [col for col in df.columns if col not in [
            'home_goals', 'away_goals', 'total_goals',
            'over_0.5', 'over_1.5', 'over_2.5', 'over_3.5', 'over_4.5',
            'home_team', 'away_team', 'date', 'season'
        ]]

        logger.info(f"\nAnalyzing {len(feature_cols)} features...")

        # Analysis 1: Feature importance from coefficients
        importance_df = analyze_feature_importance(model, feature_cols)

        # Analysis 2: Feature-target correlations
        corr_df = analyze_feature_correlations(df, feature_cols)

        # Analysis 3: Multicollinearity (VIF)
        vif_df = analyze_multicollinearity(df, feature_cols)

        # Identify problematic features
        problematic = identify_problematic_features(importance_df, corr_df, vif_df)

        # Create visualizations
        plot_feature_analysis(importance_df, corr_df, output_dir)

        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
        corr_df.to_csv(output_dir / 'feature_correlations.csv', index=False)
        vif_df.to_csv(output_dir / 'feature_vif.csv', index=False)

        # Save problematic features list
        with open(output_dir / 'problematic_features.txt', 'w') as f:
            f.write("PROBLEMATIC FEATURES ANALYSIS\n")
            f.write("="*80 + "\n\n")

            f.write(f"High Correlation with Target (>{0.3}):\n")
            for feat in problematic['high_correlation']:
                f.write(f"  - {feat}\n")

            f.write(f"\nHigh VIF (>10 - Multicollinearity):\n")
            for feat in problematic['high_vif']:
                f.write(f"  - {feat}\n")

            f.write(f"\nStrong Positive Bias:\n")
            for feat in problematic['strong_positive_bias']:
                f.write(f"  - {feat}\n")

            f.write(f"\n{'='*80}\n")
            f.write(f"RECOMMENDED TO REMOVE: {len(problematic['recommended_to_remove'])} features\n")
            f.write(f"{'='*80}\n")
            for feat in problematic['recommended_to_remove']:
                f.write(f"  - {feat}\n")

        logger.info(f"\n{'='*80}")
        logger.info("ANALYSIS COMPLETED")
        logger.info(f"{'='*80}")
        logger.info(f"\nResults saved to: {output_dir}")
        logger.info(f"  - feature_importance.csv")
        logger.info(f"  - feature_correlations.csv")
        logger.info(f"  - feature_vif.csv")
        logger.info(f"  - problematic_features.txt")
        logger.info(f"  - feature_analysis.png")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
