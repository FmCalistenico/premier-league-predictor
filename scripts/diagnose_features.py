"""
Diagnose features to identify numerical issues.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logging, get_logger, Config
from src.features import FeatureEngineerV2


def diagnose_features(df, feature_cols):
    """Diagnose features for numerical issues."""
    logger = get_logger(__name__)

    logger.info("\n" + "=" * 80)
    logger.info("FEATURE DIAGNOSTICS")
    logger.info("=" * 80)

    issues = {
        'nan': [],
        'inf': [],
        'extreme': [],
        'zero_variance': [],
        'high_correlation': []
    }

    logger.info(f"\nAnalyzing {len(feature_cols)} features...")

    for col in feature_cols:
        # NaN values
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            issues['nan'].append((col, nan_count))

        # Inf values
        if df[col].dtype in [np.float64, np.float32]:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                issues['inf'].append((col, inf_count))

        # Extreme values
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            max_val = df[col].abs().max()
            if max_val > 1e6:
                issues['extreme'].append((col, max_val))

        # Zero variance
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            std_val = df[col].std()
            if std_val < 1e-10:
                issues['zero_variance'].append((col, std_val))

    # Report issues
    logger.info("\n" + "-" * 80)
    logger.info("ISSUES FOUND:")
    logger.info("-" * 80)

    if issues['nan']:
        logger.warning(f"\n❌ Features with NaN values ({len(issues['nan'])}):")
        for col, count in issues['nan'][:10]:
            logger.warning(f"  - {col}: {count} NaN values")

    if issues['inf']:
        logger.warning(f"\n❌ Features with Inf values ({len(issues['inf'])}):")
        for col, count in issues['inf'][:10]:
            logger.warning(f"  - {col}: {count} Inf values")

    if issues['extreme']:
        logger.warning(f"\n❌ Features with extreme values ({len(issues['extreme'])}):")
        for col, max_val in issues['extreme'][:10]:
            logger.warning(f"  - {col}: max={max_val:.2e}")

    if issues['zero_variance']:
        logger.warning(f"\n❌ Features with zero variance ({len(issues['zero_variance'])}):")
        for col, std_val in issues['zero_variance'][:10]:
            logger.warning(f"  - {col}: std={std_val:.2e}")

    total_issues = (len(issues['nan']) + len(issues['inf']) +
                   len(issues['extreme']) + len(issues['zero_variance']))

    if total_issues == 0:
        logger.info("\n✅ No numerical issues found!")
    else:
        logger.warning(f"\n⚠️  Total issues: {total_issues}")

    return issues


def main():
    """Main diagnostic function."""
    setup_logging()
    logger = get_logger(__name__)
    config = Config()

    logger.info("=" * 80)
    logger.info("FEATURE DIAGNOSTICS FOR V2")
    logger.info("=" * 80)

    try:
        # Load data
        processed_files = list(config.data_processed_path.glob('transformed_*.csv'))
        if not processed_files:
            logger.error("No processed data found.")
            sys.exit(1)

        data_path = max(processed_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"\nLoading data from: {data_path}")

        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])

        # Ensure required columns
        if 'total_goals' not in df.columns:
            df['total_goals'] = df['home_goals'] + df['away_goals']
        if 'over_2.5' not in df.columns:
            df['over_2.5'] = (df['total_goals'] > 2.5).astype(int)

        logger.info(f"Loaded {len(df)} matches")

        # Apply V2 features
        logger.info("\n" + "=" * 80)
        logger.info("Creating V2 features...")
        logger.info("=" * 80)

        engineer = FeatureEngineerV2()
        df_features = engineer.engineer_features(df)

        logger.info(f"✓ Features created: {len(df_features)} matches")

        # Get feature columns
        feature_cols = engineer.get_feature_list(df_features)
        logger.info(f"Feature count: {len(feature_cols)}")

        # Diagnose
        issues = diagnose_features(df_features, feature_cols)

        # Save diagnostics
        output_dir = config.models_path / 'analysis'
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / 'feature_diagnostics_v2.txt', 'w') as f:
            f.write("FEATURE DIAGNOSTICS - V2\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total features: {len(feature_cols)}\n")
            f.write(f"Features with NaN: {len(issues['nan'])}\n")
            f.write(f"Features with Inf: {len(issues['inf'])}\n")
            f.write(f"Features with extreme values: {len(issues['extreme'])}\n")
            f.write(f"Features with zero variance: {len(issues['zero_variance'])}\n\n")

            if issues['nan']:
                f.write("\nFeatures with NaN:\n")
                for col, count in issues['nan']:
                    f.write(f"  - {col}: {count}\n")

            if issues['inf']:
                f.write("\nFeatures with Inf:\n")
                for col, count in issues['inf']:
                    f.write(f"  - {col}: {count}\n")

            if issues['extreme']:
                f.write("\nFeatures with extreme values:\n")
                for col, max_val in issues['extreme']:
                    f.write(f"  - {col}: {max_val:.2e}\n")

            if issues['zero_variance']:
                f.write("\nFeatures with zero variance:\n")
                for col, std_val in issues['zero_variance']:
                    f.write(f"  - {col}: {std_val:.2e}\n")

        logger.info(f"\n✓ Diagnostics saved to: {output_dir / 'feature_diagnostics_v2.txt'}")

        # Summary statistics
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 80)

        numeric_features = df_features[feature_cols].select_dtypes(include=[np.number])

        logger.info(f"\nNumeric features: {len(numeric_features.columns)}")
        logger.info(f"Min value: {numeric_features.min().min():.2e}")
        logger.info(f"Max value: {numeric_features.max().max():.2e}")
        logger.info(f"Mean std: {numeric_features.std().mean():.2e}")

        logger.info("\n" + "=" * 80)

    except Exception as e:
        logger.error(f"\n❌ Diagnostic failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
