"""
Configuration management module.
Handles loading and accessing configuration from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, List
import yaml
from dotenv import load_dotenv


class Config:
    """
    Configuration manager that loads settings from config.yaml and environment variables.

    Singleton pattern ensures only one configuration instance exists.
    """

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None, env_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.yaml file. If None, uses default path.
            env_path: Path to .env file. If None, uses default path.
        """
        if self._initialized:
            return

        # Load environment variables
        if env_path is None:
            env_path = Path(__file__).parent.parent.parent / '.env'

        if os.path.exists(env_path):
            load_dotenv(env_path)

        # Load YAML configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        self._initialized = True

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation key.

        Args:
            key: Configuration key in dot notation (e.g., 'project.name')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config = Config()
            >>> config.get('project.name')
            'premier-league-predictor'
        """
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_env(self, key: str, default: Any = None) -> Any:
        """
        Get environment variable value.

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Environment variable value or default
        """
        return os.getenv(key, default)

    @property
    def league_id(self) -> int:
        """Get Premier League ID."""
        return int(self.get_env('LEAGUE_ID', 39))  # 39 is Premier League in most APIs

    @property
    def current_season(self) -> str:
        """Get current season."""
        return self.get_env('CURRENT_SEASON', '2024-2025')

    @property
    def rolling_windows(self) -> List[int]:
        """Get rolling window sizes for feature engineering."""
        windows = self.get('preprocessing.rolling_windows', [3, 5, 10])
        return windows if isinstance(windows, list) else [3, 5, 10]

    @property
    def model_threshold(self) -> float:
        """Get model prediction threshold."""
        return float(self.get('prediction.confidence_threshold', 0.7))

    @property
    def random_seed(self) -> int:
        """Get random seed for reproducibility."""
        return int(self.get('project.random_seed', 42))

    # Data paths
    @property
    def data_raw_path(self) -> Path:
        """Get raw data directory path."""
        path = self.get('paths.data.raw', './data/raw')
        return Path(path)

    @property
    def data_processed_path(self) -> Path:
        """Get processed data directory path."""
        path = self.get('paths.data.processed', './data/processed')
        return Path(path)

    @property
    def data_final_path(self) -> Path:
        """Get final data directory path."""
        path = self.get('paths.data.final', './data/final')
        return Path(path)

    @property
    def models_path(self) -> Path:
        """Get models directory path."""
        path = self.get('paths.models', './models')
        return Path(path)

    @property
    def logs_path(self) -> Path:
        """Get logs directory path."""
        path = self.get('paths.logs', './logs')
        return Path(path)

    # Database configuration
    @property
    def db_config(self) -> Dict[str, str]:
        """Get database configuration."""
        return {
            'host': self.get_env('DB_HOST', 'localhost'),
            'port': self.get_env('DB_PORT', '5432'),
            'database': self.get_env('DB_NAME', 'premier_league'),
            'user': self.get_env('DB_USER', 'postgres'),
            'password': self.get_env('DB_PASSWORD', ''),
            'engine': self.get_env('DB_ENGINE', 'postgresql'),
        }

    @property
    def db_connection_string(self) -> str:
        """Get database connection string."""
        db = self.db_config
        return f"{db['engine']}://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}"

    # Databricks configuration
    @property
    def databricks_config(self) -> Dict[str, str]:
        """Get Databricks configuration."""
        return {
            'host': self.get_env('DATABRICKS_HOST', ''),
            'token': self.get_env('DATABRICKS_TOKEN', ''),
            'cluster_id': self.get_env('DATABRICKS_CLUSTER_ID', ''),
            'workspace': self.get_env('DATABRICKS_WORKSPACE', ''),
        }

    @property
    def use_databricks(self) -> bool:
        """Check if Databricks is configured and should be used."""
        return bool(self.databricks_config['host'] and self.databricks_config['token'])

    # API configuration
    @property
    def api_keys(self) -> Dict[str, str]:
        """Get API keys for external data sources."""
        return {
            'football_data': self.get_env('FOOTBALL_DATA_API_KEY', ''),
            'sportmonks': self.get_env('SPORTMONKS_API_KEY', ''),
        }

    @property
    def api_config(self) -> Dict[str, Any]:
        """Get API server configuration."""
        return {
            'host': self.get('api.host', '0.0.0.0'),
            'port': int(self.get('api.port', 8000)),
            'debug': self.get('api.debug', False),
            'workers': int(self.get('api.workers', 4)),
            'reload': self.get('api.reload', False),
        }

    # Model configuration
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model training configuration."""
        return self.get('models', {})

    @property
    def enabled_models(self) -> List[str]:
        """Get list of enabled models."""
        algorithms = self.get('models.algorithms', {})
        return [name for name, config in algorithms.items() if config.get('enabled', False)]

    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get('training', {})

    @property
    def preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.get('preprocessing', {})

    # MLflow configuration
    @property
    def mlflow_config(self) -> Dict[str, str]:
        """Get MLflow configuration."""
        return {
            'tracking_uri': self.get_env('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
            'experiment_name': self.get_env('MLFLOW_EXPERIMENT_NAME', 'premier_league_predictor'),
        }

    # AWS configuration
    @property
    def aws_config(self) -> Dict[str, str]:
        """Get AWS configuration."""
        return {
            'access_key_id': self.get_env('AWS_ACCESS_KEY_ID', ''),
            'secret_access_key': self.get_env('AWS_SECRET_ACCESS_KEY', ''),
            'region': self.get_env('AWS_REGION', 'us-east-1'),
            's3_bucket': self.get_env('S3_BUCKET', ''),
        }

    @property
    def use_aws(self) -> bool:
        """Check if AWS is configured and should be used."""
        aws = self.aws_config
        return bool(aws['access_key_id'] and aws['secret_access_key'] and aws['s3_bucket'])

    def reload(self, config_path: Optional[str] = None) -> None:
        """
        Reload configuration from file.

        Args:
            config_path: Path to config.yaml file. If None, uses default path.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def __repr__(self) -> str:
        """String representation."""
        return f"Config(project={self.get('project.name')}, version={self.get('project.version')})"
