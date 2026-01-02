"""
Example usage of utility modules.
Demonstrates how to use Config and logging utilities.
"""

from src.utils import (
    Config,
    setup_logging,
    get_logger,
    LoggerMixin,
    StructuredLogger,
    log_execution_time,
)


# Example 1: Basic configuration usage
def example_config():
    """Example of using Config class."""
    config = Config()

    print("=" * 50)
    print("Configuration Example")
    print("=" * 50)

    # Project info
    print(f"Project: {config.get('project.name')}")
    print(f"Version: {config.get('project.version')}")
    print(f"Random Seed: {config.random_seed}")

    # Paths
    print(f"\nData Raw Path: {config.data_raw_path}")
    print(f"Models Path: {config.models_path}")

    # League settings
    print(f"\nLeague ID: {config.league_id}")
    print(f"Current Season: {config.current_season}")
    print(f"Model Threshold: {config.model_threshold}")
    print(f"Rolling Windows: {config.rolling_windows}")

    # Database
    print(f"\nDatabase: {config.db_config['database']}")
    print(f"DB Connection String: {config.db_connection_string}")

    # Enabled models
    print(f"\nEnabled Models: {config.enabled_models}")

    print()


# Example 2: Basic logging usage
def example_logging():
    """Example of using logging utilities."""
    # Setup logging
    setup_logging()

    # Get logger
    logger = get_logger(__name__)

    print("=" * 50)
    print("Logging Example")
    print("=" * 50)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    print("Check logs/ directory for log files\n")


# Example 3: Using LoggerMixin
class DataProcessor(LoggerMixin):
    """Example class using LoggerMixin."""

    def __init__(self, name: str):
        self.name = name

    def process(self):
        """Process data with logging."""
        self.logger.info(f"Starting data processing for {self.name}")
        # Processing logic here
        self.logger.info(f"Completed data processing for {self.name}")


def example_logger_mixin():
    """Example of using LoggerMixin."""
    setup_logging()

    print("=" * 50)
    print("LoggerMixin Example")
    print("=" * 50)

    processor = DataProcessor("Premier League Data")
    processor.process()

    print()


# Example 4: Structured logging
def example_structured_logging():
    """Example of using StructuredLogger."""
    setup_logging()

    print("=" * 50)
    print("Structured Logging Example")
    print("=" * 50)

    logger = StructuredLogger(
        __name__,
        environment="development",
        version="0.1.0"
    )

    logger.info("Processing match data", match_id=12345, season="2024-2025")
    logger.warning("Low confidence prediction", confidence=0.45, threshold=0.7)
    logger.error("Failed to fetch data", api="football-data", status_code=404)

    print()


# Example 5: Execution time decorator
@log_execution_time()
def train_model():
    """Example function with execution time logging."""
    import time
    print("Training model...")
    time.sleep(2)  # Simulate training
    print("Model trained!")


def example_execution_time():
    """Example of using log_execution_time decorator."""
    setup_logging()

    print("=" * 50)
    print("Execution Time Logging Example")
    print("=" * 50)

    train_model()

    print()


# Example 6: Combined usage in a realistic scenario
class ModelTrainer(LoggerMixin):
    """Example model trainer using both Config and logging."""

    def __init__(self):
        self.config = Config()

    @log_execution_time()
    def train(self):
        """Train models with configuration and logging."""
        self.logger.info("Initializing model training")

        # Get configuration
        models = self.config.enabled_models
        random_seed = self.config.random_seed
        threshold = self.config.model_threshold

        self.logger.info(f"Training {len(models)} models with seed {random_seed}")
        self.logger.info(f"Models to train: {', '.join(models)}")
        self.logger.info(f"Prediction threshold: {threshold}")

        # Simulate training
        for model_name in models:
            self.logger.info(f"Training {model_name}...")
            # Training logic would go here

        self.logger.info("Model training completed successfully")


def example_realistic_usage():
    """Realistic example combining Config and logging."""
    setup_logging()

    print("=" * 50)
    print("Realistic Usage Example")
    print("=" * 50)

    trainer = ModelTrainer()
    trainer.train()

    print()


if __name__ == "__main__":
    # Run all examples
    example_config()
    example_logging()
    example_logger_mixin()
    example_structured_logging()
    example_execution_time()
    example_realistic_usage()

    print("=" * 50)
    print("All examples completed!")
    print("Check the logs/ directory for detailed logs")
    print("=" * 50)
