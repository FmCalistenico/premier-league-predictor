"""
Logging utilities module.
Provides structured logging with rotation and custom formatters.
"""

import logging
import logging.config
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import yaml
from datetime import datetime


def setup_logging(
    config_path: Optional[str] = None,
    default_level: int = logging.INFO,
    log_dir: Optional[str] = None
) -> None:
    """
    Setup logging configuration from YAML file.

    Args:
        config_path: Path to logging.yaml file. If None, uses default path.
        default_level: Default logging level if config file not found.
        log_dir: Directory for log files. If None, uses default './logs'.

    Example:
        >>> setup_logging()
        >>> logger = get_logger(__name__)
        >>> logger.info("Logging configured successfully")
    """
    # Default log directory
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / 'logs'
    else:
        log_dir = Path(log_dir)

    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'logging.yaml'

    # Load logging configuration
    if Path(config_path).exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Update file paths in handlers to use absolute paths
            if 'handlers' in config:
                for handler_name, handler_config in config['handlers'].items():
                    if 'filename' in handler_config:
                        # Convert relative path to absolute
                        filename = handler_config['filename']
                        if not Path(filename).is_absolute():
                            handler_config['filename'] = str(log_dir / Path(filename).name)

            logging.config.dictConfig(config)
        except Exception as e:
            # Fallback to basic configuration
            logging.basicConfig(
                level=default_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            logging.error(f"Failed to load logging configuration from {config_path}: {e}")
            logging.info("Using basic logging configuration")
    else:
        # Setup basic configuration with file handler
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add rotating file handler
        log_file = log_dir / 'app.log'
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        logging.getLogger().addHandler(file_handler)

        logging.warning(f"Logging configuration file not found at {config_path}")
        logging.info(f"Using basic logging configuration with file: {log_file}")


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__ of the module)
        level: Optional logging level to set for this logger

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
        >>> logger.error("This is an error message")
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    return logger


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.

    Automatically creates a logger based on the class name.

    Example:
        >>> class MyModel(LoggerMixin):
        ...     def train(self):
        ...         self.logger.info("Training started")
        ...         # training code
        ...         self.logger.info("Training completed")
        >>>
        >>> model = MyModel()
        >>> model.train()
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger


class StructuredLogger:
    """
    Structured logger with additional context and metadata.

    Provides consistent structured logging with timestamps and custom fields.
    """

    def __init__(self, name: str, **default_context):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            **default_context: Default context fields to include in all logs
        """
        self.logger = get_logger(name)
        self.default_context = default_context

    def _format_message(self, message: str, **context) -> str:
        """Format message with context."""
        combined_context = {**self.default_context, **context}

        if combined_context:
            context_str = ' | '.join(f"{k}={v}" for k, v in combined_context.items())
            return f"{message} | {context_str}"
        return message

    def debug(self, message: str, **context) -> None:
        """Log debug message with context."""
        self.logger.debug(self._format_message(message, **context))

    def info(self, message: str, **context) -> None:
        """Log info message with context."""
        self.logger.info(self._format_message(message, **context))

    def warning(self, message: str, **context) -> None:
        """Log warning message with context."""
        self.logger.warning(self._format_message(message, **context))

    def error(self, message: str, **context) -> None:
        """Log error message with context."""
        self.logger.error(self._format_message(message, **context))

    def critical(self, message: str, **context) -> None:
        """Log critical message with context."""
        self.logger.critical(self._format_message(message, **context))

    def exception(self, message: str, **context) -> None:
        """Log exception with traceback and context."""
        self.logger.exception(self._format_message(message, **context))


def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance to use. If None, creates a new logger.

    Example:
        >>> @log_execution_time()
        ... def train_model():
        ...     # training code
        ...     pass
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.info(f"Starting {func.__name__}")

            try:
                result = func(*args, **kwargs)
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"Completed {func.__name__} in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.error(f"Failed {func.__name__} after {elapsed:.2f}s: {str(e)}")
                raise

        return wrapper
    return decorator


def log_function_call(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
    """
    Decorator to log function calls with arguments.

    Args:
        logger: Logger instance to use. If None, creates a new logger.
        level: Logging level to use

    Example:
        >>> @log_function_call()
        ... def process_data(data, normalize=True):
        ...     # processing code
        ...     pass
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        def wrapper(*args, **kwargs):
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)

            logger.log(level, f"Calling {func.__name__}({signature})")

            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func.__name__} returned {result!r}")
                return result
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {str(e)}")
                raise

        return wrapper
    return decorator
