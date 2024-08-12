"""Logging."""

from logging.config import dictConfig


def configure_logging():
    """Create logging function."""
    log_config = {
        "version": 1,
        "formatters": {"simple": {"format": "%(levelname)s - %(message)s"}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "custom_logger": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "optuna": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console"]},
    }
    dictConfig(log_config)
