# spaceship_titanic/utils/logger.py
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging():
    """
    Configure logger to wrtite into file and console
    """
    log_dir = Path("spaceship_titanic/data")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter for logs
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()
