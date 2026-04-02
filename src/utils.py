import logging
import time
import numpy as np
import random
import os

def get_logger(name: str):
    """Create and return a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def print_section(title: str):
    """Pretty section header for console output."""
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class Timer:
    """Simple context manager for timing code blocks."""
    def __init__(self, message="Elapsed time"):
        self.message = message

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"{self.message}: {elapsed:.2f} seconds")