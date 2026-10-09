"""Logging helper module for AI Chatbot Mental Health project."""

import os
import sys
import logging
from typing import Optional


def get_logger(name: str = "mental_health_ai", log_file: Optional[str] = "logs/pipeline.log") -> logging.Logger:
    """Get or configure a logger with console and optional file handlers.
    
    Args:
        name: Name of the logger.
        log_file: Path to log file, or None for console only.
        
    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
