# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unified logger configuration for parrots project
"""
import os
import sys
from loguru import logger  as _logger

# Remove default logger
_logger.remove()

# Default log level
LOG_LEVEL = os.environ.get("PARROTS_LOG_LEVEL", "INFO").upper()



# Add custom logger with configurable level
_logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=LOG_LEVEL,
    colorize=True,
)


def set_log_level(level: str):
    """
    Set the log level for the parrots logger.
    
    Args:
        level: str, one of "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    
    Examples:
        >>> from parrots.log import set_log_level
        >>> set_log_level("DEBUG")  # Enable debug logging
        >>> set_log_level("WARNING")  # Only show warnings and errors
    """
    level = level.upper()
    valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    
    if level not in valid_levels:
        raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")
    
    # Remove all handlers and add new one with updated level
    _logger.remove()
    _logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )


def add_file_logger(log_file: str, level: str = "DEBUG"):
    """
    Add a file handler to the logger.

    Args:
        log_file: Path to the log file
        level: Log level for the file handler (default: DEBUG)
    """
    _logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level.upper(),
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )


# Export the configured logger
logger = _logger

__all__ = ["logger", "set_log_level", "add_file_logger", "LOG_LEVEL"]
