"""
Structured logging system for production monitoring
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ProductionLogger:
    """
    Production-grade logging with structured output

    Logs to:
    - Console (INFO and above)
    - File (all levels with rotation)
    """

    def __init__(self, log_path: str, log_level: str = "INFO"):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_path / f"distill_{datetime.utcnow().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("distill")

    def info(self, message: str, **kwargs):
        """Log info message"""
        extra_info = " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self.logger.info(f"{message}{extra_info}")

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        extra_info = " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self.logger.warning(f"{message}{extra_info}")

    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message"""
        extra_info = " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        error_detail = f" | error={str(error)}" if error else ""
        self.logger.error(f"{message}{extra_info}{error_detail}")

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        extra_info = " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self.logger.debug(f"{message}{extra_info}")


def get_logger(log_path: str, log_level: str = "INFO") -> ProductionLogger:
    """Get production logger instance"""
    return ProductionLogger(log_path, log_level)
