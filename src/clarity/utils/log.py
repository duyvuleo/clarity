"""Configure a logger without adding duplicate handlers or requiring a log path."""

import logging
from pathlib import Path


def configure_logger(logger, logging_file_path=None, level=logging.INFO):
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handlers = [logging.StreamHandler()]
    if logging_file_path:
        path = Path(logging_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    return logger
