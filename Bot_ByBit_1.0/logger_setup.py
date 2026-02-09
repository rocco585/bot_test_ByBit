import logging
from logging.handlers import RotatingFileHandler

def setup_logger(log_path: str = "bot.log") -> logging.Logger:
    logger = logging.getLogger("bybit_bot")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
