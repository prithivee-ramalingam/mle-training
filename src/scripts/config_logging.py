import logging
import sys


def setup_logging(log_level='DEBUG', log_path=None, console_log=True):
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if console_log:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_path:
        fh = logging.FileHandler(log_path)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
