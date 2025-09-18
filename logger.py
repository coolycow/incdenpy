import logging
import os
from datetime import datetime

# ----------- Logger setup -----------

class ColorizingStreamHandler(logging.StreamHandler):
    COLOR_MAP = {
        logging.DEBUG: "\033[37m",   # Gray
        logging.INFO: "\033[32m",    # Green
        logging.WARNING: "\033[33m", # Yellow
        logging.ERROR: "\033[31m",   # Red
        logging.CRITICAL: "\033[41m" # Red background
    }
    RESET = "\033[0m"

    def emit(self, record):
        try:
            message = self.format(record)
            color = self.COLOR_MAP.get(record.levelno, self.RESET)
            stream = self.stream
            stream.write(color + message + self.RESET + "\n")
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logger(log_dir="logs", log_filename=None):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if log_filename is None:
        log_filename = datetime.now().strftime("run_%Y%m%d_%H%M%S.log")
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger("LaserDataLogger")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = ColorizingStreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Return logger and path for file log
    return logger, log_path

# ----------- Initialize logger for each worker process -----------

_worker_logger = None

def init_worker_logger(log_dir="logs", log_filename=None):
    global _worker_logger

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if log_filename is None:
        log_filename = datetime.now().strftime("run_%Y%m%d_%H%M%S_worker.log")

    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger("LaserDataLogger")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [PID:%(process)d TID:%(thread)d] %(message)s')

    # Clean previous handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = ColorizingStreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    _worker_logger = logger
# ----------- File processing functions ----------

def get_worker_logger():
    global _worker_logger
    return _worker_logger

def log_config(logger, config_dict):
    logger.debug("Запуск программы с настройками:")
    for k, v in config_dict.items():
        logger.debug(f"{k}: {v}")
    logger.debug("")