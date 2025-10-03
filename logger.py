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

# ----------- Setup function -----------
def setup_logger(log_dir="logs", log_filename=None):
    logger = logging.getLogger("LaserDataLogger")
    if logger.hasHandlers():
        return logger, None

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if log_filename is None:
        log_filename = datetime.now().strftime("run_%Y%m%d_%H%M%S.log")

    log_path = os.path.join(log_dir, log_filename)

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [PID:%(process)d TID:%(thread)d] %(message)s')

    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = ColorizingStreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger, log_path

# ----------- File processing functions ----------
def log_config(config_dict):
    logger = logging.getLogger("LaserDataLogger.config")
    logger.debug("Запуск программы с настройками:")
    for k, v in config_dict.items():
        logger.debug(f"{k}: {v}")
    logger.debug("")