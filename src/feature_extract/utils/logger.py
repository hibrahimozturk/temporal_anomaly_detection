import logging


def create_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = 0
        hd = logging.StreamHandler()
        logger.addHandler(hd)
        hd.setFormatter(logging.Formatter('%(asctime)s - [%(filename)s]:%(funcName)s - %(levelname)s - %(message)s'))
    return logger
