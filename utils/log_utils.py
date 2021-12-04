import logging


def get_logger():
    logger = logging.getLogger("adgrf")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    return logger
