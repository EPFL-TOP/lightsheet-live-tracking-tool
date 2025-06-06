import logging

def init_logger(name) :
    logger = logging.getLogger(name)
    # Set default logger configuration
    if not logger.hasHandlers():
        # console handler if no global config is provided
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger