import logging
import os
import sys

class Adapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if 'extra' not in kwargs:
            kwargs["extra"] = self.extra
        return msg, kwargs

def configure_logger():
    logger = logging.getLogger("MAIN")

    # Set logging level
    logger.setLevel(logging.INFO)

    # Longging format
    fmt = logging.Formatter(f"%(asctime)-15s %(type)-15s - %(message)s")

    # Logging Handlers - console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    logger = Adapter(logger, { "type": "INFO" })
    return logger
