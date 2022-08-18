#ukrywanie gpu
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# print("wył")

import argparse
import os

import numpy as np
import uvicorn
import logging

from lego_sorter_server.api.ServerApi import app
from lego_sorter_server.database import Models
from lego_sorter_server.database.Database import SessionLocal

from loguru import logger
import sys
import threading
import warnings


brickCategoryConfig = None


LOG_LEVEL = logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO"))
# LOG_LEVEL = logging.getLevelName("INFO")
JSON_LOGS = True if os.environ.get("JSON_LOGS", "0") == "1" else False


def exception_handler(exc_type, value, tb):
    logger.exception(f"Uncaught exception: {str(value)}")


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging():
    # intercept everything at the root logger
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(LOG_LEVEL)
    logging.getLogger().setLevel(logging.INFO)

    # remove every other logger's handlers
    # and propagate to root logger
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

    # configure loguru
    logger.configure(handlers=[{"sink": sys.stdout, "level": LOG_LEVEL, "format":"<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>", "serialize": JSON_LOGS}])
    # logger.configure(handlers=[{"sink": sys.stdout, "serialize": JSON_LOGS}])


if __name__ == '__main__':
    sys.excepthook = exception_handler
    threading.excepthook = exception_handler
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    db = SessionLocal()
    server_fastapi_port = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "server_fastapi_port").one_or_none()
    if server_fastapi_port is None:
        server_fastapi_port = Models.DBConfiguration(option="server_fastapi_port", value="5005")
        db.add(server_fastapi_port)
        db.commit()
        db.refresh(server_fastapi_port)
    db.close()
    # setup_logging()
    # uvicorn.run(app, host="0.0.0.0", port=int(server_fastapi_port.value), log_level="info")

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="0.0.0.0",
            port=int(server_fastapi_port.value),
            log_level=LOG_LEVEL,
        ),
    )
    setup_logging()
    server.run()
    # Server.run(brickCategoryConfig)

