from lego_sorter_server.server import Server
import logging
import sys
import threading


def exception_handler(exc_type, value, tb):
    logging.exception(f"Uncaught exception: {str(value)}")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    sys.excepthook = exception_handler
    threading.excepthook = exception_handler
    Server.run()
