import argparse

from lego_sorter_server.server import Server
import logging
import sys
import threading

from lego_sorter_server.service.BrickCategoryConfig import BrickCategoryConfig


def exception_handler(exc_type, value, tb):
    logging.exception(f"Uncaught exception: {str(value)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--brick_category_config", "-c", help='.json file with brick-category mapping specification', type=str, required=False)
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    sys.excepthook = exception_handler
    threading.excepthook = exception_handler
    Server.run(BrickCategoryConfig(args.brick_category_config))
