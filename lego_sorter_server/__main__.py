from lego_sorter_server.images.queue import ImageProcessingQueue
from lego_sorter_server.server import Server
from pathlib import Path
from PIL import Image

import time
import logging
import sys
import threading

from lego_sorter_server.service.LegoCaptureService import LegoCaptureService


def exception_handler(exc_type, value, tb):
    logging.exception(f"Uncaught exception: {str(value)}")


IMAGES_DIRECTORY = "/backup/RENDER_2/original/"


def process_images_from_directory():
    capture_service = LegoCaptureService()

    for part_dir in Path(IMAGES_DIRECTORY).iterdir():
        label = part_dir.name
        print(f"Processing {label} lego class")

        for image_path in part_dir.glob("*"):
            print(f"Processing {image_path}")
            image = Image.open(image_path)
            while capture_service.processing_queue.is_full(ImageProcessingQueue.CAPTURE_TAG):
                logging.info("Queue is full, waiting...")
                time.sleep(1)
            capture_service.processing_queue.add(ImageProcessingQueue.CAPTURE_TAG, image, label)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # sys.excepthook = exception_handler
    # threading.excepthook = exception_handler
    # Server.run()
    process_images_from_directory()
