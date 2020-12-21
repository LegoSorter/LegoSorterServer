import random
import string
import time
import logging
from concurrent import futures

import numpy as np

from lego_sorter_server.detection import DetectionUtils
from lego_sorter_server.detection.LegoDetector import LegoDetector
from lego_sorter_server.images.queue.ImageProcessingQueue import ImageProcessingQueue
from lego_sorter_server.images.storage.LegoImageStorage import LegoImageStorage


class LegoDetectionRunner:
    def __init__(self, queue: ImageProcessingQueue, detector: LegoDetector, store: LegoImageStorage):
        self.queue = queue
        self.detector = detector
        self.storage = store
        # queue, detector and storage are not thread safe, so it limits the number of workers to one
        self.executor = futures.ThreadPoolExecutor(max_workers=1)
        logging.info("[LegoDetectionRunner] Initialized\n")

    def start_detecting(self):
        logging.info("[LegoDetectionRunner] Started processing the queue\n")
        self.executor.submit(self._process_queue)

    def stop_detecting(self):
        logging.info("[LegoDetectionRunner] Processing is being terminated\n")
        self.executor.shutdown()

    def _process_queue(self):
        polling_rate = 0.2  # in seconds
        logging_rate = 5  # in seconds
        logging_counter = 0
        while True:
            if self.queue.len() == 0:
                if polling_rate * logging_counter >= logging_rate:
                    logging.info("Queue is empty. Waiting... ")
                    logging_counter = 0
                else:
                    logging_counter += 1
                time.sleep(polling_rate)
                continue
            logging.info("Queue not empty - processing data")
            image, lego_class = self.queue.next()
            prefix = self._get_random_hash() + "_"

            width, height = image.size
            image_resized, scale = DetectionUtils.resize(image, 640)
            detections = self.detector.detect_lego(np.array(image_resized))
            abs_margin = 0  # in pixels
            rel_margin = 0.10
            detected_counter = 0
            for i in range(100):
                if detections['detection_scores'][i] < 0.5:
                    break  # IF SORTED

                detected_counter += 1
                ymin, xmin, ymax, xmax = [int(i * 640 * 1 / scale) for i in detections['detection_boxes'][i]]

                # if bb is out of bounds
                if ymax >= height or xmax >= width:
                    continue

                # Apply margins
                avg_length = ((xmax - xmin) + (ymax - ymin)) / 2
                ymin = max(ymin - abs_margin - rel_margin * avg_length, 0)
                xmin = max(xmin - abs_margin - rel_margin * avg_length, 0)
                ymax = min(ymax + abs_margin + rel_margin * avg_length, height)
                xmax = min(xmax + abs_margin + rel_margin * avg_length, width)

                image_new = image.crop([xmin, ymin, xmax, ymax])

                self.storage.save_image(image_new, lego_class, prefix)

            prefix = f'{detected_counter}_{prefix}'
            self.storage.save_image(image, 'original', prefix)

    @staticmethod
    def _get_random_hash(length=4):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
