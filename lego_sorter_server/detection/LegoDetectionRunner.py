import random
import string
import time
import logging
from concurrent import futures

import numpy as np

from lego_sorter_server.detection import DetectionUtils
from lego_sorter_server.detection.DetectionUtils import crop_with_margin
from lego_sorter_server.detection.detectors.LegoDetector import LegoDetector
from lego_sorter_server.detection.LegoLabeler import LegoLabeler
from lego_sorter_server.images.queue.ImageProcessingQueue import ImageProcessingQueue, CAPTURE_TAG
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

    def _process_queue(self, save_cropped_image=True, save_label_file=False):
        polling_rate = 0.2  # in seconds
        logging_rate = 5  # in seconds
        logging_counter = 0
        while True:
            if self.queue.len(CAPTURE_TAG) == 0:
                if polling_rate * logging_counter >= logging_rate:
                    logging.info("Queue is empty. Waiting... ")
                    logging_counter = 0
                else:
                    logging_counter += 1
                time.sleep(polling_rate)
                continue
            logging.info("Queue not empty - processing data")
            image, lego_class = self.queue.next(CAPTURE_TAG)
            prefix = self._get_random_hash() + "_"

            width, height = image.size
            image_resized, scale = DetectionUtils.resize(image, 640)
            detections = self.detector.detect_lego(np.array(image_resized))

            detected_counter = 0
            bbs = []
            for i in range(len(detections['detection_classes'])):
                if detections['detection_scores'][i] < 0.5:
                    break  # IF SORTED

                detected_counter += 1
                ymin, xmin, ymax, xmax = [int(i * 640 * 1 / scale) for i in detections['detection_boxes'][i]]

                # # if bb is out of bounds
                # if ymax >= height or xmax >= width:
                #     continue

                bbs.append((xmin, ymin, xmax, ymax))

                if save_cropped_image is True:
                    image_new = crop_with_margin(image, ymin, xmin, ymax, xmax)
                    self.storage.save_image(image_new, lego_class, prefix)

            prefix = f'{detected_counter}_{prefix}'
            filename = self.storage.save_image(image, 'original', prefix)

            if save_label_file is True:
                path = self.storage.find_image_path(filename)
                label_file = LegoLabeler().to_label_file(filename, str(path), width, height, bbs)
                xml_path = path.parent.absolute() / (filename.split(".")[-2] + ".xml")
                with open(xml_path, "w") as label_xml:
                    label_xml.write(label_file)

    @staticmethod
    def _get_random_hash(length=4):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
