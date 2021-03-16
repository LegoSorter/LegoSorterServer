import time
from concurrent import futures

from PIL import Image
from object_detection.utils import visualization_utils as viz_utils

import numpy

from lego_sorter_server.classifier import LegoClassifierProvider
from lego_sorter_server.detection import DetectionUtils
from lego_sorter_server.detection.detectors import LegoDetectorProvider
from lego_sorter_server.images.queue.ImageProcessingQueue import ImageProcessingQueue, SORTER_TAG
from lego_sorter_server.sorter.LegoSorterController import LegoSorterController

import logging

POLLING_RATE = 0.1  # seconds


class SortingProcessor:
    def __init__(self, queue: ImageProcessingQueue, sorter_controller: LegoSorterController):
        self.queue = queue
        self.sorter_controller = sorter_controller
        self.detector = LegoDetectorProvider.get_default_detector()
        self.classifier = LegoClassifierProvider.get_default_classifier()
        self.executor = futures.ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def find_closest_brick(detections):
        bbs = detections["detection_boxes"]
        # The closest brick is a brick with the smallest ymin value
        closest = min(bbs, key=lambda bb: bb[0])
        return closest

    @staticmethod
    def get_best_result(results):
        # TODO - max score, average score, max count?
        pass

    @staticmethod
    def is_following_position(previous_position_ymin, current_position_ymin):
        return previous_position_ymin > current_position_ymin

    def start_processing(self):
        logging.info(f"[SortingProcessor] Started processing the queue")
        logging.info(f"[SortingProcessor] Cleaning the queue...")

        self.queue.clear(SORTER_TAG)
        self.executor.submit(self.__run)

    def stop_processing(self):
        logging.info(f"[SortingProcessor] Stopping processing the queue...")
        self.executor.shutdown()

    def __run(self):
        results_for_last_brick = []
        while True:
            if self.queue.len(SORTER_TAG) == 0:
                time.sleep(POLLING_RATE)
                continue

            current = self._process_next_image()

            if len(current) == 0 and len(results_for_last_brick) == 0:
                logging.info(f"[SortingProcessor] Got an empty image.")
                continue

            if len(current) == 0:
                # Nothing more to check, we can return results to machine
                logging.info(f"[SortingProcessor] The brick surpassed the camera line, sending results.")
                self._send_result_and_clear_history(results_for_last_brick)
                continue

            if len(results_for_last_brick) > 0:
                # We are probably processing still the same brick, checking
                previous_position, _ = results_for_last_brick[-1]
                current_position, _ = current

                if not self.is_following_position(previous_position[0], current_position[0]):
                    self._send_result_and_clear_history(results_for_last_brick)

            logging.info(f"[SortingProcessor] Saving results for further analysis...")
            results_for_last_brick.append(current)

        logging.info(f"[SortingProcessor] Processing stopped.")

    def _send_result_and_clear_history(self, results_for_last_brick):
        best_result = self.get_best_result(results_for_last_brick)
        logging.info(f"[SortingProcessor] Got the best result {best_result}. Returning the results...")
        self.sorter_controller.on_brick_recognized(best_result)
        results_for_last_brick.clear()

    def _process_next_image(self):
        captured_image, _ = self.queue.next(SORTER_TAG)

        # TODO - resize image!
        image_resized, scale = DetectionUtils.resize(captured_image, 640)
        detections = self.detector.detect_lego(numpy.array(image_resized))
        # for i in range(len(detections["detection_classes"])):
        #     ymin, xmin, ymax, xmax = [int(i * 640 * 1 / scale) for i in detections['detection_boxes'][i]]
        #     detections['detection_boxes'][i] = [ymin, xmin, ymax, xmax]

        detected_count = len(detections["detection_classes"])
        if detected_count == 0:
            return ()

        logging.info(f"[SortingProcessor] Detected a lego brick, processing...")

        if detected_count > 1:
            logging.warning(f"[SortingProcessor] More than one brick detected '(detected_count = {detected_count}), "
                            f"there should be only one brick on the tape at the same time.")

        # TODO - we can just run inference for all detected bricks rather than only the closest one and then store the
        #  results for future use.
        target_brick = self.find_closest_brick(detections)
        results = self.classifier.classify(target_brick)

        return target_brick, results
