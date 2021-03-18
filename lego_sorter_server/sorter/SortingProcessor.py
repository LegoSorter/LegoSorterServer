import numpy
import logging

from concurrent import futures

from PIL.Image import Image

from lego_sorter_server.classifier.LegoClassifierProvider import LegoClassifierProvider
from lego_sorter_server.detection import DetectionUtils
from lego_sorter_server.detection.detectors.LegoDetectorProvider import LegoDetectorProvider
from lego_sorter_server.sorter.LegoSorterController import LegoSorterController


class SortingProcessor:
    def __init__(self, sorter_controller: LegoSorterController):
        self.sorter_controller = sorter_controller
        self.detector = LegoDetectorProvider.get_default_detector()
        self.classifier = LegoClassifierProvider.get_default_classifier()
        self.executor = futures.ThreadPoolExecutor(max_workers=1)
        self.last_results = []

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

    def process_next_image(self, image: Image):
        current_result = self._process(image)

        if len(current_result) == 0 and len(self.last_results) == 0:
            logging.info(f"[SortingProcessor] Got an empty image.")
            return current_result

        if len(current_result) == 0:
            # Nothing more to check, we can return results to machine
            logging.info(f"[SortingProcessor] The brick surpassed the camera line, sending results.")
            self._send_result_and_clear_history()
            return current_result

        if len(self.last_results) > 0:
            # We are probably processing still the same brick, checking
            previous_position, _ = self.last_results[-1]
            current_position, _ = current_result

            if not self.is_following_position(previous_position[0], current_position[0]):
                self._send_result_and_clear_history()

        logging.info(f"[SortingProcessor] Saving results for further analysis...")
        self.last_results.append(current_result)

    def _send_result_and_clear_history(self):
        best_result = self.get_best_result(self.last_results)
        logging.info(f"[SortingProcessor] Got the best result {best_result}. Returning the results...")
        self.sorter_controller.on_brick_recognized(best_result)
        self.last_results.clear()

    def _process(self, image: Image):
        # TODO - resize image!
        image_resized, scale = DetectionUtils.resize(image, 640)
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
