import logging
import numpy

from PIL.Image import Image
from lego_sorter_server.analysis.classification.LegoClassifierProvider import LegoClassifierProvider
from lego_sorter_server.analysis.classification.classifiers.TFLegoClassifier import TFLegoClassifier
from lego_sorter_server.analysis.detection import DetectionUtils
from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults
from lego_sorter_server.analysis.detection.detectors.LegoDetector import LegoDetector
from lego_sorter_server.analysis.detection.detectors.LegoDetectorProvider import LegoDetectorProvider


class AnalysisService:
    DEFAULT_IMAGE_DETECTION_SIZE = (640, 640)

    def __init__(self):
        self.detector: LegoDetector = LegoDetectorProvider.get_default_detector()
        self.classifier: TFLegoClassifier = LegoClassifierProvider.get_default_classifier()

    def detect(self, image: Image, resize: bool = True) -> DetectionResults:
        if image.size is not AnalysisService.DEFAULT_IMAGE_DETECTION_SIZE and resize is False:
            logging.warning(f"[AnalysisService] Requested detection on an image with a non-standard size {image.size} "
                            f"but 'resize' parameter is {resize}.")

        scale = 1

        if image.size is not AnalysisService.DEFAULT_IMAGE_DETECTION_SIZE and resize is True:
            logging.info(f"[AnalysisService] Resizing an image from "
                         f"{image.size} to {AnalysisService.DEFAULT_IMAGE_DETECTION_SIZE}")
            image, scale = DetectionUtils.resize(image, AnalysisService.DEFAULT_IMAGE_DETECTION_SIZE)

        detection_results = self.detector.detect_lego(image)

        return detection_results if scale == 1 else \
            self.translate_bounding_boxes_to_original_size(detection_results,
                                                           scale,
                                                           image.size,
                                                           self.DEFAULT_IMAGE_DETECTION_SIZE[0])

    def classify(self, images: [Image]):
        pass

    def detect_and_classify(self):
        pass

    @staticmethod
    def translate_bounding_boxes_to_original_size(detection_results: DetectionResults,
                                                  scale: float,
                                                  target_image_size: (int, int),  # (width, height)
                                                  detection_image_size: int = 640) -> DetectionResults:
        bbs = []
        for i in range(len(detection_results.detection_classes)):
            y_min, x_min, y_max, x_max = [int(i * detection_image_size * 1 / scale) for i in
                                          detection_results.detection_boxes[i]]

            if y_max >= target_image_size[1] or x_max >= target_image_size[0]:
                continue

            bbs.append((y_min, x_min, y_max, x_max))

        detection_results_translated = DetectionResults(detection_results.detection_scores,
                                                        detection_results.detection_classes,
                                                        bbs)
        return detection_results_translated
