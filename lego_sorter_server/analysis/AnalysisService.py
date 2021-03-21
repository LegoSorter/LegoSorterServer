import logging
from typing import Tuple, List

import numpy

from PIL.Image import Image

from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
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
        original_size = image.size
        if image.size is not AnalysisService.DEFAULT_IMAGE_DETECTION_SIZE and resize is True:
            logging.info(f"[AnalysisService] Resizing an image from "
                         f"{image.size} to {AnalysisService.DEFAULT_IMAGE_DETECTION_SIZE}")
            image, scale = DetectionUtils.resize(image, AnalysisService.DEFAULT_IMAGE_DETECTION_SIZE[0])

        detection_results = self.detector.detect_lego(numpy.array(image))

        return self.translate_bounding_boxes_to_original_size(detection_results,
                                                              scale,
                                                              original_size,
                                                              self.DEFAULT_IMAGE_DETECTION_SIZE[0])

    def classify(self, images: List[Image]) -> ClassificationResults:
        return self.classifier.predict_from_pil(images)

    def detect_and_classify(self, image: Image) -> Tuple[DetectionResults, ClassificationResults]:
        detection_results = self.detect(image)

        cropped_images = []
        for bounding_box in detection_results.detection_boxes:
            cropped_image = DetectionUtils.crop_with_margin_from_bb(image, bounding_box)
            cropped_images.append(cropped_image)

        classification_results = self.classify(cropped_images)

        return detection_results, classification_results

    @staticmethod
    def translate_bounding_boxes_to_original_size(detection_results: DetectionResults,
                                                  scale: float,
                                                  target_image_size: Tuple[int, int],  # (width, height)
                                                  detection_image_size: int = 640) -> DetectionResults:
        bbs = []
        for i in range(len(detection_results.detection_classes)):
            y_min, x_min, y_max, x_max = [int(coord * detection_image_size * 1 / scale) for coord in
                                          detection_results.detection_boxes[i]]

            if y_max >= target_image_size[1] or x_max >= target_image_size[0]:
                continue

            bbs.append((y_min, x_min, y_max, x_max))

        detection_results_translated = DetectionResults(detection_results.detection_scores,
                                                        detection_results.detection_classes,
                                                        bbs)
        return detection_results_translated
