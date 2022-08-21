import pyvips
from loguru import logger
import time
from typing import Tuple, List

import numpy

from PIL import Image

from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.classification.LegoClassifierProvider import LegoClassifierProvider
from lego_sorter_server.analysis.classification.classifiers.TFLegoClassifier import TFLegoClassifier
from lego_sorter_server.analysis.detection import DetectionUtils
from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults
from lego_sorter_server.analysis.detection.detectors.LegoDetector import LegoDetector
from lego_sorter_server.analysis.detection.detectors.LegoDetectorProvider import LegoDetectorProvider


class AnalysisFastService:
    DEFAULT_IMAGE_DETECTION_SIZE = (640, 640)
    BORDER_MARGIN_RELATIVE = 0.001

    def __init__(self):
        self.detector: LegoDetector = LegoDetectorProvider.get_default_detector()
        self.classifier = LegoClassifierProvider.get_default_classifier()

    def detect(self, image: pyvips.Image, resize: bool = True, threshold=0.5,
    # def detect(self, image: Image.Image, resize: bool = True, threshold=0.5,
               discard_border_results: bool = True) -> DetectionResults:
        start_time = time.time()
        # print(image.size)
        # if image.size is not AnalysisFastService.DEFAULT_IMAGE_DETECTION_SIZE and resize is False:
        #     logger.warning(f"[AnalysisFastService][detect] Requested detection on an image with a non-standard size {image.size} "
        #                     f"but 'resize' parameter is {resize}.")

        # print((image.width, image.height))
        if (image.width, image.height) is not AnalysisFastService.DEFAULT_IMAGE_DETECTION_SIZE and resize is False:
            logger.warning(f"[AnalysisFastService][detect] Requested detection on an image with a non-standard size {(image.width, image.height)} "
                            f"but 'resize' parameter is {resize}.")

        scale = 1
        # original_size = image.size
        # if image.size is not AnalysisFastService.DEFAULT_IMAGE_DETECTION_SIZE and resize is True:
        #     logger.info(f"[AnalysisFastService][detect] Resizing an image from "
        #                  f"{image.size} to {AnalysisFastService.DEFAULT_IMAGE_DETECTION_SIZE}")
        #     image, scale = DetectionUtils.resize(image, AnalysisFastService.DEFAULT_IMAGE_DETECTION_SIZE[0])
        #     # image.show()

        original_size = (image.width, image.height)
        if (image.height, image.width) is not AnalysisFastService.DEFAULT_IMAGE_DETECTION_SIZE and resize is True:
            logger.info(f"[AnalysisFastService][detect] Resizing an image from "
                         f"{(image.height, image.width)} to {AnalysisFastService.DEFAULT_IMAGE_DETECTION_SIZE}")
            image, scale = DetectionUtils.resizeVips(image, AnalysisFastService.DEFAULT_IMAGE_DETECTION_SIZE[0])
            # im = Image.fromarray(image.numpy())
            # im.show()

        if discard_border_results:
            # accepted_xy_range = [(original_size[0] * scale) / image.size[0], (original_size[1] * scale) / image.size[1]]
            accepted_xy_range = [(original_size[0] * scale) / image.height, (original_size[1] * scale) / image.width]
        else:
            accepted_xy_range = [1, 1]
        resize_time = 1000 * (time.time() - start_time)
        numpy_image = numpy.array(image)
        numpy_image_time = 1000 * (time.time() - start_time) - resize_time
        detection_results = self.detector.detect_lego(numpy_image)
        after_detect = 1000 * (time.time() - start_time) - resize_time - numpy_image_time
        detection_results = self.filter_detection_results(detection_results, threshold, accepted_xy_range)
        translated = self.translate_bounding_boxes_to_original_size(detection_results, scale, original_size,
                                                                    self.DEFAULT_IMAGE_DETECTION_SIZE[0])

        all = 1000 * (time.time() - start_time)
        logger.debug(f"[AnalysisFastService][detect] Resizing and detection took {all} ms, resize {resize_time} ms, "
                     f"numpy {numpy_image_time}ms, detection {after_detect}ms.")
        return translated

    def classify(self, images: List[pyvips.Image]) -> ClassificationResults:
    # def classify(self, images: List[Image.Image]) -> ClassificationResults:
        return self.classifier.predict(images)

    # def detect_and_classify(self, image: Image.Image, detection_threshold: float = 0.5, discard_border_results: bool = True) \
    def detect_and_classify(self, image: pyvips.Image, detection_threshold: float = 0.5, discard_border_results: bool = True) \
            -> Tuple[DetectionResults, ClassificationResults]:
        start_time = time.time()
        detection_results = self.detect(image, threshold=detection_threshold,
                                        discard_border_results=discard_border_results)
        detect = 1000 * (time.time() - start_time)
        cropped_images = []
        for bounding_box in detection_results.detection_boxes:
            # cropped_image = DetectionUtils.crop_with_margin_from_bb(image, bounding_box)
            cropped_image = DetectionUtils.crop_with_margin_from_bb_vips(image, bounding_box)
            cropped_images.append(cropped_image)
        classification_results = self.classify(cropped_images)
        classification = 1000 * (time.time() - start_time) - detect
        elapsed_millis = 1000 * (time.time() - start_time)
        logger.debug(f"[AnalysisFastService][detect_and_classify] Detecting and classifying took {elapsed_millis} ms, "
                     f"detection {detect} ms, classification {classification} ms.")
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

            # if y_max >= target_image_size[1] or x_max >= target_image_size[0]:
            #     continue
            y_max = min(y_max, target_image_size[1])
            x_max = min(x_max, target_image_size[0])

            bbs.append((y_min, x_min, y_max, x_max))

        detection_results_translated = DetectionResults(detection_results.detection_scores,
                                                        detection_results.detection_classes,
                                                        bbs)
        return detection_results_translated

    def filter_detection_results(self, detection_results, threshold, accepted_xy_range):
        limit = len(detection_results.detection_scores)
        for index, score in enumerate(detection_results.detection_scores):
            if score < threshold:
                limit = index
                break

        filtered_results = DetectionResults(detection_results.detection_scores[:limit],
                                            detection_results.detection_classes[:limit],
                                            detection_results.detection_boxes[:limit])
        if accepted_xy_range != [1, 1]:
            results = []
            for score, clazz, box in zip(filtered_results.detection_scores,
                                         filtered_results.detection_classes,
                                         detection_results.detection_boxes):
                # (ymin, xmin, ymax, xmax)
                if box[0] < self.BORDER_MARGIN_RELATIVE \
                        or box[1] < self.BORDER_MARGIN_RELATIVE \
                        or box[2] > accepted_xy_range[1] - self.BORDER_MARGIN_RELATIVE \
                        or box[3] > accepted_xy_range[0] - self.BORDER_MARGIN_RELATIVE:
                    continue
                results.append([score, clazz, box])

            if len(results) != 0:
                results = numpy.stack(results)
                filtered_results = DetectionResults(results[:, 0], results[:, 1], results[:, 2])
            else:
                filtered_results = DetectionResults([], [], [])

        return filtered_results
