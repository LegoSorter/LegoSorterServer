from concurrent import futures
from typing import List

from lego_sorter_server.analysis.AnalysisService import AnalysisService
from lego_sorter_server.analysis.classification.LegoClassifierProvider import LegoClassifierProvider
from lego_sorter_server.analysis.detection.LegoDetectionRunner import LegoDetectionRunner
from lego_sorter_server.generated import LegoBrick_pb2_grpc
from lego_sorter_server.generated.LegoBrick_pb2 import ImageRequest, Empty, ImageStore as LegoImageStore, \
    BoundingBox, \
    ListOfBoundingBoxes

import logging
import time

from lego_sorter_server.images.queue.ImageProcessingQueue import ImageProcessingQueue, CAPTURE_TAG
from lego_sorter_server.images.storage.LegoImageStorage import LegoImageStorage
from lego_sorter_server.service.ImageProtoUtils import ImageProtoUtils


class LegoBrickService(LegoBrick_pb2_grpc.LegoBrickServicer):
    def __init__(self):
        self.analysis_service = AnalysisService()
        self.classifier = LegoClassifierProvider.get_default_classifier()
        self.storage = LegoImageStorage()
        self.executor = futures.ThreadPoolExecutor(max_workers=16)
        self.processing_queue = ImageProcessingQueue()
        self.detection_runner = LegoDetectionRunner(self.processing_queue, self.storage)
        self.detection_runner.start_detecting()

    def CollectCroppedImages(self, request: LegoImageStore, context):
        self.executor.submit(self._handle_collect_cropped_images, request)

        return Empty()

    def _handle_collect_cropped_images(self, request: LegoImageStore):
        image = ImageProtoUtils.prepare_image(request)
        self.processing_queue.add(CAPTURE_TAG, image, request.label)

    def CollectImages(self, request: LegoImageStore, context):
        image = ImageProtoUtils.prepare_image(request)
        self.storage.save_image(CAPTURE_TAG, image, "unprocessed_" + request.label)

        return Empty()

    def _detect_bricks(self, request: ImageRequest) -> List[BoundingBox]:
        image = ImageProtoUtils.prepare_image(request)
        detection_results = self.analysis_service.detect(image)
        bounding_boxes = ImageProtoUtils.prepare_bbs_response_from_detection_results(detection_results)

        return bounding_boxes

    def DetectBricks(self, request: ImageRequest, context):
        logging.info("[DetectBricks] Request received, processing...")

        start_time = time.time()
        bbs = self._detect_bricks(request)
        elapsed_millis = (time.time() - start_time) * 1000
        logging.info(f"[DetectBricks] Detecting took {elapsed_millis} milliseconds.")
        bb_list = ListOfBoundingBoxes()
        bb_list.packet.extend(bbs)

        logging.info(f"[DetectBricks] {len(bbs)} bricks detected. Returning response.")
        return bb_list

    def DetectAndClassifyBricks(self, request: ImageRequest, context):
        logging.info("[DetectAndClassifyBricks] Request received, processing...")
        start_time_detect = time.time()

        bbs = self._detect_bricks(request)

        elapsed_millis_detect = (time.time() - start_time_detect) * 1000
        logging.info(f"[DetectAndClassifyBricks] Detecting took {elapsed_millis_detect} milliseconds.")

        image = ImageProtoUtils.prepare_image(request)
        bbs_with_blobs = ImageProtoUtils.crop_bounding_boxes(image, bbs)

        logging.info(f"[DetectAndClassifyBricks] {len(bbs)} bricks detected, classifying...")
        start_time_classify = time.time()

        classification_results = self.classifier.predict_from_pil([img for _, img in bbs_with_blobs])

        elapsed_millis_classify = 1000 * (time.time() - start_time_classify)
        logging.info(f"[DetectAndClassifyBricks] Classifying took {elapsed_millis_classify} milliseconds.")

        bb_list = ImageProtoUtils.prepare_response_from_bbs_and_labels(bbs, classification_results)

        logging.info("[DetectAndClassifyBricks] Returning response")

        return bb_list
