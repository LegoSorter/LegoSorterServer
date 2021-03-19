from concurrent import futures
from typing import List

from lego_sorter_server.analysis.AnalysisService import AnalysisService
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
        image = ImageProtoUtils.prepare_image(request)
        detection_results, classification_results = self.analysis_service.detect_and_classify(image)
        bb_list = ImageProtoUtils.prepare_response_from_analysis_results(detection_results, classification_results)

        logging.info("[DetectAndClassifyBricks] Returning response")

        return bb_list
