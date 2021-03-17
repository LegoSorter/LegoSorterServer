from concurrent import futures
from typing import List

from lego_sorter_server.classifier.LegoClassifierProvider import LegoClassifierProvider
from lego_sorter_server.detection.DetectionUtils import crop_with_margin
from lego_sorter_server.detection.LegoDetectionRunner import LegoDetectionRunner
from lego_sorter_server.detection.detectors.LegoDetectorProvider import LegoDetectorProvider
from lego_sorter_server.generated import LegoBrick_pb2_grpc
from lego_sorter_server.generated.LegoBrick_pb2 import Image as LegoImage, Empty, ImageStore as LegoImageStore, \
    BoundingBox, \
    ListOfBoundingBoxes

from PIL import Image
from io import BytesIO
import numpy as np
import logging
import time

from lego_sorter_server.detection import DetectionUtils
from lego_sorter_server.images.queue.ImageProcessingQueue import ImageProcessingQueue, SORTER_TAG, CAPTURE_TAG
from lego_sorter_server.images.storage.LegoImageStorage import LegoImageStorage
from lego_sorter_server.sorter.LegoSorterController import LegoSorterController
from lego_sorter_server.sorter.SortingProcessor import SortingProcessor


class LegoBrickService(LegoBrick_pb2_grpc.LegoBrickServicer):
    def __init__(self):
        self.detector = LegoDetectorProvider.get_default_detector()
        self.classifier = LegoClassifierProvider.get_default_classifier()
        self.storage = LegoImageStorage()
        self.processing_queue = ImageProcessingQueue()
        self.executor = futures.ThreadPoolExecutor(max_workers=16)

        self.sorter = SortingProcessor(self.processing_queue, LegoSorterController())
        self.sorter.start_processing()

        self.detection_runner = LegoDetectionRunner(self.processing_queue, self.detector, self.storage)
        self.detection_runner.start_detecting()

    @staticmethod
    def _prepare_image(request: LegoImage) -> Image.Image:
        image = Image.open(BytesIO(request.image))
        image = image.convert('RGB')

        if request.rotation == 90:
            image = image.transpose(Image.ROTATE_270)
        elif request.rotation == 180:
            image = image.rotate(180)
        elif request.rotation == 270:
            image = image.transpose(Image.ROTATE_90)

        return image

    def CollectCroppedImages(self, request: LegoImageStore, context):
        self.executor.submit(self._handle_collect_cropped_images, request)

        return Empty()

    def _handle_collect_cropped_images(self, request: LegoImageStore):
        image = self._prepare_image(request)
        self.processing_queue.add(CAPTURE_TAG, image, request.label)

    def CollectImages(self, request: LegoImageStore, context):
        image = self._prepare_image(request)
        self.storage.save_image(CAPTURE_TAG, image, "unprocessed_" + request.label)

        return Empty()

    def _detect_bricks(self, request: LegoImage) -> List[BoundingBox]:
        image = self._prepare_image(request)
        width, height = image.size
        image_resized, scale = DetectionUtils.resize(image, 640)  # image.resize((640, 640), 0)
        detections = self.detector.detect_lego(np.array(image_resized))

        bbs = []
        for i in range(len(detections['detection_classes'])):
            if detections['detection_scores'][i] < 0.5:
                continue

            bb = BoundingBox()

            bb.ymin, bb.xmin, bb.ymax, bb.xmax = [int(i * 640 * 1 / scale) for i in detections['detection_boxes'][i]]
            if bb.ymax >= height or bb.xmax >= width:
                continue

            bb.score = detections['detection_scores'][i]
            bb.label = 'lego'
            bbs.append(bb)

        return bbs

    def DetectBricks(self, request: LegoImage, context):
        logging.info("[DetectBricks] Request received, processing...")

        start_time = time.time()
        bbs = self._detect_bricks(request)
        elapsed_millis = (time.time() - start_time) * 1000
        logging.info(f"[DetectBricks] Detecting took {elapsed_millis} milliseconds.")
        bb_list = ListOfBoundingBoxes()
        bb_list.packet.extend(bbs)

        logging.info(f"[DetectBricks] {len(bbs)} bricks detected. Returning response.")
        return bb_list

    def DetectAndClassifyBricks(self, request: LegoImage, context):
        logging.info("[DetectAndClassifyBricks] Request received, processing...")
        start_time_detect = time.time()
        bbs = self._detect_bricks(request)
        elapsed_millis_detect = (time.time() - start_time_detect) * 1000
        logging.info(f"[DetectAndClassifyBricks] Detecting took {elapsed_millis_detect} milliseconds.")
        image = self._prepare_image(request)

        bbs_with_blobs = []

        for bb in bbs:
            cropped_brick = crop_with_margin(image, bb.ymin, bb.xmin, bb.ymax, bb.xmax)
            bbs_with_blobs.append((bb, cropped_brick))

        logging.info(f"[DetectAndClassifyBricks] {len(bbs)} bricks detected, classifying...")

        start_time_classify = time.time()
        bbs_labels = self.classifier.predict_from_pil([img for _, img in bbs_with_blobs])
        elapsed_millis_classify = 1000 * (time.time() - start_time_classify)
        logging.info(f"[DetectAndClassifyBricks] Classifying took {elapsed_millis_classify} milliseconds.")
        bbs = [bb for bb, _ in bbs_with_blobs]
        for i in range(0, len(bbs)):
            bbs[i].label = bbs_labels[i]
        bb_list = ListOfBoundingBoxes()
        bb_list.packet.extend(bbs)

        logging.info("[DetectAndClassifyBricks] Returning response")
        return bb_list
