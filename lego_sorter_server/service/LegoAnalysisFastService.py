import io
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from lego_sorter_server.analysis.AnalysisFastService import AnalysisFastService
import logging
import time

from lego_sorter_server.analysis.LegoAnalysisFastRunner import LegoAnalysisFastRunner
from lego_sorter_server.analysis.LegoAnnotationFastRunner import LegoAnnotationFastRunner
from lego_sorter_server.analysis.LegoStorageFastRunner import LegoStorageFastRunner
from lego_sorter_server.generated import LegoAnalysisFast_pb2_grpc
from lego_sorter_server.generated.LegoAnalysisFast_pb2 import FastImageRequest
from lego_sorter_server.generated.Messages_pb2 import ImageRequest, ListOfBoundingBoxes
from lego_sorter_server.images.queue.ImageAnnotationQueueFast import ImageAnnotationQueueFast
from lego_sorter_server.images.queue.ImageProcessingQueueFast import ImageProcessingQueueFast, CAPTURE_TAG
from lego_sorter_server.images.queue.ImageStorageQueueFast import ImageStorageQueueFast
from lego_sorter_server.images.storage.LegoImageStorageFast import LegoImageStorageFast
from lego_sorter_server.service.ImageProtoUtils import ImageProtoUtils

import logging
import sys
from signalrcore.hub_connection_builder import HubConnectionBuilder


class MyMessage:
    def __init__(self, ymin, xmin, ymax, xmax, label, score):
        self.ymin = ymin
        self.xmin = xmin
        self.ymax = ymax
        self.xmax = xmax
        self.label = label
        self.score = score


class LegoAnalysisFastService(LegoAnalysisFast_pb2_grpc.LegoAnalysisFastServicer):

    def __init__(self, hub_connection: HubConnectionBuilder, lastImages: deque, storageFastRunerExecutor: ThreadPoolExecutor, analyzerFastRunerExecutor: ThreadPoolExecutor, annotationFastRunerExecutor: ThreadPoolExecutor):
        self.storage = LegoImageStorageFast()
        # self.lastImages = lastImages
        self.processing_queue = ImageProcessingQueueFast()
        self.storage_queue = ImageStorageQueueFast()
        self.storage_runner = LegoStorageFastRunner(self.storage_queue, self.processing_queue, self.storage, storageFastRunerExecutor, lastImages)
        self.storage_runner.start_detecting()
        self.annotation_queue = ImageAnnotationQueueFast()
        self.annotation_runner = LegoAnnotationFastRunner(self.annotation_queue, annotationFastRunerExecutor)
        self.annotation_runner.start_detecting()
        self.detection_runner = LegoAnalysisFastRunner(self.processing_queue, hub_connection, analyzerFastRunerExecutor, self.annotation_queue)
        self.detection_runner.start_detecting()
        self.analysis_service = AnalysisFastService()
        self.handler = logging.StreamHandler()
        self.handler.setLevel(logging.DEBUG)

        bounding_boxes = []
        self.empty_bb_list = ListOfBoundingBoxes()
        self.empty_bb_list.packet.extend(bounding_boxes)
        logging.info("[LegoAnalysisFastService] started")
        # .configure_logging(logging.DEBUG, socket_trace=True, handler=self.handler) \
        # .with_url("http://localhost:5002/hubs/sorter", options={"verify_ssl": False}) \
        # self.hub_connection = hub_connection
        # self.hub_connection = HubConnectionBuilder()\
        #     .with_url("http://localhost:80/hubs/sorter", options={"verify_ssl": False}) \
        #     .with_automatic_reconnect({
        #             "type": "interval",
        #             "keep_alive_interval": 10,
        #             "intervals": [1, 3, 5, 6, 7, 87, 3]
        #         }).build()
        # self.hub_connection.on_open(lambda: print("connection opened and handshake received ready to send messages"))
        # self.hub_connection.on_close(lambda: print("connection closed"))
        # self.hub_connection.on("messageReceived", print)
        # self.hub_connection.start()

    def DetectBricks(self, request: ImageRequest, context):
        logging.info("[LegoAnalysisFastService] Request received, processing...")
        start_time = time.time()

        detection_results = self._detect_bricks(request)
        bbs_list = ImageProtoUtils.prepare_bbs_response_from_detection_results(detection_results)

        elapsed_millis = int((time.time() - start_time) * 1000)
        logging.info(f"[LegoAnalysisFastService] Detecting and preparing response took {elapsed_millis} milliseconds.")
        self.hub_connection.send("SendMessage", [bbs_list[0]])

        return bbs_list


    cnt = 0;


    def DetectAndClassifyBricks(self, request: FastImageRequest, context):
        # print(f"{self.cnt};{time.time()}")
        self.cnt += 1
        self.storage_queue.add(CAPTURE_TAG, request, request.session)

        # logging.info("[LegoAnalysisFastService] Request received, added to queue...")
        logging.info(f"[LegoAnalysisFastService] Request received, added to queue {self.cnt};{time.time()}")
        return self.empty_bb_list

    def _detect_bricks(self, request: ImageRequest) -> ListOfBoundingBoxes:
        image = ImageProtoUtils.prepare_image(request)
        detection_results = self.analysis_service.detect(image)

        return detection_results
