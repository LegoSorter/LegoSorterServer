import os
import random
import string
import time
from loguru import logger
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import Event

from signalrcore.hub_connection_builder import HubConnectionBuilder
from uuid6 import uuid7

from lego_sorter_server.analysis.detection.DetectionUtils import crop_with_margin
from lego_sorter_server.analysis.detection.LegoLabeler import LegoLabeler
from lego_sorter_server.database import Models
from lego_sorter_server.database.Database import SessionLocal, get_or_create
from lego_sorter_server.generated.Messages_pb2 import ListOfBoundingBoxes
from lego_sorter_server.images.queue.ImageAnnotationQueueFast import ImageAnnotationQueueFast
from lego_sorter_server.images.queue.ImageCropQueueFast import ImageCropQueueFast
from lego_sorter_server.images.queue.ImageSortQueueFast import ImageSortQueueFast
from lego_sorter_server.images.queue.ImageStorageQueueFast import ImageStorageQueueFast
from lego_sorter_server.images.queue.ImageProcessingQueueFast import ImageProcessingQueueFast, CAPTURE_TAG
from lego_sorter_server.images.storage.LegoImageStorageFast import LegoImageStorageFast
from lego_sorter_server.service.ImageProtoUtils import ImageProtoUtils
from lego_sorter_server.database.Models import *


class QueueInfoMessage:
    def __init__(self, last_images_length, storage_queue_length, processing_length, annotation_length, sort_length, crops_length):
        self.lastImages_length = last_images_length
        self.storage_queue_length = storage_queue_length
        self.processing_length = processing_length
        self.annotation_length = annotation_length
        self.sort_length = sort_length
        self.crops_length = crops_length


class LegoQueueInfoFastRunner:
    DETECTION_SCORE_THRESHOLD = 0.5

    def __init__(self, hub_connection: HubConnectionBuilder, storage_queue: ImageStorageQueueFast, processing_queue: ImageProcessingQueueFast,
                 sort_queue: ImageSortQueueFast, annotation_queue: ImageAnnotationQueueFast, crops_queue: ImageCropQueueFast, lastImages: deque, queue_info_fast_runer_executor: ThreadPoolExecutor, event: Event):
        self.hub_connection = hub_connection
        self.event = event
        self.storage_queue = storage_queue
        self.processing_queue = processing_queue
        self.sort_queue = sort_queue
        self.annotation_queue = annotation_queue
        self.crops_queue = crops_queue
        self.lastImages = lastImages
        # self.storage = store
        # queue, detector and storage are not thread safe, so it limits the number of workers to one
        self.executor = queue_info_fast_runer_executor
        self.hub_connection.on("requestQueuesInfoMessage", lambda _: self.send_queue_info_message())
        # self.executor = futures.ThreadPoolExecutor(max_workers=4)
        logger.info("[LegoQueueInfoRunner] Ready for send queues length.")

    def start_detecting(self):
        logger.info("[LegoQueueInfoRunner] Started sending queues length.")
        return self.executor.submit(self._exception_handler, self._process_queue)

    def stop_detecting(self):
        logger.info("[LegoQueueInfoRunner] Processing is being terminated.")
        self.event.set()
        self.executor.shutdown()

    @staticmethod
    def _exception_handler(method, args=[]):
        try:
            method(*args)
        except Exception as exc:
            logger.exception(f"[LegoQueueInfoRunner] Got an exception:\n {str(exc)}")
            raise exc

    def _process_queue(self):
        polling_rate = 0.2  # in seconds
        logging_rate = 30  # in seconds
        logging_counter = 0
        limit = 2
        futures = set()
        last_queue_info_message = self.get_queue_info_message()
        no_change_cnt = 0
        while True:
            if self.event.isSet():
                logger.info("[LegoQueueInfoRunner] Processing queue is being terminated.")
                return
            queueInfoMessage = self.get_queue_info_message()
            if (
                    queueInfoMessage.storage_queue_length != last_queue_info_message.storage_queue_length
                    or queueInfoMessage.processing_length != last_queue_info_message.processing_length
                    or queueInfoMessage.sort_length != last_queue_info_message.sort_length
                    or queueInfoMessage.annotation_length != last_queue_info_message.annotation_length
                    or queueInfoMessage.lastImages_length != last_queue_info_message.lastImages_length
                    or queueInfoMessage.crops_length != last_queue_info_message.crops_length
            ):
                self.send_queue_info_message(queueInfoMessage)
                no_change_cnt = 0
            else:
                no_change_cnt += 1
            last_queue_info_message = queueInfoMessage
            if polling_rate * logging_counter >= logging_rate:
                logger.info(f"[LegoQueueInfoRunner] Queues lengths lastImages:{len(self.lastImages)}; storage_queue:{self.storage_queue.len(CAPTURE_TAG)}; processing_queue:{self.processing_queue.len(CAPTURE_TAG)}; annotation_queue:{self.annotation_queue.len(CAPTURE_TAG)}; sort_queue:{self.sort_queue.len(CAPTURE_TAG)}; crops_length:{self.crops_queue.len(CAPTURE_TAG)}")
                if no_change_cnt * polling_rate >= 4 * logging_rate:
                    self.send_queue_info_message(queueInfoMessage)
                    no_change_cnt = 0
                logging_counter = 0
            else:
                logging_counter += 1
            time.sleep(polling_rate)
            continue

    def get_queue_info_message(self):
        return QueueInfoMessage(len(self.lastImages), self.storage_queue.len(CAPTURE_TAG), self.processing_queue.len(CAPTURE_TAG), self.annotation_queue.len(CAPTURE_TAG), self.sort_queue.len(CAPTURE_TAG), self.crops_queue.len(CAPTURE_TAG))

    def send_queue_info_message(self, queue_info_message=None):
        if queue_info_message is None:
            queue_info_message = self.get_queue_info_message()
        self.hub_connection.send("sendQueuesInfoMessage", [queue_info_message])

