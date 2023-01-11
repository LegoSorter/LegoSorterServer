import os
import random
import string
import time
from loguru import logger
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import Event

from uuid6 import uuid7

from lego_sorter_server.analysis.detection.DetectionUtils import crop_with_margin
from lego_sorter_server.analysis.detection.LegoLabeler import LegoLabeler
from lego_sorter_server.database import Models
from lego_sorter_server.database.Database import SessionLocal, get_or_create
from lego_sorter_server.generated.Messages_pb2 import ListOfBoundingBoxes
from lego_sorter_server.images.queue.ImageCropQueueFast import ImageCropQueueFast
from lego_sorter_server.images.queue.ImageStorageQueueFast import ImageStorageQueueFast
from lego_sorter_server.images.queue.ImageProcessingQueueFast import ImageProcessingQueueFast, CAPTURE_TAG
from lego_sorter_server.images.storage.LegoImageStorageFast import LegoImageStorageFast
from lego_sorter_server.service.ImageProtoUtils import ImageProtoUtils
from lego_sorter_server.database.Models import *


class LegoStorageCropsFastRunner:
    DETECTION_SCORE_THRESHOLD = 0.5

    def __init__(self, storage_queue: ImageCropQueueFast, store: LegoImageStorageFast, storage_queue_fast_runer_executor: ThreadPoolExecutor, event: Event):
        self.storage = store
        self.event = event
        self.storage_queue = storage_queue
        db = SessionLocal()
        self.crop = db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "crop").first().value == "true"
        db.close()
        # self.storage = store
        # queue, detector and storage are not thread safe, so it limits the number of workers to one
        self.executor = storage_queue_fast_runer_executor
        # self.executor = futures.ThreadPoolExecutor(max_workers=4)
        logger.info("[LegoStorageCropsFastRunner] Ready for processing the queue.")

    def start_detecting(self):
        logger.info("[LegoStorageCropsFastRunner] Started processing the queue.")
        return self.executor.submit(self._exception_handler, self._process_queue)

    def stop_detecting(self):
        logger.info("[LegoStorageCropsFastRunner] Processing is being terminated.")
        self.event.set()
        self.executor.shutdown()

    @staticmethod
    def _exception_handler(method, args=[]):
        try:
            method(*args)
        except Exception as exc:
            logger.exception(f"[LegoStorageCropsFastRunner] Got an exception:\n {str(exc)}")
            raise exc

    def _process_queue(self):
        polling_rate = 0.2  # in seconds
        logging_rate = 30  # in seconds
        logging_counter = 0
        db = SessionLocal()
        storage_fast_runer_executor_max_workers = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "storage_fast_runer_executor_max_workers").first()
        db.close()
        # storage_fast_runer_executor_max_workers = DBConfiguration.get(option="storage_fast_runer_executor_max_workers")
        limit = (int(storage_fast_runer_executor_max_workers.value)-1)*2
        futures = set()

        while True:
            if self.event.isSet():
                logger.info("[LegoStorageCropsFastRunner] Processing queue is being terminated.")
                return
            if self.storage_queue.len(CAPTURE_TAG) == 0:
                if polling_rate * logging_counter >= logging_rate:
                    logger.info("[LegoStorageCropsFastRunner] Queue is empty. Waiting... ")
                    logging_counter = 0
                else:
                    logging_counter += 1
                time.sleep(polling_rate)
                continue

            # logger.info("[LegoStorageFastRunner] Queue not empty - processing data")
            logger.info(f"[LegoStorageCropsFastRunner] Storage queue length:{self.storage_queue.len(CAPTURE_TAG)}")
            # self.__process_next_image()
            if storage_fast_runer_executor_max_workers.value == "1":
                images, detection_results, classification_results, Imageid, Id, lego_class = self.storage_queue.next(CAPTURE_TAG)
                self.__process_next_image(images, detection_results, classification_results, Imageid, Id, lego_class)
            else:
                if len(futures) >= limit:
                    completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                images, detection_results, classification_results, Imageid, Id, lego_class = self.storage_queue.next(CAPTURE_TAG)
                futures.add(self.executor.submit(self.__process_next_image, images, detection_results, classification_results, Imageid, Id, lego_class))
            # self.executor.submit(self.__process_next_image, request, lego_class)

    def __process_next_image(self, images, detection_results, classification_results, Imageid, Id, lego_class):
        start_time = time.time()
        db = SessionLocal()
        self.crop = db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "crop").first().value == "true"
        db.close()
        # request, lego_class = self.storage_queue.next(CAPTURE_TAG)

        # image = ImageProtoUtils.prepare_image_cv2(images)

        # self.lastImages.append(image)
        old_lego_class = lego_class
        if self.crop:
            for ind, image in enumerate(images):
                lego_class = old_lego_class
                prefix = Id
                cls = classification_results.classification_classes[ind]
                score = classification_results.classification_scores[ind]
                prefix += "_" + cls
                prefix += "_" + "{:.2f}".format(score)
                if lego_class == "":
                    lego_class = "crops"
                prefix += "_" + str(uuid7())
                filename = self.storage.save_image_cv2(image, lego_class, prefix)
                elapsed_millis = (time.time() - start_time) * 1000
                logger.debug(f"[LegoStorageCropsFastRunner] JPG saved in {elapsed_millis} ms.")

        elapsed_millis = (time.time() - start_time) * 1000
        logger.info(f"[LegoStorageCropsFastRunner] Request processed in {elapsed_millis} ms.")


    @staticmethod
    def _get_random_hash(length=4):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
