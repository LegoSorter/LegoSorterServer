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
from lego_sorter_server.images.queue.ImageStorageQueueFast import ImageStorageQueueFast
from lego_sorter_server.images.queue.ImageProcessingQueueFast import ImageProcessingQueueFast, CAPTURE_TAG
from lego_sorter_server.images.storage.LegoImageStorageFast import LegoImageStorageFast
from lego_sorter_server.service.ImageProtoUtils import ImageProtoUtils
from lego_sorter_server.database.Models import *


class LegoStorageFastRunner:
    DETECTION_SCORE_THRESHOLD = 0.5

    def __init__(self, storage_queue: ImageStorageQueueFast, processing_queue: ImageProcessingQueueFast,
                 store: LegoImageStorageFast, storage_fast_runer_executor: ThreadPoolExecutor, last_images: deque, event: Event):
        self.storage = store
        self.event = event
        self.storage_queue = storage_queue
        self.processing_queue = processing_queue
        self.lastImages = last_images
        # self.storage = store
        # queue, detector and storage are not thread safe, so it limits the number of workers to one
        self.executor = storage_fast_runer_executor
        # self.executor = futures.ThreadPoolExecutor(max_workers=4)
        logger.info("[LegoStorageFastRunner] Ready for processing the queue.")

    def start_detecting(self):
        logger.info("[LegoStorageFastRunner] Started processing the queue.")
        return self.executor.submit(self._exception_handler, self._process_queue)

    def stop_detecting(self):
        logger.info("[LegoStorageFastRunner] Processing is being terminated.")
        self.event.set()
        self.executor.shutdown()

    @staticmethod
    def _exception_handler(method, args=[]):
        try:
            method(*args)
        except Exception as exc:
            logger.exception(f"[LegoStorageFastRunner] Got an exception:\n {str(exc)}")
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
                logger.info("[LegoStorageFastRunner] Processing queue is being terminated.")
                return
            if self.storage_queue.len(CAPTURE_TAG) == 0:
                if polling_rate * logging_counter >= logging_rate:
                    logger.info("[LegoStorageFastRunner] Queue is empty. Waiting... ")
                    logging_counter = 0
                else:
                    logging_counter += 1
                time.sleep(polling_rate)
                continue

            # logger.info("[LegoStorageFastRunner] Queue not empty - processing data")
            logger.info(f"[LegoStorageFastRunner] Storage queue length:{self.storage_queue.len(CAPTURE_TAG)}")
            # self.__process_next_image()
            if (storage_fast_runer_executor_max_workers.value == "1"):
                request, lego_class = self.storage_queue.next(CAPTURE_TAG)
                self.__process_next_image(request, lego_class)
            else:
                if len(futures) >= limit:
                    completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                request, lego_class = self.storage_queue.next(CAPTURE_TAG)
                futures.add(self.executor.submit(self.__process_next_image, request, lego_class))
            # self.executor.submit(self.__process_next_image, request, lego_class)

    def __process_next_image(self, request, lego_class):
        start_time = time.time()
        # request, lego_class = self.storage_queue.next(CAPTURE_TAG)

        image = ImageProtoUtils.prepare_image_cv2(request)

        self.lastImages.append(image)


        # prefix = f'{detected_counter}_{prefix}'
        prefix = str(uuid7())
        imageid = None
        dbimage = None
        db = SessionLocal()
        store_img_override = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "store_img_override").first().value == "true"
        if store_img_override:
            lego_class = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "store_img_session").first().value
        if lego_class != "":
            filename = self.storage.save_image_cv2(image, lego_class, prefix)
            path = os.path.abspath("lego_sorter_server/images/storage/stored")
            session, _ = get_or_create(db, Models.DBSession, name=lego_class, path=os.path.join(path, lego_class))
            # session, _ = DBSession.get_or_create(name=lego_class, path=f"{pathlib.Path().resolve()}/lego_sorter_server/images/storage/stored/{lego_class}")
            height, width, channels = image.shape
            dbimage = Models.DBImage(owner=session, filename=filename, image_width=width, image_height=height, VOC_exist=False)
            db.add(dbimage)
            db.commit()
            imageid = dbimage.id
            # dbimage = DBImage.create(owner=session, filename=filename, image_width=image.width, image_height=image.height, VOC_exist=False)
            elapsed_millis = (time.time() - start_time) * 1000
            logger.debug(f"[LegoStorageFastRunner] JPG saved in {elapsed_millis} ms.")
        db.close()
        self.processing_queue.add(CAPTURE_TAG, image, imageid, prefix, lego_class)

        elapsed_millis = (time.time() - start_time) * 1000
        logger.info(f"[LegoStorageFastRunner] Request processed in {elapsed_millis} ms.")

        # if save_label_file is True:
        #     path = self.storage.find_image_path(filename)
        #     width, height = image.size
        #     label_file = LegoLabeler().to_label_file(filename, str(path), width, height, bbs)
        #     xml_path = path.parent.absolute() / (filename.split(".")[-2] + ".xml")
        #     with open(xml_path, "w") as label_xml:
        #         label_xml.write(label_file)

    @staticmethod
    def _get_random_hash(length=4):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
