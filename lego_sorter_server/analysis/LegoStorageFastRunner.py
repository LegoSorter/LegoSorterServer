import random
import string
import time
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from uuid6 import uuid7
import pathlib

from lego_sorter_server.analysis.AnalysisFastService import AnalysisFastService
from lego_sorter_server.analysis.detection.DetectionUtils import crop_with_margin
from lego_sorter_server.analysis.detection.LegoLabeler import LegoLabeler
from lego_sorter_server.generated.Messages_pb2 import ListOfBoundingBoxes
from lego_sorter_server.images.queue.ImageStorageQueueFast import ImageStorageQueueFast
from lego_sorter_server.images.queue.ImageProcessingQueueFast import ImageProcessingQueueFast, CAPTURE_TAG
from lego_sorter_server.images.storage.LegoImageStorageFast import LegoImageStorageFast
from lego_sorter_server.service.ImageProtoUtils import ImageProtoUtils
from peewee import *
from lego_sorter_server.database.Models import *


class MyMessage:
    def __init__(self, ymin, xmin, ymax, xmax, label, score):
        self.ymin = ymin
        self.xmin = xmin
        self.ymax = ymax
        self.xmax = xmax
        self.label = label
        self.score = score


class LegoStorageFastRunner:
    DETECTION_SCORE_THRESHOLD = 0.5

    def __init__(self, storage_queue: ImageStorageQueueFast, processing_queue: ImageProcessingQueueFast,
                 store: LegoImageStorageFast, storageFastRunerExecutor: ThreadPoolExecutor, lastImages: deque):
        self.storage = store
        self.storage_queue = storage_queue
        self.processing_queue = processing_queue
        self.analysis_service = AnalysisFastService()
        self.lastImages = lastImages
        # self.storage = store
        # queue, detector and storage are not thread safe, so it limits the number of workers to one
        self.executor = storageFastRunerExecutor
        # self.executor = futures.ThreadPoolExecutor(max_workers=4)
        logging.info("[LegoStorageFastRunner] Ready for processing the queue.")

    def start_detecting(self):
        logging.info("[LegoStorageFastRunner] Started processing the queue.")
        return self.executor.submit(self._exception_handler, self._process_queue)

    def stop_detecting(self):
        logging.info("[LegoStorageFastRunner] Processing is being terminated.")
        self.executor.shutdown()

    @staticmethod
    def _exception_handler(method, args=[]):
        try:
            method(*args)
        except Exception as exc:
            logging.exception(f"[LegoStorageFastRunner] Got an exception:\n {str(exc)}")
            raise exc

    def _process_queue(self):
        polling_rate = 0.2  # in seconds
        logging_rate = 30  # in seconds
        logging_counter = 0
        storageFastRunerExecutor_max_workers = DBConfiguration.get(option="storageFastRunerExecutor_max_workers")
        limit = (int(storageFastRunerExecutor_max_workers.value)-1)*2
        futures = set()

        while True:
            if self.storage_queue.len(CAPTURE_TAG) == 0:
                if polling_rate * logging_counter >= logging_rate:
                    logging.info("[LegoStorageFastRunner] Queue is empty. Waiting... ")
                    logging_counter = 0
                else:
                    logging_counter += 1
                time.sleep(polling_rate)
                continue

            # logging.info("[LegoStorageFastRunner] Queue not empty - processing data")
            logging.info(f"[LegoStorageFastRunner] Storage queue length:{self.storage_queue.len(CAPTURE_TAG)}")
            # self.__process_next_image()
            if (storageFastRunerExecutor_max_workers.value == "1"):
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

        image = ImageProtoUtils.prepare_image(request)

        self.lastImages.append(image)


        # prefix = f'{detected_counter}_{prefix}'
        prefix = str(uuid7())
        dbimage = None
        if lego_class != "":
            filename = self.storage.save_image(image, lego_class, prefix)
            session, _ = DBSession.get_or_create(name=lego_class, path=f"{pathlib.Path().resolve()}/lego_sorter_server/images/storage/stored/{lego_class}")
            dbimage = DBImage.create(owner=session, filename=filename, image_width=image.width, image_height=image.height, VOC_exist=False)
            elapsed_millis = int((time.time() - start_time) * 1000)
            logging.info(f"[LegoStorageFastRunner] JPG saved in {elapsed_millis} milliseconds.")
        self.processing_queue.add(CAPTURE_TAG, image, dbimage)

        elapsed_millis = int((time.time() - start_time) * 1000)
        logging.info(f"[LegoStorageFastRunner] Request processed in {elapsed_millis} milliseconds.")

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
