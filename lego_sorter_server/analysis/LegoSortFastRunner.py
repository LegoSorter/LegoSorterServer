import random
import string
import time
from loguru import logger
from collections import deque
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import Event
from collections import Counter

from signalrcore.hub_connection_builder import HubConnectionBuilder
from uuid6 import uuid7

from lego_sorter_server.analysis.detection.DetectionUtils import crop_with_margin
from lego_sorter_server.analysis.detection.LegoLabeler import LegoLabeler
from lego_sorter_server.database import Models
from lego_sorter_server.database.Database import SessionLocal
from lego_sorter_server.generated.Messages_pb2 import ListOfBoundingBoxes
from lego_sorter_server.images.queue.ImageAnnotationQueueFast import ImageAnnotationQueueFast
from lego_sorter_server.images.queue.ImageSortQueueFast import ImageSortQueueFast
from lego_sorter_server.images.queue.ImageStorageQueueFast import ImageStorageQueueFast
from lego_sorter_server.images.queue.ImageProcessingQueueFast import ImageProcessingQueueFast, CAPTURE_TAG
from lego_sorter_server.images.storage.LegoImageStorageFast import LegoImageStorageFast
from lego_sorter_server.service.BrickCategoryConfig import BrickCategoryConfig
from lego_sorter_server.service.ImageProtoUtils import ImageProtoUtils
from lego_sorter_server.database.Models import *


class MySortMessage:
    def __init__(self, label, score, info, id, session):
        self.label = label
        self.score = score
        self.info = info
        self.id = id
        self.session = session


class LegoSortFastRunner:
    DETECTION_SCORE_THRESHOLD = 0.5

    def __init__(self, storage_queue: ImageSortQueueFast, sorter_config: BrickCategoryConfig, hub_connection: HubConnectionBuilder, sort_fast_runer_executor: ThreadPoolExecutor, event: Event):
        self.storage_queue = storage_queue
        self.event = event
        self.sorter_config = sorter_config
        self.hub_connection = hub_connection

        self.low_label = []  # history of the lowest label, probably should be [[]] for each level
        self.low_label_top5 = []
        self.low_score_top5 = []

        self.poz = 0  # last detection poz
        self.sec_poz = 0  # last detection poz of second

        self.selected_label = ""  # label selected on
        self.active_time_start = 0.0
        self.wait_time_start = 0.0

        db = SessionLocal()
        self.sort = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "sort").first().value == "true"
        self.active_time = int(db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "camera_conveyor_active_time").first().value)
        self.wait_time = int(db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "camera_conveyor_wait_time").first().value)
        db.close()

        # self.storage = store

        # queue, detector and storage are not thread safe, so it limits the number of workers to one
        self.executor = sort_fast_runer_executor
        # self.executor = futures.ThreadPoolExecutor(max_workers=4)
        logger.info("[LegoSortFastRunner] Ready for processing the queue.")

    def start_detecting(self):
        logger.info("[LegoSortFastRunner] Started processing the queue.")
        return self.executor.submit(self._exception_handler, self._process_queue)

    def stop_detecting(self):
        logger.info("[LegoSortFastRunner] Processing is being terminated.")
        self.event.set()
        self.executor.shutdown()

    @staticmethod
    def _exception_handler(method, args=[]):
        try:
            method(*args)
        except Exception as exc:
            logger.exception(f"[LegoSortFastRunner] Got an exception:\n {str(exc)}")
            raise exc

    def _process_queue(self):
        polling_rate = 0.2  # in seconds
        logging_rate = 30  # in seconds
        logging_counter = 0
        limit = 2
        futures = set()

        while True:
            if self.event.isSet():
                logger.info("[LegoSortFastRunner] Processing queue is being terminated.")
                return
            if self.storage_queue.len(CAPTURE_TAG) == 0:
                if polling_rate * logging_counter >= logging_rate:
                    logger.info("[LegoSortFastRunner] Queue is empty. Waiting... ")
                    logging_counter = 0
                else:
                    logging_counter += 1
                time.sleep(polling_rate)
                continue

            # logger.info("[LegoSortFastRunner] Queue not empty - processing data")
            logger.info(f"[LegoSortFastRunner] Queue length:{self.storage_queue.len(CAPTURE_TAG)}")

            # if (sortFastRunerExecutor_max_workers.value == "1"):
            #     detection_results, classification_results, imageid = self.storage_queue.next(CAPTURE_TAG)
            #     self.__process_next_image(detection_results, classification_results, imageid)
            # else:
            if len(futures) >= limit:
                completed, futures = wait(futures, return_when=FIRST_COMPLETED)
            detection_results, classification_results, id, session = self.storage_queue.next(CAPTURE_TAG)
            futures.add( self.executor.submit(self.__process_next_image,detection_results, classification_results, id, session))

    def __process_next_image(self, detection_results, classification_results, id, session):
        start_time = time.time()
        db = SessionLocal()
        self.sort = db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "sort").first().value == "true"
        self.active_time = int(db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "camera_conveyor_active_time").first().value)
        self.wait_time = int(db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "camera_conveyor_wait_time").first().value)
        db.close()
        logger.info(f"[LegoSortFastRunner] sort {self.sort} active_time {self.active_time} wait_time {self.wait_time}")
        # detectionResults, classificationResults, image = self.storage_queue.next(CAPTURE_TAG)

        objects = ""

        valid_detections = []

        lowest_label = ""
        lowest_cat = ""
        lowest_pos = ""
        lowest_y = 0
        lowest_index = 0

        for i in range(len(detection_results.detection_boxes)):
            if detection_results.detection_scores[i] < self.DETECTION_SCORE_THRESHOLD:
                continue

            y_min, x_min, y_max, x_max = [int(coord) for coord in detection_results.detection_boxes[i]]
            score = classification_results.classification_scores[i]
            score_top5 = classification_results.classification_scores_top5[i]
            label = classification_results.classification_classes[i]
            label_top5 = classification_results.classification_classes_top5[i]
            x_mean = int((x_max+x_min)/2)
            y_mean = int((y_max+y_min)/2)
            if y_mean >= lowest_y:
                lowest_y = y_mean
                lowest_label = label
                lowest_label_top5 = label_top5
                lowest_score_top5 = score_top5
                if self.sorter_config is not None:
                    lowest_cat, lowest_pos = self.sorter_config[label]
            valid_detections.append({"x": x_mean, "y": y_mean, "score": score, "label": label, "score_top5": score_top5, "label_top5": label_top5})

        # wait if detected new lego
        if self.wait_time_start != 0.0 and time.time() * 1000 < self.wait_time_start + self.wait_time:
            if self.sort:
                dictionary_sort = []
                dictionary_sort.append(MySortMessage("new lego", 0.0, f"Start waiting, detected new lego", id, session))
                self.hub_connection.send("sendSortMessage", [dictionary_sort])
            time.sleep(((self.wait_time_start + self.wait_time) - (time.time() * 1000))/1000)

        # reset if wait time ended or active_time + wait_time ended (if no detection since active_time start)
        if self.active_time_start != 0.0 and time.time()*1000 > self.active_time_start+self.active_time+self.wait_time or self.wait_time_start != 0.0 and time.time()*1000 > self.wait_time_start + self.wait_time:
            if self.sort:
                dictionary_sort = []
                dictionary_sort.append(MySortMessage(self.selected_label, 0.0, f"Reset", id, session))
                self.hub_connection.send("sendSortMessage", [dictionary_sort])
            self.selected_label = ""
            self.active_time_start = 0.0
            self.wait_time_start = 0.0
            self.low_label = []
            self.lowest_label_top5 = []
            self.lowest_score_top5 = []


        # if wait time ended trust model, if something detected lower on conveyor previous lego dropped
        if self.wait_time_start == 0.0 and len(valid_detections) > 0 and lowest_y < self.poz:
            self.poz = lowest_y
            self.low_label.append(lowest_label)
            if self.sort:
                dictionary_sort = []
                dictionary_sort.append(MySortMessage(lowest_label, 0.0, f"New LEGO sorting started", id, session))
                self.hub_connection.send("sendSortMessage", [dictionary_sort])

        # last lego still on conveyor
        if len(valid_detections) > 0 and lowest_y >= self.poz:
            self.low_label.append(lowest_label)
            self.poz = lowest_y
            if self.sort:
                dictionary_sort = []
                dictionary_sort.append(MySortMessage(lowest_label, 0.0, f"Updating poz of last LEGO {lowest_label}", id, session))
                self.hub_connection.send("sendSortMessage", [dictionary_sort])

        # start setting sorting arm
        if len(self.low_label) > 0 and self.poz > 1920/2 and self.selected_label == "":
            occurence_count = Counter(self.low_label)
            self.selected_label = occurence_count.most_common(1)[0][0]
            how_many = occurence_count.most_common(1)[0][1]
            if self.sort:
                dictionary_sort = []
                dictionary_sort.append(MySortMessage(self.selected_label, 0.0, f"set arm for {self.selected_label} {how_many}/{len(self.low_label)} from {lowest_cat} to poz {lowest_pos}", id, session))
                self.hub_connection.send("sendSortMessage", [dictionary_sort])

        if self.poz > 1920 * (3 / 4) and self.active_time_start == 0.0:
            self.active_time_start = time.time() * 1000
            occurence_count = Counter(self.low_label)
            most_common = occurence_count.most_common(1)[0][0]
            how_many = occurence_count.most_common(1)[0][1]
            if most_common != self.selected_label and self.sort:
                self.selected_label = most_common  # todo
                dictionary_sort = []
                dictionary_sort.append(MySortMessage(self.selected_label, 0.0,
                                                    f"Changed arm for {self.selected_label} {how_many}/{len(self.low_label)} from {lowest_cat} to poz {lowest_pos}",
                                                    id, session))
                self.hub_connection.send("sendSortMessage", [dictionary_sort])

        # start waiting, reset variables (trust timing that it will drop lego)
        if self.active_time_start != 0.0 and time.time()*1000 > self.active_time_start+self.active_time and self.wait_time_start == 0.0:
            self.wait_time_start = time.time() * 1000
            if self.sort:
                dictionary_sort = []
                dictionary_sort.append(MySortMessage(self.selected_label, 0.0,f"Started waiting for {self.selected_label} sorting end", id, session))
                self.hub_connection.send("sendSortMessage", [dictionary_sort])

            # trust model, if nothing detected assume no lego on conveyor, override stop
            if len(valid_detections) == 0:
                if self.sort:
                    dictionary_sort = []
                    dictionary_sort.append(MySortMessage("nothing", 0.0,f"Override stop, nothing detected", id, session))
                    self.hub_connection.send("sendSortMessage", [dictionary_sort])
            # eles wait till end of wait time
            else:
                time.sleep(self.wait_time/1000)

        elapsed_millis = (time.time() - start_time) * 1000
        logger.info(f"[LegoSortFastRunner] Request processed in {elapsed_millis} ms.")
