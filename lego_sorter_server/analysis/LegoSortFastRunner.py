import random
import string
import time
from loguru import logger
from collections import deque
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import Event
from collections import Counter
import requests
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

        self.pos = 0  # last detection pos

        self.selected_label = ""  # label selected on
        self.active_time_start = 0.0
        self.wait_time_start = 0.0

        db = SessionLocal()
        self.sort = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "sort").first().value == "true"
        self.active_time = int(db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "camera_conveyor_active_time").first().value)
        self.wait_time = int(db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "camera_conveyor_wait_time").first().value)
        self.conveyor_local_address = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "conveyor_local_address").first().value
        self.sorter_local_address = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "sorter_local_address").first().value
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

    def _started_active_time(self):
        return self.active_time_start != 0.0

    def _finished_active_time(self):
        return time.time()*1000 > self.active_time_start + self.active_time

    def _finished_active_time_and_wait_time(self):
        return time.time()*1000 > self.active_time_start + self.active_time + self.wait_time

    def _started_wait_time(self):
        return self.wait_time_start != 0.0

    def _finished_wait_time(self):
        return time.time()*1000 > self.wait_time_start + self.wait_time

    # we trust speed configuration and splitter that next lego will be in upper part of picture (earlier than last lego)
    def _new_lego_brick(self, lowest_y):
        return self.pos > lowest_y or self.pos == 0

    def _should_prepare_sorter_arm_pos(self, height):
        return self.pos > height/2 and self.selected_label == ""

    def _should_set_definitive_sorter_arm_pos(self, height):
        return self.pos > height * (3 / 4) and not self._started_active_time()

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
            detection_results, classification_results, id, session, height, width = self.storage_queue.next(CAPTURE_TAG)
            futures.add( self.executor.submit(self.__process_next_image,detection_results, classification_results, id, session, height, width))

    def __process_next_image(self, detection_results, classification_results, id, session, height, width):
        start_time = time.time()
        db = SessionLocal()
        self.sort = db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "sort").first().value == "true"
        self.active_time = int(db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "camera_conveyor_active_time").first().value)
        self.wait_time = int(db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "camera_conveyor_wait_time").first().value)
        self.camera_conveyor_frequency = int(db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "camera_conveyor_frequency").first().value)
        self.camera_conveyor_duty_cycle = int(db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "camera_conveyor_duty_cycle").first().value)
        self.splitting_conveyor_frequency = int(db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "splitting_conveyor_frequency").first().value)
        self.splitting_conveyor_duty_cycle = int(db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "splitting_conveyor_duty_cycle").first().value)
        self.conveyor_local_address = db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "conveyor_local_address").first().value
        self.sorter_local_address = db.query(Models.DBConfiguration).filter(
            Models.DBConfiguration.option == "sorter_local_address").first().value
        db.close()

        if not self.sort:
            return

        logger.info(f"[LegoSortFastRunner] sort {self.sort} active_time {self.active_time} wait_time {self.wait_time}")
        # detectionResults, classificationResults, image = self.storage_queue.next(CAPTURE_TAG)

        objects = ""

        valid_detections = []

        lowest_label = ""
        lowest_cat = ""
        lowest_pos = ""
        lowest_y = 0
        lowest_index = 0
        lowest_label_top5 = []
        lowest_score_top5 = [0]


        # get details about lowest LEGO in image
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

        # wait if detected new lego - at the moment LEGO is still on camera conveyor,
        # so it can't be stopped we stop splitter conveyor to prevent new lego entering camera conveyor
        if len(valid_detections) > 0 and self._new_lego_brick(lowest_y) and self._started_active_time() and not self._finished_active_time():
            dictionary_sort = [MySortMessage("new lego", lowest_score_top5[0], f"Start waiting for start of sorting, detected new lego", id, session)]
            self.hub_connection.send("sendSortMessage", [dictionary_sort])

            requests.get(f"{self.conveyor_local_address}/stop_splitting_conveyor")
            time.sleep(((self.active_time_start + self.active_time) - (time.time() * 1000)) / 1000)

        # start waiting, reset variables (trust timing that it will drop lego)
        if self._started_active_time() and self._finished_active_time() and not self._started_wait_time() and not self._finished_active_time_and_wait_time():
            self.wait_time_start = self.active_time_start + self.active_time
            dictionary_sort = [MySortMessage(self.selected_label, lowest_score_top5[0], f"Started waiting for {self.selected_label} sorting end", id, session)]
            self.hub_connection.send("sendSortMessage", [dictionary_sort])

        # wait if detected new lego - at the moment sorter is sorting previous lego
        if len(valid_detections) > 0 and self._new_lego_brick(lowest_y) and self._started_wait_time() and not self._finished_wait_time():
            dictionary_sort = [MySortMessage("new lego", lowest_score_top5[0], f"Start waiting for end of sorting, detected new lego", id, session)]
            self.hub_connection.send("sendSortMessage", [dictionary_sort])

            requests.get(f"{self.conveyor_local_address}/stop_splitting_conveyor")
            requests.get(f"{self.conveyor_local_address}/stop_camera_conveyor")
            time.sleep(((self.wait_time_start + self.wait_time) - (time.time() * 1000))/1000)

        # reset if processing new lego
        # requirement thar sctve time started prevent errors when conveyor is stopped, but will cause error if lego won't be detected in 75% - 100% part
        if self._started_active_time() and self._new_lego_brick(lowest_y):
            dictionary_sort = [MySortMessage(self.selected_label, lowest_score_top5[0], f"Reset", id, session)]
            self.hub_connection.send("sendSortMessage", [dictionary_sort])
            requests.get(f"{self.conveyor_local_address}/start_camera_conveyor",
                         params={"frequency": self.camera_conveyor_frequency,
                                 "duty_cycle": self.camera_conveyor_duty_cycle})
            requests.get(f"{self.conveyor_local_address}/start_splitting_conveyor",
                         params={"frequency": self.splitting_conveyor_frequency,
                                 "duty_cycle": self.splitting_conveyor_duty_cycle})
            self.selected_label = ""
            self.active_time_start = 0.0
            self.wait_time_start = 0.0
            self.low_label = []
            self.low_label_top5 = []
            self.low_score_top5 = []

        # last lego still on conveyor update it position
        if len(valid_detections) > 0 and not self._new_lego_brick(lowest_y):
            self.low_label.append(lowest_label)
            self.low_label_top5.append(lowest_label_top5)
            self.low_score_top5.append(lowest_score_top5)
            self.pos = lowest_y
            dictionary_sort = [MySortMessage(lowest_label, lowest_score_top5[0], f"Updating pos of last LEGO {lowest_label}", id, session)]
            self.hub_connection.send("sendSortMessage", [dictionary_sort])

        # if wait time ended trust model, if something detected lower on conveyor previous lego dropped
        if not self._started_wait_time() and len(valid_detections) > 0 and self._new_lego_brick(lowest_y):
            self.pos = lowest_y
            self.low_label.append(lowest_label)
            self.low_label_top5.append(lowest_label_top5)
            self.low_score_top5.append(lowest_score_top5)
            dictionary_sort = [MySortMessage(lowest_label, lowest_score_top5[0], f"New LEGO sorting started", id, session)]
            self.hub_connection.send("sendSortMessage", [dictionary_sort])

        # start setting sorting arm
        if len(self.low_label) > 0 and self._should_prepare_sorter_arm_pos(height):
            occurrence_count = Counter(self.low_label)
            self.selected_label = occurrence_count.most_common(1)[0][0]
            how_many = occurrence_count.most_common(1)[0][1]
            dictionary_sort = [MySortMessage(self.selected_label, lowest_score_top5[0], f"Set arm for {self.selected_label} {how_many}/{len(self.low_label)} from {lowest_cat} to poz {lowest_pos}", id, session)]
            self.hub_connection.send("sendSortMessage", [dictionary_sort])
            requests.get(f"{self.sorter_local_address}/sort?action={lowest_pos}")

        # change sorting arm if last detections changed score
        if self._should_set_definitive_sorter_arm_pos(height):
            self.active_time_start = time.time() * 1000
            occurrence_count = Counter(self.low_label)
            most_common = occurrence_count.most_common(1)[0][0]
            how_many = occurrence_count.most_common(1)[0][1]
            if most_common != self.selected_label:
                self.selected_label = most_common  # todo
                dictionary_sort = [MySortMessage(self.selected_label, lowest_score_top5[0], f"Changed arm for {self.selected_label} {how_many}/{len(self.low_label)} from {lowest_cat} to poz {lowest_pos}", id, session)]
                self.hub_connection.send("sendSortMessage", [dictionary_sort])
                requests.get(f"{self.sorter_local_address}/sort?action={lowest_pos}")

        elapsed_millis = (time.time() - start_time) * 1000
        logger.info(f"[LegoSortFastRunner] Request processed in {elapsed_millis} ms.")
