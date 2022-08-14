import random
import string
import time
import logging
from concurrent import futures
from uuid6 import uuid7

from signalrcore.hub_connection_builder import HubConnectionBuilder

from lego_sorter_server.analysis.AnalysisFastService import AnalysisFastService
from lego_sorter_server.analysis.detection.DetectionUtils import crop_with_margin
from lego_sorter_server.analysis.detection.LegoLabeler import LegoLabeler
from lego_sorter_server.generated.Messages_pb2 import ListOfBoundingBoxes
from lego_sorter_server.images.queue.ImageStorageQueueFast import ImageStorageQueueFast
from lego_sorter_server.images.queue.ImageProcessingQueueFast import ImageProcessingQueueFast, CAPTURE_TAG
from lego_sorter_server.images.storage.LegoImageStorageFast import LegoImageStorageFast
from lego_sorter_server.service.ImageProtoUtils import ImageProtoUtils


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
                 hub_connection: HubConnectionBuilder, store: LegoImageStorageFast):
        self.storage = store
        self.storage_queue = storage_queue
        self.processing_queue = processing_queue
        self.analysis_service = AnalysisFastService()
        # self.storage = store
        self.hub_connection = hub_connection
        # queue, detector and storage are not thread safe, so it limits the number of workers to one
        self.executor = futures.ThreadPoolExecutor(max_workers=4)
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

    def _process_queue(self, save_cropped_image=True, save_label_file=True):
        polling_rate = 0.2  # in seconds
        logging_rate = 30  # in seconds
        logging_counter = 0

        while True:
            if self.storage_queue.len(CAPTURE_TAG) == 0:
                if polling_rate * logging_counter >= logging_rate:
                    logging.info("[LegoStorageFastRunner] Queue is empty. Waiting... ")
                    logging_counter = 0
                else:
                    logging_counter += 1
                time.sleep(polling_rate)
                continue

            logging.info("[LegoStorageFastRunner] Queue not empty - processing data")
            self.__process_next_image(save_cropped_image, save_label_file)

    def __process_next_image(self, save_cropped_image, save_label_file):
        image, lego_class = self.storage_queue.next(CAPTURE_TAG)
        # start_time = time.time()
        #
        # # image = ImageProtoUtils.prepare_image(request)
        # detection_results, classification_results = self.analysis_service.detect_and_classify(image)
        # bb_list: ListOfBoundingBoxes = ImageProtoUtils.prepare_response_from_analysis_results(detection_results,
        #                                                                                       classification_results)
        #
        # elapsed_millis = (time.time() - start_time) * 1000
        # logging.info(f"[LegoAnalysisFastService] Detecting, classifying and preparing response took "
        #              f"{elapsed_millis} milliseconds.")
        #
        # dictionary = []
        # if len(bb_list.packet) > 0:
        #     for packet in bb_list.packet:
        #         dictionary.append(
        #             MyMessage(packet.ymin, packet.xmin, packet.ymax, packet.xmax, packet.label, packet.score))
        #     self.hub_connection.send("sendMessage", [dictionary])

        # return bb_list

        # detection_results = self.analysis_service.detect(image)
        # detected_counter = 0
        # bbs = []
        # for i in range(len(detection_results.detection_classes)):
        #     if detection_results.detection_scores[i] < LegoStorageFastRunner.DETECTION_SCORE_THRESHOLD:
        #         logging.info(
        #             f"[LegoAnalysisFastRunner] One result discarded for {lego_class} as it is under the threshold:\n"
        #             f"Score = {detection_results.detection_scores[i]}, "
        #             f"BoundingBox = {detection_results.detection_boxes[i]}")
        #         continue
        #
        #     detected_counter += 1
        #     bbs.append(detection_results.detection_boxes[i])
        #
        #     if save_cropped_image is True:
        #         image_new = crop_with_margin(image, *detection_results.detection_boxes[i])
        #         self.storage.save_image(image_new, lego_class, prefix)

        # prefix = f'{detected_counter}_{prefix}'
        prefix = str(uuid7())

        # if lego_class != "":
            # filename = self.storage.save_image(image, lego_class, prefix)
        self.processing_queue.add(CAPTURE_TAG, image, prefix)

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
