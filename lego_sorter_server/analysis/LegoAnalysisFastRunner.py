import random
import string
import time
from loguru import logger

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import Event

from signalrcore.hub_connection_builder import HubConnectionBuilder

from lego_sorter_server.analysis.AnalysisFastService import AnalysisFastService
from lego_sorter_server.database import Models
from lego_sorter_server.database.Database import SessionLocal
from lego_sorter_server.generated.Messages_pb2 import ListOfBoundingBoxes
from lego_sorter_server.images.queue.ImageAnnotationQueueFast import ImageAnnotationQueueFast
from lego_sorter_server.images.queue.ImageProcessingQueueFast import ImageProcessingQueueFast, CAPTURE_TAG
from lego_sorter_server.images.queue.ImageSortQueueFast import ImageSortQueueFast
from lego_sorter_server.service.ImageProtoUtils import ImageProtoUtils


class LegoAnalysisFastRunner:
    DETECTION_SCORE_THRESHOLD = 0.5

    def __init__(self, processing_queue: ImageProcessingQueueFast, hub_connection: HubConnectionBuilder, analyzer_fast_runer_executor: ThreadPoolExecutor, annotation_queue: ImageAnnotationQueueFast, sort_queue: ImageSortQueueFast, event: Event):
        self.processing_queue = processing_queue
        self.event = event
        self.annotation_queue = annotation_queue
        self.sort_queue = sort_queue
        self.hub_connection = hub_connection
        self.analysis_service = AnalysisFastService()
        # queue, detector and storage are not thread safe, so it limits the number of workers to one
        self.executor = analyzer_fast_runer_executor
        # self.executor = futures.ThreadPoolExecutor(max_workers=1)
        logger.info("[LegoAnalysisFastRunner] Ready for processing the queue.")

    def start_detecting(self):
        logger.info("[LegoAnalysisFastRunner] Started processing the queue.")
        return self.executor.submit(self._exception_handler, self._process_queue)

    def stop_detecting(self):
        logger.info("[LegoAnalysisFastRunner] Processing is being terminated.")
        self.event.set()
        self.executor.shutdown()

    @staticmethod
    def _exception_handler(method, args=[]):
        try:
            method(*args)
        except Exception as exc:
            logger.exception(f"[LegoAnalysisFastRunner] Got an exception:\n {str(exc)}")
            raise exc

    def _process_queue(self):
        polling_rate = 0.2  # in seconds
        logging_rate = 30  # in seconds
        logging_counter = 0
        db = SessionLocal()
        analyzer_fast_runer_executor_max_workers = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "analyzer_fast_runer_executor_max_workers").first()
        db.close()
        limit = 2
        futures = set()

        while True:
            if self.event.isSet():
                logger.info("[LegoAnalysisFastRunner] Processing queue is being terminated.")
                return
            if self.processing_queue.len(CAPTURE_TAG) == 0:
                if polling_rate * logging_counter >= logging_rate:
                    logger.info("[LegoAnalysisFastRunner] Queue is empty. Waiting... ")
                    logging_counter = 0
                else:
                    logging_counter += 1
                time.sleep(polling_rate)
                continue
            logger.info(f"[LegoAnalysisFastService] Analysis queue length:{self.processing_queue.len(CAPTURE_TAG)}")
            # logging.info("[LegoAnalysisFastRunner] Queue not empty - processing data")
            if analyzer_fast_runer_executor_max_workers.value == "1":
                self.__process_next_image()
            else:
                if len(futures) >= limit:
                    completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                futures.add(self.executor.submit(self.__process_next_image))

    def __process_next_image(self):
        start_time = time.time()
        image, imageid, id, session = self.processing_queue.next(CAPTURE_TAG)

        # image = ImageProtoUtils.prepare_image(request)
        detection_results, classification_results = self.analysis_service.detect_and_classify(image, imageid, id, session)

        # bb_list: ListOfBoundingBoxes = ImageProtoUtils.prepare_response_from_analysis_results(detection_results,classification_results)
        bb_list = ImageProtoUtils.prepare_response_from_analysis_results_top5(detection_results,classification_results)

        elapsed_millis = (time.time() - start_time) * 1000
        logger.info(f"[LegoAnalysisFastService] Detecting, classifying and preparing response took "
                     f"{elapsed_millis} ms.")

        dictionary = []

        if imageid is not None:
            self.annotation_queue.add(CAPTURE_TAG, detection_results, classification_results, imageid)
        if len(bb_list) > 0:
            for packet in bb_list:
                packet.id = id
                packet.session = session
                dictionary.append(packet)
            self.sort_queue.add(CAPTURE_TAG, detection_results, classification_results, id, session)
            self.hub_connection.send("sendMessage", [dictionary])

        # return bb_list


        # prefix = self._get_random_hash() + "_"
        # detection_results = self.analysis_service.detect(image)
        # detected_counter = 0
        # bbs = []
        # for i in range(len(detection_results.detection_classes)):
        #     if detection_results.detection_scores[i] < LegoAnalysisFastRunner.DETECTION_SCORE_THRESHOLD:
        #         logger.info(
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
        #
        # prefix = f'{detected_counter}_{prefix}'
        # filename = self.storage.save_image(image, f'original-{lego_class}', prefix)
        #
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
