import random
import string
import time
import logging
from concurrent import futures

from lego_sorter_server.analysis.AnalysisService import AnalysisService
from lego_sorter_server.analysis.detection.DetectionUtils import crop_with_margin
from lego_sorter_server.analysis.detection.LegoLabeler import LegoLabeler
from lego_sorter_server.images.queue.ImageProcessingQueue import ImageProcessingQueue, CAPTURE_TAG
from lego_sorter_server.images.storage.LegoImageStorage import LegoImageStorage


class LegoDetectionRunner:
    DETECTION_SCORE_THRESHOLD = 0.5

    def __init__(self, queue: ImageProcessingQueue, store: LegoImageStorage):
        self.queue = queue
        self.analysis_service = AnalysisService()
        self.storage = store
        # queue, detector and storage are not thread safe, so it limits the number of workers to one
        self.executor = futures.ThreadPoolExecutor(max_workers=1)
        logging.info("[LegoDetectionRunner] Ready for processing the queue.")

    def start_detecting(self):
        logging.info("[LegoDetectionRunner] Started processing the queue.")
        return self.executor.submit(self._exception_handler, self._process_queue)

    def stop_detecting(self):
        logging.info("[LegoDetectionRunner] Processing is being terminated.")
        self.executor.shutdown()

    @staticmethod
    def _exception_handler(method, args=[]):
        try:
            method(*args)
        except Exception as exc:
            logging.exception(f"[LegoDetectionRunner] Got an exception:\n {str(exc)}")
            raise exc

    def _process_queue(self, save_cropped_image=True, save_label_file=False):
        polling_rate = 0.2  # in seconds
        logging_rate = 30  # in seconds
        logging_counter = 0

        while True:
            if self.queue.len(CAPTURE_TAG) == 0:
                if polling_rate * logging_counter >= logging_rate:
                    logging.info("[LegoDetectionRunner] Queue is empty. Waiting... ")
                    logging_counter = 0
                else:
                    logging_counter += 1
                time.sleep(polling_rate)
                continue

            logging.info("[LegoDetectionRunner] Queue not empty - processing data")
            self.__process_next_image(save_cropped_image, save_label_file)

    def __process_next_image(self, save_cropped_image, save_label_file):
        image, lego_class = self.queue.next(CAPTURE_TAG)
        prefix = self._get_random_hash() + "_"
        detection_results = self.analysis_service.detect(image)
        detected_counter = 0
        bbs = []
        for i in range(len(detection_results.detection_classes)):
            if detection_results.detection_scores[i] < LegoDetectionRunner.DETECTION_SCORE_THRESHOLD:
                continue

            detected_counter += 1
            bbs.append(detection_results.detection_boxes[i])

            if save_cropped_image is True:
                image_new = crop_with_margin(image, *detection_results.detection_boxes[i])
                self.storage.save_image(image_new, lego_class, prefix)

        prefix = f'{detected_counter}_{prefix}'
        filename = self.storage.save_image(image, f'original_{lego_class}', prefix)

        if save_label_file is True:
            path = self.storage.find_image_path(filename)
            width, height = image.size
            label_file = LegoLabeler().to_label_file(filename, str(path), width, height, bbs)
            xml_path = path.parent.absolute() / (filename.split(".")[-2] + ".xml")
            with open(xml_path, "w") as label_xml:
                label_xml.write(label_file)

    @staticmethod
    def _get_random_hash(length=4):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
