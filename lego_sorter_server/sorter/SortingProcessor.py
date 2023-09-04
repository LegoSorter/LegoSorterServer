import logging
import time
import config

from typing import List, Tuple
from PIL.Image import Image
from multiprocessing import Queue, Process
from datetime import datetime

from lego_sorter_server.analysis.AnalysisService import AnalysisService
from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.detection import DetectionUtils
from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults
from lego_sorter_server.images.storage.LegoImageStorage import LegoImageStorage
from lego_sorter_server.images.storage.LegoImageSave import LegoImageSave
from lego_sorter_server.service.BrickCategoryConfig import BrickCategoryConfig
from lego_sorter_server.sorter.LegoSorterController import LegoSorterController
from lego_sorter_server.sorter.ordering.SimpleOrdering import SimpleOrdering


class SortingProcessor:
    #TODO: move this variable to the separate config file
    CLASSIFICATION_IN_ROW_MIN_COUNT = config.CLASSIFICATION_IN_ROW_MIN_COUNT
    CLASSIFICATION_BRICK_COUNT = config.CLASSIFICATION_IN_ROW_MIN_COUNT


    def __init__(self, brickCategoryConfig: BrickCategoryConfig):
        self.image_queue: Queue = Queue()
        self.running = False
        self.paused = False
        self.same_class_in_row_counter = 0
        self.brick_class_check_counter = 0
        self.prev_classification_result: ClassificationResults = None
        self.brick_classification_result = []
        self.brick_classification_result_counter = []
        self.brick_classification_id = -1
        self.ordering: SimpleOrdering = SimpleOrdering()

        self.queue_handler_process: Process = Process(target=self._image_queue_handler_service, args=(), daemon=False)
        self.queue_handler_process.start()

        self.analysis_service: AnalysisService = AnalysisService()
        self.sorter_controller: LegoSorterController = LegoSorterController(brickCategoryConfig)
        self.storage: LegoImageStorage = LegoImageStorage()
        

    # Function which will be used exclusively by the additional process, to handle images that will be added from other sources
    def _image_queue_handler_service(self):
        self.analysis_service: AnalysisService = AnalysisService()

        while True:
            if self.paused:
                time.sleep(0.01)
                pass

            image = self.image_queue.get()
            logging.info(f"[SortingProcessor] Image was taken from the queue.")
            
            #TODO: add some code which will analyze how many images are held in the queue compared to the last time and some set limiter and then clear out some of the images in queue if needed

            if self.running:
                current_state = self.process_next_image_continuously(image)

    def _clear_image_queue(self):
        while not self.image_queue.empty():
            self.image_queue.get()

    def queue_next_image(self, image: Image):
        self.image_queue.put(image)
        logging.info(f"[SortingProcessor] New image was added to the queue.")

    def save_detected_image(self, image: Image):
        start_time_saving = time.time()
        #time_prefix = f"{int(start_time_saving * 10000) % 10000}"  # 10 seconds
        time_prefix = datetime.now().strftime("%d-%m-%Y_%H-%M-%S_%f_")[:-3]
        for key, value in self.ordering.get_current_state().items():
            bounding_box = value[0]
            cropped_image = DetectionUtils.crop_with_margin(image, *bounding_box)
            #self.storage.save_image(cropped_image, str(key), time_prefix)
            self.storage.save_image_with_results(cropped_image, str(key), str(self.brick_classification_id), str(value[1]), str(value[2]), time_prefix)
        #self.storage.save_image(image, "original_sorter", time_prefix)
        logging.info(f"[SortingProcessor] Saving images took {1000 * (time.time() - start_time_saving)} ms.")


    def process_next_image_continuously(self, image: Image, save_image: bool = True):
        start_time = time.time()
        current_results = self._process_continuously(image)
        elapsed_ms = 1000 * (time.time() - start_time)
        
        logging.info(f"[SortingProcessor] Processing an image took {elapsed_ms} ms.")

        self.ordering.process_current_results(current_results, image_height=image.height, border_image_excluded=True)

        if save_image is True and len(current_results) > 0:
            self.save_detected_image(image)

        #while self.ordering.get_count_of_results_to_send() > 0 and self.same_class_in_row_counter == self.CLASSIFICATION_IN_ROW_MIN_COUNT:
        while self.ordering.get_count_of_results_to_send() > 0 and self.brick_class_check_counter == self.CLASSIFICATION_BRICK_COUNT:
            # Clear out the queue of processed bricks
            final_label_index = self.brick_classification_result_counter.index(max(self.brick_classification_result_counter))

            self.storage.set_json_save_data_final_label(str(self.brick_classification_id), str(self.brick_classification_result[final_label_index]))
            self.storage.save_images_results_to_json()
            self._send_results_to_controller()

        return self.ordering.get_current_state()

    def process_next_image(self, image: Image, save_image: bool = True):
        start_time = time.time()
        current_results = self._process(image)
        elapsed_ms = 1000 * (time.time() - start_time)

        logging.info(f"[SortingProcessor] Processing an image took {elapsed_ms} ms.")

        self.ordering.process_current_results(current_results, image_height=image.height)

        if save_image is True and len(current_results) > 0:
            self.save_detected_image(image)
            self.storage.set_json_save_data_final_label(str(self.brick_classification_id))
            self.storage.save_images_results_to_json()

        while self.ordering.get_count_of_results_to_send() > 0:
            # Clear out the queue of processed bricks
            self._send_results_to_controller()

        return self.ordering.get_current_state()

    def _send_results_to_controller(self):
        processed_brick = self.ordering.pop_first_processed_brick()

        if len(processed_brick) == 0:
            return False

        best_result = self.get_best_result(processed_brick)
        logging.info(f"[SortingProcessor] Got the best result {best_result}. Returning the results...")
        self.sorter_controller.on_brick_recognized(best_result)

    def _process_continuously(self, image: Image) -> List[Tuple]:
        """
        Returns a list of recognized bricks ordered by the position on the belt - ymin desc
        """
        # Detect bricks on image
        start_time = time.time()
        detection_results = self.analysis_service.detect(image, threshold=0.8)
        elapsed_ms = 1000 * (time.time() - start_time)
        logging.info(f"[SortingProcessor] Detecting objects on image took {elapsed_ms} ms.")

        # Check if any bricks were detected
        detected_count = len(detection_results.detection_classes)
        if detected_count == 0:
            return []

        logging.info(f"[SortingProcessor] Detected a lego brick, processing...")

        if detected_count > 1:
            logging.warning(f"[SortingProcessor] More than one brick detected '(detected_count = {detected_count}), "
                            f"there should be only one brick on the tape at the same time.")

        # Zipping detected bricks data to single list
        zipped_detection_results = list(zip(detection_results.detection_boxes, detection_results.detection_classes, detection_results.detection_scores))

        # Sort detected bricks by the position on the conveyor belt
        zipped_detection_results = self.order_by_bounding_box_position(zipped_detection_results)

        # Filter out bricks that are too close to the borders
        zipped_detection_results = self.ordering.discard_border_results(zipped_detection_results, image.height)

        # Check if any bricks were detected after filtering
        detected_count = len(zipped_detection_results)
        if detected_count == 0:
            return []

        # Check if new brick was detected and previous brick dissapeared
        new_brick = self.ordering.detect_new_brick_appearence(zipped_detection_results, image_height=image.height, border_image_excluded=True)
        if new_brick:
            self.brick_class_check_counter = 0
            self.brick_classification_result.clear()
            self.brick_classification_result_counter.clear()
            self.brick_classification_id = round(time.time() * 1000)


        # Check if current brick was already matched with certain class
        if self.brick_class_check_counter <= self.CLASSIFICATION_IN_ROW_MIN_COUNT:
            # Prepare smaller images consisting of detected bricks for classification
            cropped_images = []
            for object in zipped_detection_results:
                bounding_box = object[0]
                cropped_image = DetectionUtils.crop_with_margin_from_bb(image, bounding_box)
                cropped_images.append(cropped_image)

            # Classify detected bricks
            start_time = time.time()
            classification_results = self.analysis_service.classify(cropped_images)
            elapsed_ms = 1000 * (time.time() - start_time)
            logging.info(f"[SortingProcessor] Classifying bricks on image took {elapsed_ms} ms.")
        else:
            # Use previously found classification
            classification_results = self.brick_classification_result[-1]
            logging.info(f"[SortingProcessor] Skipped classyfiying bricks on the image.")

        # Prepare data that will be returned by function
        detection_boxes = (zip(*zipped_detection_results))[0]
        zipped_results = list(zip(detection_boxes,
                                  classification_results.classification_classes,
                                  classification_results.classification_scores))

        # Count how many times in a row first brick was classified as the same class
        first_detected_brick = self.ordering.get_first_detected_brick(zipped_results, image_height=image.height)
        if first_detected_brick != []:
            brick_class = first_detected_brick[1][0]
            if self.brick_classification_result == [] or brick_class not in self.brick_classification_result:
                #self.same_class_in_row_counter = 1
                self.brick_classification_result.append(brick_class)
                self.brick_classification_result_counter.append(1)
            else:
                self.brick_classification_result_counter[self.brick_classification_result.index(brick_class)] += 1

        return zipped_results

    def _process(self, image: Image) -> List[Tuple]:
        """
        Returns a list of recognized bricks ordered by the position on the belt - ymin desc
        """
        self.brick_classification_id = round(time.time() * 1000)
        results = self.analysis_service.detect_and_classify(image, detection_threshold=0.8)

        detected_count = len(results[0].detection_classes)
        if detected_count == 0:
            return []

        logging.info(f"[SortingProcessor] Detected a lego brick, processing...")

        if detected_count > 1:
            logging.warning(f"[SortingProcessor] More than one brick detected '(detected_count = {detected_count}), "
                            f"there should be only one brick on the tape at the same time.")

        zipped_results = list(zip(results[0].detection_boxes,
                                  results[1].classification_classes,
                                  results[1].classification_scores))

        return self.order_by_bounding_box_position(zipped_results)

    def start_machine(self):
        self.sorter_controller.run_conveyor()
        self.running = True
        self.paused = False

    def pause_machine(self):
        self.sorter_controller.stop_conveyor()
        self.running = False
        self.paused = True

    def stop_machine(self):
        self.sorter_controller.stop_conveyor()
        self.running = False
        self.paused = False
        # Reset machine variables
        self.same_class_in_row_counter = 0
        self.prev_classification_result = None
        self.brick_class_check_counter = 0
        self.brick_classification_result.clear()
        self.brick_classification_result_counter.clear()
        self.brick_classification_id = -1

    def set_machine_speed(self, speed: int):
        self.sorter_controller.set_machine_speed(speed)

    @staticmethod
    def order_by_bounding_box_position(zipped_results: List[Tuple[Tuple, str, float]]) -> List[Tuple]:
        # sort by ymin
        return sorted(zipped_results, key=lambda res: res[0][0], reverse=True)

    @staticmethod
    def get_best_result(results):
        # TODO - max score, average score, max count?
        return results[0]