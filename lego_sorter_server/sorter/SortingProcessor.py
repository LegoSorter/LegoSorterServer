import logging
import time
import config

from typing import List, Tuple
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from multiprocessing import Queue, Process, Value
from datetime import datetime
import os
import json

from lego_sorter_server.analysis.AnalysisService import AnalysisService
from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.detection import DetectionUtils
from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults
from lego_sorter_server.images.storage.LegoImageStorage import LegoImageStorage
from lego_sorter_server.images.storage.LegoImageSave import LegoImageSave
from lego_sorter_server.service.BrickCategoryConfig import BrickCategoryConfig
from lego_sorter_server.sorter.LegoSorterController import LegoSorterController
from lego_sorter_server.sorter.ordering.SimpleOrdering import SimpleOrdering

QUEUE_TIMEOUT_VALUE = 0.01
QUEUE_MAX_SIZE = 1000

class SortingProcessor:
    CLASSIFICATION_IN_ROW_MIN_COUNT = config.CLASSIFICATION_IN_ROW_MIN_COUNT
    CLASSIFICATION_BRICK_COUNT = config.CLASSIFICATION_IN_ROW_MIN_COUNT


    def __init__(self, brickCategoryConfig: BrickCategoryConfig):
        self.image_queue: Queue = Queue(QUEUE_MAX_SIZE)
        self.running = Value('i', 0)
        self.paused = Value('i', 0)
        self.brick_class_check_counter = Value('i', 0)
        self.brick_class_result = []
        self.brick_classification_result = []
        self.brick_classification_result_counter = []
        self.brick_classification_id = -1
        self.ordering: SimpleOrdering = SimpleOrdering()
        self.storage: LegoImageStorage = LegoImageStorage()
        self.sorter_controller: LegoSorterController = LegoSorterController(brickCategoryConfig)

        self.image_detection_queue: Queue = Queue(QUEUE_MAX_SIZE)
        self.image_crop_queue: Queue = Queue(QUEUE_MAX_SIZE)
        self.image_classification_queue: Queue = Queue(QUEUE_MAX_SIZE)
        self.image_tape_queue: Queue = Queue(QUEUE_MAX_SIZE)

        # Base process for continous mode - replaced by the observer process
        #self.image_queue_handler_process: Process = Process(target=self._image_queue_handler_service, args=(), daemon=False)
        #self.image_queue_handler_process.start()

        # Observer process which watches if there's any data queued for processing
        self.image_queue_provider_handler_process: Process = Process(target=self._image_queue_provider_handler_service, args=(), daemon=False)
        self.image_queue_provider_handler_process.start()

        # Different stages of parallel image processing using pipeline parallelism
        self.image_detection_queue_handler_process: Process = Process(target=self._image_detection_queue_handler_service, args=(), daemon=False)
        self.image_detection_queue_handler_process.start()
        self.image_crop_queue_handler_process: Process = Process(target=self._image_crop_queue_handler_service, args=(), daemon=False)
        self.image_crop_queue_handler_process.start()
        self.image_classification_queue_handler_process: Process = Process(target=self._image_classification_queue_handler_service, args=(), daemon=False)
        self.image_classification_queue_handler_process.start()
        self.image_tape_queue_handler_process: Process = Process(target=self._image_tape_queue_handler_service, args=(), daemon=False)
        self.image_tape_queue_handler_process.start()


    def __getstate__(self):
        # Capture what is normally pickled
        state = self.__dict__.copy()

        # Remove unpicklable/problematic variables 
        state['image_queue_provider_handler_process'] = None
        state['image_detection_queue_handler_process'] = None
        state['image_crop_queue_handler_process'] = None
        state['image_classification_queue_handler_process'] = None
        state['image_tape_queue_handler_process'] = None

        return state
        

    # Function which will be used exclusively by the additional process, to handle images that will be added from other sources
    def _image_queue_handler_service(self):
        self.analysis_service: AnalysisService = AnalysisService()
    
        while True:
            if bool(self.paused.value):
                time.sleep(0.01)
                pass
        
            image = self.image_queue.get()
            logging.info(f"[SortingProcessor] Image was taken from the queue.")
        
            if bool(self.running.value):
                current_state = self.process_next_image_continuously(image)


    def _image_queue_provider_handler_service(self):
        while True:
            if bool(self.paused.value):
                time.sleep(0.01)
                pass

            image = self.image_queue.get()
            logging.info(f"[SortingProcessor] Image was taken from the queue.")
            print(f"[SortingProcessor] Image was taken from the queue.")

            if bool(self.running.value):
                success = False
                while not success:
                    try:
                        if self.image_detection_queue.qsize() < QUEUE_MAX_SIZE:
                            self.image_detection_queue.put((image))
                            success = True
                        else:
                            time.sleep(QUEUE_TIMEOUT_VALUE)
                    except:
                        time.sleep(QUEUE_TIMEOUT_VALUE)


    def _image_detection_queue_handler_service(self):
        self.analysis_service: AnalysisService = AnalysisService()

        while True:
            if bool(self.paused.value):
                time.sleep(0.01)
                pass

            image = self.image_detection_queue.get()
            logging.info(f"[SortingProcessorDetection] Image was taken from the queue.")
    
            # Detect bricks on image
            start_detection_time = time.time()
            detection_results = self.analysis_service.detect(image, threshold=0.8)
            elapsed_ms = 1000 * (time.time() - start_detection_time)
            logging.info(f"[SortingProcessor] Detecting objects on image took {elapsed_ms} ms.")

            self.image_crop_queue.put((detection_results, image))


    def _image_crop_queue_handler_service(self):
        while True:
            if bool(self.paused.value):
                time.sleep(0.01)
                pass
    
            detection_results, image = self.image_crop_queue.get()
            logging.info(f"[SortingProcessorCrop] Image was taken from the queue.")
    
            # Check if any bricks were detected
            detected_count = len(detection_results.detection_classes)
            if detected_count == 0:
                continue

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
                continue
            
            # Prepare smaller images consisting of detected bricks for classification
            cropped_images = []
            for object in zipped_detection_results:
                bounding_box = object[0]
                cropped_image = DetectionUtils.crop_with_margin_from_bb(image, bounding_box)
                cropped_images.append(cropped_image)

            self.image_classification_queue.put((cropped_images, zipped_detection_results, image))


    def _image_classification_queue_handler_service(self):
        self.analysis_service: AnalysisService = AnalysisService()
    
        while True:
            if bool(self.paused.value):
                time.sleep(0.01)
                pass
    
            cropped_images, zipped_detection_results, image = self.image_classification_queue.get()
            logging.info(f"[SortingProcessorClassification] Image was taken from the queue.")
    
            # Classify detected bricks
            start_classification_time = time.time()
            classification_results = self.analysis_service.classify(cropped_images)
            elapsed_ms = 1000 * (time.time() - start_classification_time)
            logging.info(f"[SortingProcessor] Classifying bricks on image took {elapsed_ms} ms.")

            self.image_tape_queue.put((classification_results, zipped_detection_results, image))


    def _image_tape_queue_handler_service(self, save_image: bool = True):
        brick_class_list_result = []

        while True:
            if bool(self.paused.value):
                time.sleep(0.01)
                pass
    
            classification_results, zipped_detection_results, image = self.image_tape_queue.get()

            # Check if new brick was detected and previous brick dissapeared
            new_brick = self.ordering.detect_new_brick_appearence(zipped_detection_results, image_height=image.height, border_image_excluded=True)
            if new_brick:
                self.brick_class_check_counter.value = 1
                self.brick_classification_result.clear()
                self.brick_classification_result_counter.clear()
                brick_class_list_result.clear()
                self.brick_classification_id = round(time.time() * 1000)
            else:
                self.brick_class_check_counter.value += 1

            logging.info(f"[SortingProcessorTape] Image was taken from the queue.")

            # Prepare data that will be returned by function
            detection_boxes = list(zip(*(zipped_detection_results)))[0]
            zipped_results = list(zip(detection_boxes,
                                    classification_results.classification_classes,
                                    classification_results.classification_scores))
    
            # Count how many times in a row first brick was classified as the same class
            first_detected_brick = self.ordering.get_first_detected_brick(zipped_results, image_height=image.height)
            if first_detected_brick != []:
                brick_class = first_detected_brick[1]
                if brick_class_list_result == [] or brick_class not in brick_class_list_result:
                    brick_class_list_result.append(brick_class)
                    self.brick_classification_result.append(classification_results)
                    self.brick_classification_result_counter.append(1)
                else:
                    self.brick_classification_result_counter[brick_class_list_result.index(brick_class)] += 1

            
            #current_results = zipped_results
            current_results = []
            for i in range(len(detection_boxes)):
                for j in range(len(classification_results.classification_classes)):
                    current_results.append((detection_boxes[i], classification_results.classification_classes[j], classification_results.classification_scores[j]))

            new_brick = self.ordering.detect_new_brick_appearence(zipped_detection_results, image_height=image.height, border_image_excluded=True)

            self.ordering.process_current_results(current_results, image_height=image.height, border_image_excluded=True)

            if save_image is True and len(current_results) > 0:
                self.save_detected_image(image)

            image.close()
            
            while self.ordering.get_count_of_results_to_send() > 0 and (self.brick_class_check_counter.value >= self.CLASSIFICATION_BRICK_COUNT or new_brick):
                # Clear out the queue of processed bricks
                final_label_index = self.brick_classification_result_counter.index(max(self.brick_classification_result_counter))
                self.storage.set_json_save_data_final_label(str(self.brick_classification_id), str(brick_class_list_result[final_label_index]))
                self.storage.save_images_results_to_json()
                self._send_results_to_controller()


    def _clear_image_queue(self):
        while not self.image_queue.empty():
            self.image_queue.get()

    def queue_next_image(self, image: Image):
        self.image_queue.put(image)
        logging.info(f"[SortingProcessor] New image was added to the queue.")

    def save_detected_image(self, image: Image):
        start_time_saving = time.time()
        #time_prefix = f"{int(start_time_saving * 10000) % 10000}"  # 10 seconds
        time_prefix = datetime.now().strftime("%d-%m-%Y_%H-%M-%S-%f")[:-3] + '_'
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

        while self.ordering.get_count_of_results_to_send() > 0 and self.brick_class_check_counter.value >= self.CLASSIFICATION_BRICK_COUNT:
            # Clear out the queue of processed bricks
            final_label_index = self.brick_classification_result_counter.index(max(self.brick_classification_result_counter))

            self.storage.set_json_save_data_final_label(str(self.brick_classification_id), str(self.brick_class_result[final_label_index]))
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
            self.brick_class_check_counter.value = 1
            self.brick_class_result.clear()
            self.brick_classification_result.clear()
            self.brick_classification_result_counter.clear()
            self.brick_classification_id = round(time.time() * 1000)
        else:
            self.brick_class_check_counter.value += 1


        # Check if current brick was already matched with certain class
        if self.brick_class_check_counter.value <= self.CLASSIFICATION_IN_ROW_MIN_COUNT:
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
        detection_boxes = list(zip(*(zipped_detection_results)))[0]
        zipped_results = list(zip(detection_boxes,
                                  classification_results.classification_classes,
                                  classification_results.classification_scores))

        # Count how many times in a row first brick was classified as the same class
        first_detected_brick = self.ordering.get_first_detected_brick(zipped_results, image_height=image.height)
        if first_detected_brick != []:
            brick_class = first_detected_brick[1]
            if self.brick_class_result == [] or brick_class not in self.brick_class_result:
                self.brick_class_result.append(brick_class)
                self.brick_classification_result.append(classification_results)
                self.brick_classification_result_counter.append(1)
            else:
                self.brick_classification_result_counter[self.brick_class_result.index(brick_class)] += 1

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
        self.running.value = 1
        self.paused.value = 0

    def pause_machine(self):
        self.sorter_controller.stop_conveyor()
        self.running.value = 0
        self.paused.value = 1

    def stop_machine(self):
        self.sorter_controller.stop_conveyor()
        self.running.value = 0
        self.paused.value = 0
        # Reset machine variables
        self.brick_class_check_counter.value = 0
        self.brick_class_result.clear()
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
        predicted_class = None
        parsed_data = {}
        for prediction in results:
            detection_box = prediction[0]
            predicted_class = prediction[1]
            predicted_value = prediction[2]

            if predicted_class in parsed_data:
                parsed_data[predicted_class][0] += 1
                parsed_data[predicted_class][1] += predicted_value
            else:
                parsed_data[predicted_class] = [1, predicted_value]

        max_value = 0
        for brick_name, [count, value] in parsed_data.items():
            if value > max_value:
                max_value = value
                predicted_class = brick_name

        return (None, predicted_class, None)