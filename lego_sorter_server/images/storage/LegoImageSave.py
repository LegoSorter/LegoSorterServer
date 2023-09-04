import logging
import time
import config

from PIL.Image import Image
from multiprocessing import Queue, Process

from lego_sorter_server.analysis.AnalysisService import AnalysisService
from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.detection import DetectionUtils
from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults
from lego_sorter_server.images.storage.LegoImageStorage import LegoImageStorage
from lego_sorter_server.service.BrickCategoryConfig import BrickCategoryConfig
from lego_sorter_server.sorter.LegoSorterController import LegoSorterController
from lego_sorter_server.sorter.ordering.SimpleOrdering import SimpleOrdering

class LegoImageSave:
    
    def __init__(self):
        self.image_queue: Queue = Queue()
        self.paused = False
        self.queue_handler_process: Process = Process(target=self._save_image_handler_service, args=(), daemon=False)
        self.queue_handler_process.start()
        
        self.ordering: SimpleOrdering = SimpleOrdering()
        self.storage: LegoImageStorage = LegoImageStorage()

    def _clear_save_image_queue(self):
        while not self.image_queue.empty():
            self.image_queue.get()
        
    def _save_image_handler_service(self):
        logging.info(f"[LegoImageSave] Started process")
        while True:
            if self.paused:
                time.sleep(0.01)
                pass

            image = self.image_queue.get()
            logging.info(f"[LegoImageSave] Image was taken from the queue. Saving...")
            
            #TODO: add some code which will analyze how many images are held in the queue compared to the last time and some set limiter and then clear out some of the images in queue if needed

            start_time_saving = time.time()
            time_prefix = f"{int(start_time_saving * 10000) % 10000}"  # 10 seconds
            for key, value in self.ordering.get_current_state().items():
                bounding_box = value[0]
                cropped_image = DetectionUtils.crop_with_margin(image, *bounding_box)
                self.storage.save_image(cropped_image, str(key)+"TEST", time_prefix)
            self.storage.save_image(image, "original_sorter_TEST", time_prefix)
            logging.info(f"[SortingProcessor] Saving images took {1000 * (time.time() - start_time_saving)} ms.")