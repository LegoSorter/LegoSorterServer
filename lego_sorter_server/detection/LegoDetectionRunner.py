import time
from concurrent import futures

import numpy as np

from lego_sorter_server.detection import DetectionUtils
from lego_sorter_server.detection.LegoDetector import LegoDetector
from lego_sorter_server.images.queue.ImageProcessingQueue import ImageProcessingQueue
from lego_sorter_server.images.storage.LegoImageStorage import LegoImageStorage


class LegoDetectionRunner:
    def __init__(self, queue: ImageProcessingQueue, detector: LegoDetector, store: LegoImageStorage):
        self.queue = queue
        self.detector = detector
        self.storage = store
        # queue, detector and storage are not thread safe, so it limits the number of workers to one
        self.executor = futures.ThreadPoolExecutor(max_workers=1)
        print("[LegoDetectionRunner] Initialized\n")

    def start_detecting(self):
        print("[LegoDetectionRunner] Started processing the queue\n")
        self.executor.submit(self._process_queue)

    def stop_detecting(self):
        print("[LegoDetectionRunner] Processing is being terminated\n")
        self.executor.shutdown()

    def _process_queue(self):
        while True:
            if self.queue.len() == 0:
                print("Queue is empty. Waiting... ")
                time.sleep(1)
                continue
            image, lego_class = self.queue.next()
            width, height = image.size

            image_resized, scale = DetectionUtils.resize(image, 640)
            detections = self.detector.detect_lego(np.array(image_resized))

            for i in range(100):
                if detections['detection_scores'][i] < 0.5:
                    # continue # IF NOT SORTED
                    break  # IF SORTED

                ymin, xmin, ymax, xmax = [int(i * 640 * 1 / scale) for i in detections['detection_boxes'][i]]
                if ymax >= height or xmax >= width:
                    continue
                image_new = image.crop([xmin, ymin, xmax, ymax])

                self.storage.save_image(image_new, lego_class)

