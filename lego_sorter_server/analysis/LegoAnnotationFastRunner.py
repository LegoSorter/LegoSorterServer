import time
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import Event

from lego_sorter_server.database import Models
from lego_sorter_server.database.Database import SessionLocal
from lego_sorter_server.images.queue.ImageAnnotationQueueFast import ImageAnnotationQueueFast, CAPTURE_TAG


class LegoAnnotationFastRunner:
    DETECTION_SCORE_THRESHOLD = 0.5

    def __init__(self, storage_queue: ImageAnnotationQueueFast, annotation_fast_runer_executor: ThreadPoolExecutor, event: Event):
        self.storage_queue = storage_queue
        self.event = event
        # self.storage = store

        # queue, detector and storage are not thread safe, so it limits the number of workers to one
        self.executor = annotation_fast_runer_executor
        # self.executor = futures.ThreadPoolExecutor(max_workers=4)
        logger.info("[LegoAnnotationFastRunner] Ready for processing the queue.")

    def start_detecting(self):
        logger.info("[LegoAnnotationFastRunner] Started processing the queue.")
        return self.executor.submit(self._exception_handler, self._process_queue)

    def stop_detecting(self):
        logger.info("[LegoAnnotationFastRunner] Processing is being terminated.")
        self.event.set()
        self.executor.shutdown()

    @staticmethod
    def _exception_handler(method, args=[]):
        try:
            method(*args)
        except Exception as exc:
            logger.exception(f"[LegoAnnotationFastRunner] Got an exception:\n {str(exc)}")
            raise exc

    def _process_queue(self):
        polling_rate = 0.2  # in seconds
        logging_rate = 30  # in seconds
        logging_counter = 0
        db = SessionLocal()
        annotation_fast_runer_executor_max_workers = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "annotation_fast_runer_executor_max_workers").first()
        db.close()
        limit = (int(annotation_fast_runer_executor_max_workers.value) - 1) * 2
        futures = set()

        while True:
            if self.event.isSet():
                logger.info("[LegoAnnotationFastRunner] Processing queue is being terminated.")
                return
            if self.storage_queue.len(CAPTURE_TAG) == 0:
                if polling_rate * logging_counter >= logging_rate:
                    logger.info("[LegoAnnotationFastRunner] Queue is empty. Waiting... ")
                    logging_counter = 0
                else:
                    logging_counter += 1
                time.sleep(polling_rate)
                continue

            # logger.info("[LegoAnnotationFastRunner] Queue not empty - processing data")
            logger.info(f"[LegoAnnotationFastRunner] Queue length:{self.storage_queue.len(CAPTURE_TAG)}")

            if annotation_fast_runer_executor_max_workers.value == "1":
                detection_results, classification_results, imageid = self.storage_queue.next(CAPTURE_TAG)
                self.__process_next_image(detection_results, classification_results, imageid)
            else:
                if len(futures) >= limit:
                    completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                detection_results, classification_results, imageid = self.storage_queue.next(CAPTURE_TAG)
                futures.add( self.executor.submit(self.__process_next_image,detection_results, classification_results, imageid))

    def __process_next_image(self, detection_results, classification_results, imageid):
        start_time = time.time()
        # detectionResults, classificationResults, image = self.storage_queue.next(CAPTURE_TAG)

        objects = ""

        db = SessionLocal()
        image = db.query(Models.DBImage).get(imageid)

        for i in range(len(detection_results.detection_boxes)):
            if detection_results.detection_scores[i] < self.DETECTION_SCORE_THRESHOLD:
                continue

            y_min, x_min, y_max, x_max = [int(coord) for coord in detection_results.detection_boxes[i]]
            score = classification_results.classification_scores[i]
            label = classification_results.classification_classes[i]
            image_result = Models.DBImageResult(owner=image, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, score=score, label=label)
            db.add(image_result)
            # image_result = DBImageResult.create(owner=image,x_min=x_min,y_min=y_min,x_max=x_max,y_max=y_max,score=score,label=label)
            objects += f"""
    <object>
        <name>{label}</name>
        <pose></pose>
        <truncated></truncated>
        <difficult></difficult>
        <bndbox>
            <xmin>{int(x_min)}</xmin>
            <ymin>{int(y_min)}</ymin>
            <xmax>{int(x_max)}</xmax>
            <ymax>{int(y_max)}</ymax>
        </bndbox>
        <confidence>{score}</confidence>
    </object>"""

        voc =   f"""<annotation>
    <folder></folder>
    <filename>{image.filename}</filename>
    <path>{image.owner.path}/{image.filename}</path>
    <source>
        <database></database>
    </source>
    <size>
        <width>{image.image_width}</width>
        <height>{image.image_height}</height>
        <depth>3</depth>
    </size>
    <segmented></segmented>
{objects}
</annotation>"""
        xml_path = image.owner.path + "/" + (image.filename.split(".")[-2] + ".xml")  # TODO use os.path.join
        with open(xml_path, "w") as label_xml:
            label_xml.write(voc)
        # db = SessionLocal()
        image.VOC_exist = True
        db.commit()
        db.close()
        # image.save()

        elapsed_millis = (time.time() - start_time) * 1000
        logger.info(f"[LegoAnnotationFastRunner] Request processed in {elapsed_millis} ms.")
