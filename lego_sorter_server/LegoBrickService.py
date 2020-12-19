from generated import LegoBrick_pb2_grpc
from generated.LegoBrick_pb2 import Image as LegoImage, Empty, ImageStore as LegoImageStore, BoundingBox

from PIL import Image
from io import BytesIO
from detection.LegoDetector import LegoDetector
import numpy as np

from lego_sorter_server.detection import DetectionUtils
from lego_sorter_server.detection.LegoDetectionRunner import LegoDetectionRunner
from lego_sorter_server.images.queue.ImageProcessingQueue import ImageProcessingQueue
from lego_sorter_server.images.storage.LegoImageStorage import LegoImageStorage


class LegoBrickService(LegoBrick_pb2_grpc.LegoBrickServicer):
    def __init__(self):
        self.detector = LegoDetector()
        self.storage = LegoImageStorage()
        self.processing_queue = ImageProcessingQueue()
        self.detection_runner = LegoDetectionRunner(self.processing_queue, self.detector, self.storage)

        self.detector.__initialize__()
        self.detection_runner.start_detecting()

    @staticmethod
    def _prepare_image(request: LegoImage):
        image = Image.open(BytesIO(request.image))
        image = image.convert('RGB')

        if request.rotation == 90:
            image = image.transpose(Image.ROTATE_270)
        elif request.rotation == 180:
            image = image.rotate(180)
        elif request.rotation == 270:
            image = image.transpose(Image.ROTATE_90)

        return image

    def RecognizeLegoBrickInImage(self, request: LegoImage, context):
        image = self._prepare_image(request)
        self.storage.save_image(image, "unknown")

        return Empty()

    def CollectCroppedImages(self, request: LegoImageStore, context):
        image = self._prepare_image(request)

        self.processing_queue.add(image, request.label)

        return Empty()

    def CollectImages(self, request: LegoImageStore, context):
        image = self._prepare_image(request)
        self.storage.save_image(image, "unprocessed_" + request.label)

        return Empty()

    def DetectBricks(self, request: BoundingBox, context):
        image = Image.open(BytesIO(request.image))
        image = image.convert('RGB')
        image_resized, scale = DetectionUtils.resize(image, 640)  # image.resize((640, 640), 0)
        detections = self.detector.detect_lego(np.array(image_resized))
        # TODO: selecting a single bb... for now
        # best_detection = np.argmax(detections['detection_scores'])
        bb = BoundingBox()
        bb.ymin, bb.xmin, bb.ymax, bb.xmax = [int(i * 640 * 1 / scale) for i in detections['detection_boxes'][0]]

        bb.score = detections['detection_scores'][0]
        bb.label = 'DUMMY_LABEL'
        return bb
