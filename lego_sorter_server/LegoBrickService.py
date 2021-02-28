from concurrent import futures
from typing import List

from lego_sorter_server.classifier.LegoClassifierRunner import LegoClassifierRunner
from lego_sorter_server.detection.DetectionUtils import crop_with_margin
from lego_sorter_server.detection.detectors import LegoDetectorProvider
from lego_sorter_server.generated import LegoBrick_pb2_grpc
from lego_sorter_server.generated.LegoBrick_pb2 import Image as LegoImage, Empty, ImageStore as LegoImageStore, \
    BoundingBox, \
    ListOfBoundingBoxes

from PIL import Image
from io import BytesIO
import numpy as np

from lego_sorter_server.detection import DetectionUtils
from lego_sorter_server.detection.LegoDetectionRunner import LegoDetectionRunner
from lego_sorter_server.images.queue.ImageProcessingQueue import ImageProcessingQueue
from lego_sorter_server.images.storage.LegoImageStorage import LegoImageStorage


class LegoBrickService(LegoBrick_pb2_grpc.LegoBrickServicer):
    def __init__(self):
        self.detector = LegoDetectorProvider.get_default_detector()
        self.storage = LegoImageStorage()
        self.classifier = LegoClassifierRunner()
        # self.classifier.load_trained_model()
        self.processing_queue = ImageProcessingQueue()
        self.detection_runner = LegoDetectionRunner(self.processing_queue, self.detector, self.storage)
        self.executor = futures.ThreadPoolExecutor(max_workers=4)

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

    def CollectCroppedImages(self, request: LegoImageStore, context):
        self.executor.submit(self._handle_collect_cropped_images, request)

        return Empty()

    def _handle_collect_cropped_images(self, request: LegoImageStore):
        image = self._prepare_image(request)
        self.processing_queue.add(image, request.label)

    def CollectImages(self, request: LegoImageStore, context):
        image = self._prepare_image(request)
        self.storage.save_image(image, "unprocessed_" + request.label)

        return Empty()

    def _detect_bricks(self, request: LegoImage) -> List[BoundingBox]:
        image = self._prepare_image(request)
        width, height = image.size
        image_resized, scale = DetectionUtils.resize(image, 640)  # image.resize((640, 640), 0)
        detections = self.detector.detect_lego(np.array(image_resized))

        bbs = []
        for i in range(len(detections['detection_classes'])):
            if detections['detection_scores'][i] < 0.5:
                continue

            bb = BoundingBox()

            bb.ymin, bb.xmin, bb.ymax, bb.xmax = [int(i * 640 * 1 / scale) for i in detections['detection_boxes'][i]]
            if bb.ymax >= height or bb.xmax >= width:
                continue

            bb.score = detections['detection_scores'][i]
            bb.label = 'lego'
            bbs.append(bb)

        return bbs

    def DetectBricks(self, request: LegoImage, context):
        bbs = self._detect_bricks(request)
        bb_list = ListOfBoundingBoxes()
        bb_list.packet.extend(bbs)
        return bb_list

    def DetectAndClassifyBricks(self, request, context):
        bbs = self._detect_bricks(request)
        image = self._prepare_image(request)

        bbs_with_blobs = []

        for bb in bbs:
            cropped_brick = crop_with_margin(image, bb.ymin, bb.xmin, bb.ymax, bb.xmax)
            bbs_with_blobs.append((bb, cropped_brick))

        bbs_labels = self.classifier.predict_from_pil([img for _, img in bbs_with_blobs])
        bbs = [bb for bb, _ in bbs_with_blobs]
        for i in range(0, len(bbs)):
            bbs[i].label = bbs_labels[i]
        bb_list = ListOfBoundingBoxes()
        bb_list.packet.extend(bbs)
        return bb_list
