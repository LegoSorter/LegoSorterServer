from concurrent import futures
from threading import Event

from lego_sorter_server.analysis.detection.LegoDetectionRunner import LegoDetectionRunner
from lego_sorter_server.generated import LegoCapture_pb2_grpc
from lego_sorter_server.generated.Messages_pb2 import Empty
from lego_sorter_server.generated.LegoCapture_pb2 import ImageStore

from lego_sorter_server.images.queue.ImageProcessingQueue import ImageProcessingQueue, CAPTURE_TAG
from lego_sorter_server.images.storage.LegoImageStorage import LegoImageStorage
from lego_sorter_server.service.ImageProtoUtils import ImageProtoUtils


class LegoCaptureService(LegoCapture_pb2_grpc.LegoCaptureServicer):
    def __init__(self, event: Event):
        self.storage = LegoImageStorage()
        self.executor = futures.ThreadPoolExecutor(max_workers=1)
        self.processing_queue = ImageProcessingQueue()
        self.detection_runner = LegoDetectionRunner(self.processing_queue, self.storage, event)
        self.detection_runner.start_detecting()

    def CollectCroppedImages(self, request: ImageStore, context) -> Empty:
        self.executor.submit(self._handle_collect_cropped_images, request)

        return Empty()

    def CollectImages(self, request: ImageStore, context) -> Empty:
        image = ImageProtoUtils.prepare_image(request)
        self.storage.save_image(image, request.label, "unprocessed")

        return Empty()

    def _handle_collect_cropped_images(self, request: ImageStore):
        image = ImageProtoUtils.prepare_image(request)
        self.processing_queue.add(CAPTURE_TAG, image, request.label)
