import time
from collections import deque
from concurrent import futures

from lego_sorter_server.analysis.detection.LegoDetectionRunner import LegoDetectionRunner
from lego_sorter_server.generated import LegoControl_pb2_grpc
from lego_sorter_server.generated.LegoControl_pb2 import ImagePreview
from lego_sorter_server.generated.Messages_pb2 import Empty
from lego_sorter_server.generated.LegoCapture_pb2 import ImageStore

from lego_sorter_server.images.queue.ImageProcessingQueue import ImageProcessingQueue, CAPTURE_TAG
from lego_sorter_server.images.storage.LegoImageStorage import LegoImageStorage
from lego_sorter_server.service.ImageProtoUtils import ImageProtoUtils


class LegoControlService(LegoControl_pb2_grpc.LegoControlServicer):
    def __init__(self, lastImages: deque):
        self.lastImages = lastImages
        a = 12

    def GetCameraPreview(self, request: Empty, context) -> ImagePreview:
        print("LegoAnalysisFastService start")
        imagePreview = ImagePreview()
        if len(self.lastImages) > 1:
            imagePreview.image = self.lastImages.popleft()
        elif len(self.lastImages) == 1:
            imagePreview.image = self.lastImages[0]
        imagePreview.timestamp = str(int(time.time()))
        return imagePreview
