import io
from loguru import logger
import time
from collections import deque
import cv2
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

    # cv2
    def GetCameraPreview(self, request: Empty, context) -> ImagePreview:
        logger.info("[LegoControlService] GetCameraPreview")
        imagePreview = ImagePreview()
        if len(self.lastImages) > 1:
            image = self.lastImages.popleft()
            # imgByteArr = io.BytesIO()
            # image.save(imgByteArr, format='JPEG', quality=75)
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 75]
            imagePreview.image = cv2.imencode('.jpg', image, encode_param)[1].tobytes()
        elif len(self.lastImages) == 1:
            image = self.lastImages[0]
            # imgByteArr = io.BytesIO()
            # image.save(imgByteArr, format='JPEG', quality=75)
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 75]
            imagePreview.image = cv2.imencode('.jpg', image, encode_param)[1].tobytes()
        imagePreview.timestamp = str(int(time.time()))
        return imagePreview

    # PIL
    # def GetCameraPreview(self, request: Empty, context) -> ImagePreview:
    #     logger.info("[LegoControlService] GetCameraPreview")
    #     imagePreview = ImagePreview()
    #     if len(self.lastImages) > 1:
    #         image = self.lastImages.popleft()
    #         imgByteArr = io.BytesIO()
    #         image.save(imgByteArr, format='JPEG', quality=75)
    #         imagePreview.image = imgByteArr.getvalue()
    #     elif len(self.lastImages) == 1:
    #         image = self.lastImages[0]
    #         imgByteArr = io.BytesIO()
    #         image.save(imgByteArr, format='JPEG', quality=75)
    #         imagePreview.image = imgByteArr.getvalue()
    #     imagePreview.timestamp = str(int(time.time()))
    #     return imagePreview
