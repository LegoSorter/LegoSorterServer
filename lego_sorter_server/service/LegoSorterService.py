from typing import List

from lego_sorter_server.generated import LegoSorter_pb2_grpc
from lego_sorter_server.generated.LegoSorter_pb2 import ImageRequest, SorterConfiguration, BoundingBox
from lego_sorter_server.service.ImageProtoUtils import ImageProtoUtils
from lego_sorter_server.sorter.SortingProcessor import SortingProcessor


class LegoSorterService(LegoSorter_pb2_grpc.LegoSorterServicer):

    def __init__(self):
        self.sortingProcessor = SortingProcessor()

    def processNextImage(self, request: ImageRequest, context) -> List[BoundingBox]:
        image = ImageProtoUtils.prepare_image(request)
        results = self.sortingProcessor.process_next_image(image)



    def getConfiguration(self, request, context) -> SorterConfiguration:
        return super().getConfiguration(request, context)

    def updateConfiguration(self, request: SorterConfiguration, context) -> SorterConfiguration:
        return super().updateConfiguration(request, context)
