from lego_sorter_server.generated import LegoBrick_pb2_grpc
from lego_sorter_server.generated.LegoBrick_pb2 import LegoBrickImage, Empty

from PIL import Image
from io import BytesIO


class LegoBrickServer(LegoBrick_pb2_grpc.LegoBrickServicer):
    def SendImage(self, request: LegoBrickImage, context):
        print(request.image[0])

        # TODO Add service for processing images
        image = Image.open(BytesIO(request.image))

        if request.rotation == 90:
            image = image.transpose(Image.ROTATE_270)
        if request.rotation == 180:
            image = image.rotate(180)
        if request.rotation == 270:
            image = image.transpose(Image.ROTATE_90)

        image.save()  # TODO Provide path

        return Empty()
