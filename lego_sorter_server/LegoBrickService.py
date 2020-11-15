from lego_sorter_server.generated import LegoBrick_pb2_grpc
from lego_sorter_server.generated.LegoBrick_pb2 import Image as LegoImage, Empty

from PIL import Image
from io import BytesIO
from datetime import datetime


class LegoBrickService(LegoBrick_pb2_grpc.LegoBrickServicer):

    # Support rate limiting - we don't want a flood of images
    def RecognizeLegoBrickInImage(self, request: LegoImage, context):
        # TODO Add service for processing images
        image = Image.open(BytesIO(request.image))

        if request.rotation == 90:
            image = image.transpose(Image.ROTATE_270)
        if request.rotation == 180:
            image = image.rotate(180)
        if request.rotation == 270:
            image = image.transpose(Image.ROTATE_90)

        # TODO Detect lego bricks and tag an image
        image.save("./images/image_{}.jpg".format(datetime.now()))

        return Empty()
