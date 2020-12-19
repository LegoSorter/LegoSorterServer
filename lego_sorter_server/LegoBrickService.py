from generated import LegoBrick_pb2_grpc
from generated.LegoBrick_pb2 import Image as LegoImage, Empty, ImageStore as LegoImageStore

from PIL import Image
from io import BytesIO
import time
import os

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
        if not os.path.exists('images'):
            os.makedirs('images')
        image.save(f'./images/image_{int(time.time()) }.jpg')

        return Empty()

    def CollectImages(self, request: LegoImageStore, context):
        # TODO Add service for processing images
        image = Image.open(BytesIO(request.image))

        if request.rotation == 90:
            image = image.transpose(Image.ROTATE_270)
        if request.rotation == 180:
            image = image.rotate(180)
        if request.rotation == 270:
            image = image.transpose(Image.ROTATE_90)

        # TODO Detect lego bricks and tag an image
        if not os.path.exists('collected_images'):
            os.makedirs('collected_images')
        if not os.path.exists(os.path.join('collected_images', request.label)):
            os.makedirs(os.path.join('collected_images', request.label))
        image.save(f'./collected_images/{request.label}/image_{int(time.time()) }.jpg')

        return Empty()
