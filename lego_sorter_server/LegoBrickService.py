from generated import LegoBrick_pb2_grpc
from generated.LegoBrick_pb2 import Image as LegoImage, Empty, ImageStore as LegoImageStore, BoundingBox

from PIL import Image
from io import BytesIO
import time
import os
from detection.LegoDetector import LegoDetector
import numpy as np
import uuid

from lego_sorter_server.images.storage.LegoImageStorage import LegoImageStorage


def resize(img, target):
    width, height = img.size
    scaling_factor = target / max(width, height)
    im_resized = img.resize((int(width * scaling_factor), int(height * scaling_factor)), Image.BICUBIC)
    new_im = Image.new('RGB', (target, target), color=(0, 0, 0))
    new_im.paste(im_resized, (0, 0))
    return new_im, scaling_factor


class LegoBrickService(LegoBrick_pb2_grpc.LegoBrickServicer):
    def __init__(self):
        self.lego_detector = LegoDetector()
        self.lego_storage = LegoImageStorage()
        self.lego_detector.__initialize__()

    @staticmethod
    def _prepare_image(request: LegoImage):
        image = Image.open(BytesIO(request.image))

        if request.rotation == 90:
            image = image.transpose(Image.ROTATE_270)
        if request.rotation == 180:
            image = image.rotate(180)
        if request.rotation == 270:
            image = image.transpose(Image.ROTATE_90)

        return image

    def RecognizeLegoBrickInImage(self, request: LegoImage, context):
        image = self._prepare_image(request)

        # Save the image as an unknown
        self.lego_storage.save_image(image, "unknown")

        return Empty()

    def CollectCroppedImages(self, request: LegoImageStore, context):
        # TODO Add service for processing images
        image = Image.open(BytesIO(request.image))
        image = image.convert('RGB')
        width, height = image.size
        if request.rotation == 90:
            image = image.transpose(Image.ROTATE_270)
        if request.rotation == 180:
            image = image.rotate(180)
        if request.rotation == 270:
            image = image.transpose(Image.ROTATE_90)

        if not os.path.exists('collected_cropped_images'):
            os.makedirs('collected_cropped_images')
        if not os.path.exists(os.path.join('collected_cropped_images', request.label)):
            os.makedirs(os.path.join('collected_cropped_images', request.label))

        # TODO Detect lego bricks and tag an image
        image_resized, scale = resize(image, 640)
        detections = self.lego_detector.detect_lego(np.array(image_resized))
        for i in range(0,100):
            if detections['detection_scores'][i] < 0.5:
                # continue # IF NOT SORTED
                break # IF SORTED

            ymin, xmin, ymax, xmax = [int(i * 640 * 1 / scale) for i in detections['detection_boxes'][i]]
            if ymax >= height or xmax >= width:
                continue
            image_new = image.crop([xmin, ymin, xmax, ymax])

            image_new.save(f'./collected_cropped_images/{request.label}/image_{uuid.uuid4().hex}.jpg')

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
        image.save(f'./collected_images/{request.label}/image_{int(time.time())}.jpg')

        return Empty()

    def DetectBricks(self, request: BoundingBox, context):
        image = Image.open(BytesIO(request.image))
        image = image.convert('RGB')
        image_resized, scale = resize(image, 640)  # image.resize((640, 640), 0)
        detections = self.lego_detector.detect_lego(np.array(image_resized))
        # TODO: selecting a single bb... for now
        # best_detection = np.argmax(detections['detection_scores'])
        bb = BoundingBox()
        bb.ymin, bb.xmin, bb.ymax, bb.xmax = [int(i * 640 * 1 / scale) for i in detections['detection_boxes'][0]]

        bb.score = detections['detection_scores'][0]
        bb.label = 'DUMMY_LABEL'
        return bb
