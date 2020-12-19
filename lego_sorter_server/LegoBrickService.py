from generated import LegoBrick_pb2_grpc
from generated.LegoBrick_pb2 import Image as LegoImage, Empty, ImageStore as LegoImageStore, BoundingBox, ListOfBoundingBoxes

from PIL import Image
from io import BytesIO
import time
import os
from detection import LegoDetector
import numpy as np

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

    def DetectBricks(self, request: BoundingBox, context):
        def resize(img, target):
            width, height = image.size
            scaling_factor = target / max(width,height)
            im_resized = image.resize((int(width*scaling_factor), int(height*scaling_factor)), Image.BICUBIC)
            new_im = Image.new('RGB', (target, target), color=(0, 0, 0))
            new_im.paste(im_resized, (0, 0))
            return new_im, scaling_factor
        lego_detector = LegoDetector('C:\\Users\\Konrad\\dev\\LegoSorterServer\\lego_sorter_server\\models\\lego_detection_model\\saved_model')
        image = Image.open(BytesIO(request.image))
        image = image.convert('RGB')
        image_resized, scale = resize(image, 640) #image.resize((640, 640), 0)
        detections = lego_detector.detect_lego(np.array(image_resized))
        # TODO: selecting a single bb... for now
        #best_detection = np.argmax(detections['detection_scores'])
        bb = BoundingBox()
        bb.ymin, bb.xmin, bb.ymax, bb.xmax = [int(i*640*1/scale) for i in detections['detection_boxes'][0]]
        bb.score = detections['detection_scores'][0]
        bb.label = 'DUMMY_LABEL'
        return bb