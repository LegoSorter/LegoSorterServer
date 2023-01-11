from io import BytesIO
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
from numpy import ndarray

from lego_sorter_server.analysis.MyMessage import MyMessage
from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.detection import DetectionUtils
from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults
from lego_sorter_server.generated.Messages_pb2 import ImageRequest, BoundingBox, ListOfBoundingBoxes


class ImageProtoUtils:
    DEFAULT_LABEL = "Lego"

    @staticmethod
    def prepare_image(request: ImageRequest) -> Image.Image:
        image = Image.open(BytesIO(request.image))
        image = image.convert('RGB')

        if request.rotation == 90:
            image = image.transpose(Image.ROTATE_270)
        elif request.rotation == 180:
            image = image.rotate(180)
        elif request.rotation == 270:
            image = image.transpose(Image.ROTATE_90)

        return image

    @staticmethod
    def prepare_image_cv2(request: ImageRequest) -> ndarray:
        file_bytes = np.asarray(bytearray(BytesIO(request.image).read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # image = Image.open(BytesIO(request.image))
        # image = image.convert('RGB')

        if request.rotation == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            # image = image.transpose(Image.ROTATE_270)
        elif request.rotation == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
            # image = image.rotate(180)
        elif request.rotation == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # image = image.transpose(Image.ROTATE_90)

        return image

    @staticmethod
    def crop_bounding_boxes(image: Image.Image, bbs: List[BoundingBox]) -> List[Tuple[BoundingBox, Image.Image]]:
        bbs_with_blobs = []

        for bb in bbs:
            cropped_brick = DetectionUtils.crop_with_margin(image, bb.ymin, bb.xmin, bb.ymax, bb.xmax)
            bbs_with_blobs.append((bb, cropped_brick))

        return bbs_with_blobs

    @staticmethod
    def prepare_response_from_analysis_results(detection_results: DetectionResults,
                                               classification_results: ClassificationResults) -> ListOfBoundingBoxes:

        bounding_boxes = []
        for i in range(len(detection_results.detection_boxes)):
            if detection_results.detection_scores[i] < 0.5:
                continue

            bb = BoundingBox()

            bb.ymin, bb.xmin, bb.ymax, bb.xmax = [int(coord) for coord in detection_results.detection_boxes[i]]
            bb.score = classification_results.classification_scores[i]
            bb.label = classification_results.classification_classes[i]
            bounding_boxes.append(bb)

        bb_list = ListOfBoundingBoxes()
        bb_list.packet.extend(bounding_boxes)

        return bb_list

    @staticmethod
    def prepare_response_from_analysis_results_top5(detection_results: DetectionResults,
                                               classification_results: ClassificationResults):

        bounding_boxes = []
        for i in range(len(detection_results.detection_boxes)):
            if detection_results.detection_scores[i] < 0.5:
                continue

            bb = MyMessage()

            bb.ymin, bb.xmin, bb.ymax, bb.xmax = [int(coord) for coord in detection_results.detection_boxes[i]]
            bb.score = classification_results.classification_scores[i]
            bb.label = classification_results.classification_classes[i]
            bb.score_top5 = classification_results.classification_scores_top5[i]
            bb.label_top5 = classification_results.classification_classes_top5[i]
            bounding_boxes.append(bb)

        return bounding_boxes

    @staticmethod
    def prepare_bbs_response_from_detection_results(detection_results: DetectionResults) -> ListOfBoundingBoxes:
        bbs = []
        for i in range(len(detection_results.detection_classes)):
            if detection_results.detection_scores[i] < 0.5:
                continue

            bb = BoundingBox()

            bb.ymin, bb.xmin, bb.ymax, bb.xmax = [int(coord) for coord in detection_results.detection_boxes[i]]
            bb.score = detection_results.detection_scores[i]
            bb.label = ImageProtoUtils.DEFAULT_LABEL
            bbs.append(bb)

        bb_list = ListOfBoundingBoxes()
        bb_list.packet.extend(bbs)

        return bb_list
