from io import BytesIO
from typing import List

from PIL import Image

from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.detection import DetectionUtils
from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults
from lego_sorter_server.generated.LegoBrick_pb2 import BoundingBox, ListOfBoundingBoxes
from lego_sorter_server.generated.LegoSorter_pb2 import ImageRequest


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
    def crop_bounding_boxes(image: Image.Image, bbs: List[BoundingBox]) -> [(BoundingBox, Image.Image)]:
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
    def prepare_bbs_response_from_detection_results(detection_results: DetectionResults) -> List[BoundingBox]:
        bbs = []
        for i in range(len(detection_results.detection_classes)):
            if detection_results.detection_scores[i] < 0.5:
                continue

            bb = BoundingBox()

            bb.ymin, bb.xmin, bb.ymax, bb.xmax = [int(coord) for coord in detection_results.detection_boxes[i]]
            bb.score = detection_results.detection_scores[i]
            bb.label = ImageProtoUtils.DEFAULT_LABEL
            bbs.append(bb)

        return bbs
