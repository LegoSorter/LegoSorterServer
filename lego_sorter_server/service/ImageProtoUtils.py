from io import BytesIO
from typing import List

from PIL import Image

from lego_sorter_server.detection import DetectionUtils
from lego_sorter_server.detection.DetectionResults import DetectionResults
from lego_sorter_server.generated.LegoBrick_pb2 import BoundingBox, ListOfBoundingBoxes
from lego_sorter_server.generated.LegoSorter_pb2 import ImageRequest


class ImageProtoUtils:

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
    def prepare_response_from_bbs_and_labels(bbs: [BoundingBox], labels: [str]) -> List[BoundingBox]:
        for bb, label in zip(bbs, labels):
            bb.label = label

        bb_list = ListOfBoundingBoxes()
        bb_list.packet.extend(bbs)

        return bb_list

    @staticmethod
    def prepare_bbs_response_from_detection_results(detection_results: DetectionResults,
                                                    scale: float,
                                                    target_image_size: (int, int),  # (width, height)
                                                    detection_image_size: int = 640):
        bbs = []
        for i in range(len(detection_results.detection_classes)):
            if detection_results.detection_scores[i] < 0.5:
                continue

            bb = BoundingBox()

            bb.ymin, bb.xmin, bb.ymax, bb.xmax = [int(i * detection_image_size * 1 / scale) for i in
                                                  detection_results.detection_boxes[i]]
            if bb.ymax >= target_image_size[1] or bb.xmax >= target_image_size[0]:
                continue

            bb.score = detection_results.detection_scores[i]
            bb.label = 'lego'
            bbs.append(bb)

        return bbs
