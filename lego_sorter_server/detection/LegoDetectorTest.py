from lego_sorter_server.detection.LegoDetector import LegoDetector
from PIL import Image
from object_detection.utils import visualization_utils as viz_utils
from pathlib import Path
import numpy as np

lego_detector = LegoDetector()


def draw_bounding_boxes_on_image(image_path):
    im = Image.open(image_path)
    im = im.convert('RGB')
    im = im.resize((640, 640), 0)
    image = np.array(im)

    detections = lego_detector.detect_lego(image)
    category_index = {1: {'id': 1, 'name': 'lego'}}

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.50,
        agnostic_mode=False)

    return image


if __name__ == '__main__':
    for img_path in Path(".").glob("*.jpg"):
        img = draw_bounding_boxes_on_image(str(img_path.absolute()))
        img = Image.fromarray(img)
        img.save("./" + img_path.name.split(".")[0] + ".jpg")
