from lego_sorter_server.detection.detectors.TFLegoDetector import TFLegoDetector
from PIL import Image
from object_detection.utils import visualization_utils as viz_utils
from pathlib import Path
import numpy as np
import tensorflow as tf

from lego_sorter_server.detection.detectors.YoloLegoDetector import YoloLegoDetector

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# lego_detector = TFLegoDetector(model_path='./models/tf_model/saved_model')
lego_detector = YoloLegoDetector(model_path='models/yolo_model/yolov5_small.pt')


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
        img.save("./" + img_path.name.split(".")[0] + "_result.jpg")
