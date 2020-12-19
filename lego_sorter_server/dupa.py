from detection.LegoDetector import LegoDetector
import numpy as np
from PIL import Image
from io import BytesIO
from generated.LegoBrick_pb2 import Image as LegoImage, Empty, ImageStore as LegoImageStore, BoundingBox, ListOfBoundingBoxes

def resize(img, target):
    width, height = image.size
    scaling_factor = target / max(width,height)
    im_resized = image.resize((int(width*scaling_factor), int(height*scaling_factor)), Image.BICUBIC)
    new_im = Image.new('RGB', (target, target), color=(0, 0, 0))
    new_im.paste(im_resized, (0, 0))
    return new_im, scaling_factor

lego_detector = LegoDetector('C:\\Users\\Konrad\\dev\\LegoSorterServer\\lego_sorter_server\\models\\lego_detection_model\\saved_model')
image = Image.open('C:\\Users\\Konrad\\dev\\LegoSorterServer\\lego_sorter_server\\images\\image_1608326519.jpg')
image = image.convert('RGB')
image_resized, scale = resize(image, 640)#image.resize((640, 640), 0)
detections = lego_detector.detect_lego(np.array(image_resized))
#best_detection = np.argmax(detections['detection_scores'])
bb = BoundingBox()
bb.ymin, bb.xmin, bb.ymax, bb.xmax = [int(i*640*1/scale) for i in detections['detection_boxes'][0]]
bb.score = detections['detection_scores'][0]
bb.label = 'DUMMY_LABEL'