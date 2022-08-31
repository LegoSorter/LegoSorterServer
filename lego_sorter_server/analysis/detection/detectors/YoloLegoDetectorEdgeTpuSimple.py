import os
import platform
import threading
import time

import cv2
import numpy as np
from loguru import logger
import torch
import numpy
from pathlib import Path

from matplotlib import pyplot as plt

try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf
    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,

# from pycoral.utils import edgetpu
# from pycoral.utils import dataset
# from pycoral.adapters import common
# from pycoral.adapters import classify

from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults
from lego_sorter_server.analysis.detection.detectors.LegoDetector import LegoDetector


class ThreadSafeSingleton(type):
    _instances = {}
    _singleton_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._singleton_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(ThreadSafeSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class YoloLegoDetectorEdgeTpuSimple(LegoDetector, metaclass=ThreadSafeSingleton):
    def __init__(self, model_path=os.path.join("lego_sorter_server", "analysis", "detection", "models", "edgetpu",
                                               "best-int8_edgetpu.tflite")):
                                               # "yolov5_medium_extended.pt")):
        self.__initialized = False
        self.model_path = Path(model_path).absolute()

    def __initialize__(self):
        if self.__initialized:
            raise Exception("YoloLegoDetectorEdgeTpu already initialized")

        if not self.model_path.exists():
            logger.error(f"[YoloLegoDetectorEdgeTpu] No model found in {str(self.model_path)}")
            raise RuntimeError(f"[YoloLegoDetectorEdgeTpu] No model found in {str(self.model_path)}")

        start_time = time.time()
        delegate = {
            'Linux': 'libedgetpu.so.1',
            'Darwin': 'libedgetpu.1.dylib',
            'Windows': 'edgetpu.dll'}[platform.system()]
        self.interpreter = Interpreter(model_path=str(self.model_path),
                                       experimental_delegates=[load_delegate(delegate)])
        # self.interpreter = edgetpu.make_interpreter(str(self.model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_zero = self.input_details[0]['quantization'][1]
        self.input_scale = self.input_details[0]['quantization'][0]
        self.output_zero = self.output_details[0]['quantization'][1]
        self.output_scale = self.output_details[0]['quantization'][0]

        self.conf_thresh = 0.25
        self.iou_thresh = 0.45
        self.filter_classes = None
        self.agnostic_nms = False
        self.max_det = 1000

        # If the model isn't quantized then these should be zero
        # Check against small epsilon to avoid comparing float/int
        if self.input_scale < 1e-9:
            self.input_scale = 1.0

        if self.output_scale < 1e-9:
            self.output_scale = 1.0

        logger.debug("Input scale: {}".format(self.input_scale))
        logger.debug("Input zero: {}".format(self.input_zero))
        logger.debug("Output scale: {}".format(self.output_scale))
        logger.debug("Output zero: {}".format(self.output_zero))

        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(self.model_path), trust_repo=True)
        # if torch.cuda.is_available():
        #     self.model.cuda()
        elapsed_time = time.time() - start_time

        logger.info("[YoloLegoDetectorEdgeTpu] Loading model took {} seconds".format(elapsed_time))
        self.__initialized = True

    @staticmethod
    def xyxy2yxyx_scaled(xyxy):
        """
        returns (ymin, xmin, ymax, xmax)
        """
        return numpy.array([[coord[1], coord[0], coord[3], coord[2]] for coord in xyxy])

    @staticmethod
    def convert_results_to_common_format(results) -> DetectionResults:
        image_predictions = results[0]  # results.xyxyn[0].cpu().numpy()
        scores = image_predictions[:, 4]
        classes = image_predictions[:, 5].astype(numpy.int64) + 1
        boxes = YoloLegoDetectorEdgeTpuSimple.xyxy2yxyx_scaled(image_predictions[:, :4])
        # scores = numpy.array([image_predictions[4]])
        # classes = numpy.array([image_predictions[5].astype(numpy.int64) + 1])
        # # boxes = YoloLegoDetectorEdgeTpu.xyxy2yxyx_scaled(numpy.array(image_predictions[:4]))
        # boxes = numpy.array([[image_predictions[1], image_predictions[0], image_predictions[3], image_predictions[2]]])

        return DetectionResults(detection_scores=scores, detection_classes=classes, detection_boxes=boxes)

    # pass photo scaled to 640x360 or 640x640
    def detect_lego(self, image: numpy.ndarray) -> DetectionResults:
        if not self.__initialized:
            logger.info("YoloLegoDetectorEdgeTpu is not initialized, this process can take a few seconds for the first time.")
            self.__initialize__()

        if image.shape[0] == 3:
            image = image.transpose((1, 2, 0))

        # Scale input, conversion is: real = (int_8 - zero)*scale
        # image = (image / self.input_scale) + self.input_zero
        # image = image[np.newaxis].astype(np.uint8)


        cropedTop = image[0:320, 20:340].copy()
        cropedMiddle = image[160:480, 20:340].copy()
        cropedBottom = image[320:640, 20:340].copy()

        # cropedTop = image[0:320, 0:320].copy()
        # cropedMiddle = image[160:480, 0:320].copy()
        # cropedBottom = image[320:640, 0:320].copy()

        # cv2.imwrite("/home/adam/lego/cv2_test/LegoSorterServer/1.jpg", cropedTop)
        # cv2.imwrite("/home/adam/lego/cv2_test/LegoSorterServer/2.jpg", cropedMiddle)
        # cv2.imwrite("/home/adam/lego/cv2_test/LegoSorterServer/3.jpg", cropedBottom)

        cropedTop = (cropedTop / self.input_scale) + self.input_zero
        cropedTop = cropedTop[np.newaxis].astype(np.uint8)
        cropedMiddle = (cropedMiddle / self.input_scale) + self.input_zero
        cropedMiddle = cropedMiddle[np.newaxis].astype(np.uint8)
        cropedBottom = (cropedBottom / self.input_scale) + self.input_zero
        cropedBottom = cropedBottom[np.newaxis].astype(np.uint8)

        start_time = time.time()

        self.interpreter.set_tensor(self.input_details[0]['index'], cropedTop)
        self.interpreter.invoke()

        # elapsed_time = 1000 * (time.time() - start_time)
        # logger.debug(f"[YoloLegoDetectorEdgeTpu][detect_lego] Detecting bricks and converting took {elapsed_time} ms.")

        # Scale output
        result1 = (self.interpreter.get_tensor(self.output_details[0]['index']).astype('float32') - self.output_zero) * self.output_scale
        result10 = result1.copy()
        result1 = result1[0]
        result11 = result1.copy()
        result111 = result1[:, [1, 3]]
        result1[:, [1, 3]] = result1[:, [1, 3]]/2
        self.interpreter.set_tensor(self.input_details[0]['index'], cropedMiddle)
        self.interpreter.invoke()
        result2 = (self.interpreter.get_tensor(self.output_details[0]['index']).astype('float32') - self.output_zero) * self.output_scale
        result2 = result2[0]
        result2[:, [1, 3]] = (result2[:, [1, 3]] / 2)  # scale in half y coordinate and height
        result2[:, 1] = result2[:, 1] + 0.25  # move y coordinate
        self.interpreter.set_tensor(self.input_details[0]['index'], cropedBottom)
        self.interpreter.invoke()
        result3 = (self.interpreter.get_tensor(self.output_details[0]['index']).astype('float32') - self.output_zero) * self.output_scale
        result3 = result3[0]
        result3[:, [1, 3]] = (result3[:, [1, 3]] / 2)
        result3[:, 1] = result3[:, 1] + 0.5
        result = np.concatenate((result1, result2, result3));
        result = result[np.newaxis]
        # result = (common.output_tensor(self.interpreter, 0).astype('float32') - self.output_zero) * self.output_scale
        result = YoloLegoDetectorEdgeTpuSimple.non_max_suppression(result, self.conf_thresh, self.iou_thresh, self.filter_classes, self.agnostic_nms, max_det=self.max_det)
        # res=result[0]
        # res[:, :4] = self.get_scaled_coords(res[:, :4], 320, 320, 140, 0)
        # output = []
        #
        # s = ""
        #
        # # Print results
        # # for c in np.unique(result[:, -1]):
        # #     n = (result[:, -1] == c).sum()  # detections per class
        # #     s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
        #
        # # Write results
        # cnt = 0
        # for *xyxy, conf, cls in reversed(res):
        #     output.append({})
        #     output[cnt]['box'] = xyxy
        #     output[cnt]['conf'] = conf
        #     output[cnt]['cls'] = cls
        #     # output[cnt]['cls_name'] = self.names[int(cls)]
        #     cnt += 1
        # logger.info("[YoloLegoDetector][detect_lego] Detecting bricks...")

        # results = self.model([image], size=image.shape[0])
        elapsed_time = 1000 * (time.time() - start_time)
        logger.debug(f"[YoloLegoDetectorEdgeTpu][detect_lego] Detecting bricks and converting took {elapsed_time} ms.")
        return self.convert_results_to_common_format(result)

    def get_scaled_coords(self, xyxy, out_h, out_w, pad_w, pad_h):
        """
        Converts raw prediction bounding box to orginal
        image coordinates.

        Args:
          xyxy: array of boxes
          output_image: np array
          pad: padding due to image resizing (pad_w, pad_h)
        """
        in_h, in_w = (320, 320)
        # out_h, out_w, _ = output_image.shape

        ratio_w = out_w / (in_w - pad_w)
        ratio_h = out_h / (in_h - pad_h)

        out = []
        for coord in xyxy:
            x1, y1, x2, y2 = coord

            x1 *= in_w * ratio_w
            x2 *= in_w * ratio_w
            y1 *= in_h * ratio_h
            y2 *= in_h * ratio_h

            x1 = max(0, x1)
            x2 = min(out_w, x2)

            y1 = max(0, y1)
            y2 = min(out_h, y2)

            out.append((x1, y1, x2, y2))

        return np.array(out).astype(int)

    @staticmethod
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    @staticmethod
    def nms(dets, scores, thresh):
        '''
        dets is a numpy array : num_dets, 4
        scores ia  nump array : num_dets,
        '''

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1 + 1e-9) * (y2 - y1 + 1e-9)
        order = scores.argsort()[::-1]  # get boxes with more ious first

        keep = []
        while order.size > 0:
            i = order[0]  # pick maxmum iou box
            other_box_ids = order[1:]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[other_box_ids])
            yy1 = np.maximum(y1[i], y1[other_box_ids])
            xx2 = np.minimum(x2[i], x2[other_box_ids])
            yy2 = np.minimum(y2[i], y2[other_box_ids])

            # print(list(zip(xx1, yy1, xx2, yy2)))

            w = np.maximum(0.0, xx2 - xx1 + 1e-9)  # maximum width
            h = np.maximum(0.0, yy2 - yy1 + 1e-9)  # maxiumum height
            inter = w * h

            ovr = inter / (areas[i] + areas[other_box_ids] - inter)

            inds = np.where(ovr <= thresh)[0]
            # order = order[inds + 1]

            # intersection over minimum size
            min_areas = np.minimum(areas[i], areas)
            ovr_min = inter / min_areas[other_box_ids]
            bad_inds = np.where(ovr_min >= 0.75)[0]

            ok_inds = inds[~np.isin(inds, bad_inds)]
            order = order[ok_inds + 1]

        return np.array(keep)

    @staticmethod
    def box_iou(box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (np.min(box1[:, None, 2:], box2[:, 2:]) - np.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    @staticmethod
    def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False,
                            labels=(), max_det=300):

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [np.zeros((0, 6))] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = np.zeros((len(l), nc + 5))
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = np.concatenate((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = YoloLegoDetectorEdgeTpuSimple.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype(float)), axis=1)
            else:  # best class only
                conf = np.amax(x[:, 5:], axis=1, keepdims=True)
                j = np.argmax(x[:, 5:], axis=1).reshape(conf.shape)
                x = np.concatenate((box, conf, j.astype(float)), axis=1)[conf.flatten() > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == np.array(classes)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

            i = YoloLegoDetectorEdgeTpuSimple.nms(boxes, scores, iou_thres)  # NMS

            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = YoloLegoDetectorEdgeTpuSimple.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = np.dot(weights, x[:, :4]).astype(float) / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            #if (time.time() - t) > time_limit:
            #    print(f'WARNING: NMS time limit {time_limit}s exceeded')
            #    break  # time limit exceeded

        return output
