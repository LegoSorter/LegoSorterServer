import cv2
import numpy
from PIL import Image

from lego_sorter_server.analysis.classification.toolkit.transformations.transformation import Transformation


class Simple(Transformation):
    @staticmethod
    def transform(img, desired_size=299):
        old_size = img.size
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = img.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))
        return new_im

    @staticmethod
    def transform_cv2(img: numpy.ndarray, target):
        height, width, channels = img.shape
        scaling_factor = target / max(width, height)
        new_size = tuple([int(x * scaling_factor) for x in (width, height)])
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        img = cv2.resize(img, new_size)
        delta_w = target - new_size[0]
        delta_h = target - new_size[1]
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left
        new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
        # cv2.imshow('image', new_im)
        # cv2.waitKey(0)
        return new_im

