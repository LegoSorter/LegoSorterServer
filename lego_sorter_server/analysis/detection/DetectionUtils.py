import cv2
import numpy
import math
from PIL import Image, ImageOps


def resize(img, target):
    width, height = img.size
    scaling_factor = target / max(width, height)
    im_resized = img.resize((int(width * scaling_factor), int(height * scaling_factor)), Image.BICUBIC)
    new_im = Image.new('RGB', (target, target), color=(0, 0, 0))
    new_im.paste(im_resized, (0, 0))
    # new_im = ImageOps.pad(img, (target, target), method=Image.BICUBIC, color=(0, 0, 0), centering=(0, 0))
    return new_im, scaling_factor


def resize_cv2(img: numpy.ndarray, target):
    height, width, channels = img.shape
    scaling_factor = target / max(width, height)
    new_size = tuple([int(x * scaling_factor) for x in (width, height)])
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    img = cv2.resize(img, new_size)
    delta_w = target - new_size[0]
    delta_h = target - new_size[1]
    new_im = cv2.copyMakeBorder(img, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
    # cv2.imshow('image', new_im)
    # cv2.waitKey(0)
    return new_im, scaling_factor


def resize_cv2_no_pad(img: numpy.ndarray, target):
    height, width, channels = img.shape
    scaling_factor = target / max(width, height)
    new_size = tuple([int(x * scaling_factor) for x in (width, height)])
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    img = cv2.resize(img, new_size)
    # delta_w = target - new_size[0]
    # delta_h = target - new_size[1]
    # new_im = cv2.copyMakeBorder(img, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
    # cv2.imshow('image', new_im)
    # cv2.waitKey(0)
    return img, scaling_factor


#  https://github.com/jveitchmichaelis/edgetpu-yolo/blob/2abb620821fb2e2f638e801bdab4cf1eaf513d34/utils.py
def resize_and_pad(image, desired_size):
    old_size = image.shape[:2]
    ratio = float(desired_size / max(old_size))
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]

    pad = (delta_w, delta_h)

    color = [100, 100, 100]
    new_im = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT,
                                value=color)

    return new_im, pad


def get_image_tensor(img, max_size):
    """
    Reshapes an input image into a square with sides max_size
    """
    # img = cv2.imread(image_path)

    resized, pad = resize_and_pad(img, max_size)
    resized = resized.astype(numpy.float32)

    # Normalise!
    resized /= 255.0

    return img, resized, pad


def crop_with_margin_from_bb(image, bounding_box, abs_margin=0, rel_margin=0.10):
    return crop_with_margin(image, bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], abs_margin,
                            rel_margin)


def crop_with_margin(image, ymin, xmin, ymax, xmax, abs_margin=0, rel_margin=0.10):
    width, height = image.size

    # Apply margins
    avg_length = ((xmax - xmin) + (ymax - ymin)) / 2
    ymin = max(ymin - abs_margin - rel_margin * avg_length, 0)
    xmin = max(xmin - abs_margin - rel_margin * avg_length, 0)
    ymax = min(ymax + abs_margin + rel_margin * avg_length, height)
    xmax = min(xmax + abs_margin + rel_margin * avg_length, width)

    return image.crop([xmin, ymin, xmax, ymax])


def crop_with_margin_from_bb_cv2(image, bounding_box, abs_margin=0, rel_margin=0.10):
    return crop_with_margin_cv2(image, bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], abs_margin,
                            rel_margin)


def crop_with_margin_cv2(image, ymin, xmin, ymax, xmax, abs_margin=0, rel_margin=0.10):
    height, width, channels = image.shape

    # Apply margins
    avg_length = ((xmax - xmin) + (ymax - ymin)) / 2
    ymin = max(ymin - abs_margin - rel_margin * avg_length, 0)
    xmin = max(xmin - abs_margin - rel_margin * avg_length, 0)
    ymax = min(ymax + abs_margin + rel_margin * avg_length, height)
    xmax = min(xmax + abs_margin + rel_margin * avg_length, width)
    croped = image[ int(math.floor(ymin)):int(math.ceil(ymax)), int(math.floor(xmin)):int(math.ceil(xmax))]
    # cv2.imshow('image', croped)
    # cv2.waitKey(0)
    return croped.copy()
