import cv2
import numpy
from loguru import logger
import os
import time

# import tensorflow as tf
import numpy as np

from typing import List

# from tensorflow import keras
from PIL import Image
from lego_sorter_server.analysis.detection import DetectionUtils
from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.classification.classifiers.LegoClassifier import LegoClassifier
from lego_sorter_server.analysis.classification.toolkit.transformations.simple import Simple

# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)



import torch
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from lego_sorter_server.analysis.classification.classifiers.tiny_vit import tiny_vit_5m_224

try:
    from timm.data import TimmDatasetTar
except ImportError:
    # for higher version of timm
    from timm.data import ImageDataset as TimmDatasetTar

from torchvision import datasets, transforms
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp


class TinyViTClassifier(LegoClassifier):
    def __init__(self, model_path=os.path.join("lego_sorter_server", "analysis", "classification", "models",
                                               "tiny_vit_model", "tinyvit.pth")):
        super().__init__()
        env_model_path = os.getenv("LEGO_SORTER_TINYVIT_MODEL_PATH")
        if env_model_path is None or env_model_path == "":
            self.model_path = model_path
        else:
            self.model_path = env_model_path
        self.model = None
        self.initialized = False
        self.transform = None
        self.size = (224, 224)

    def load_model(self):
        self.model = tiny_vit_5m_224(pretrained=False)
        self.model.load_state_dict(torch.load(self.model_path,map_location='cpu'))
        self.model = self.model.to(dev)
        self.model = self.model.eval()
        self.transform = self.build_transform()

        self.initialized = True


    def build_transform(self):
        # RGB: mean, std
        rgbs = dict(
            default=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            inception=(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD),
            clip=((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
        )
        mean, std = rgbs["default"]

        t = []
        if True:
            size = int((256 / 224) * 224)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp('bicubic')),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(224))
        else:
            t.append(
                transforms.Resize((224, 224),
                                  interpolation=_pil_interp('bicubic'))
            )

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        transform = transforms.Compose(t)
        return transform

    # based on https://github.com/YU-Zhiyang/opencv_transforms_torchvision/
    def transform_cv2(self, image):
        mean, std = (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        if False:
            size = int((256 / 224) * 224)
            # resize
            h, w, c = image.shape
            if (w <= h and w == size) or (h <= w and h == size):
                image = image
            else:
                if w < h:
                    ow = size
                    oh = int(size * h / w)
                    image =  cv2.resize(image, dsize=(ow, oh), interpolation=cv2.INTER_CUBIC)
                else:
                    oh = size
                    ow = int(size * w / h)
                    image = cv2.resize(image, dsize=(ow, oh), interpolation=cv2.INTER_CUBIC)
            # CenterCrop
            h, w, _ = image.shape
            th, tw = (224, 224)
            i = int(round((h - th) * 0.5))
            j = int(round((w - tw) * 0.5))
            x1, y1, x2, y2 = round(i), round(j), round(i + th), round(j + tw)
            try:
                check_point1 = image[x1, y1, ...]
                check_point2 = image[x2 - 1, y2 - 1, ...]
            except IndexError:
                # warnings.warn('crop region is {} but image size is {}'.format((x1, y1, x2, y2), img.shape))
                image = cv2.copyMakeBorder(image, - min(0, x1), max(x2 - image.shape[0], 0),
                                         -min(0, y1), max(y2 - image.shape[1], 0), cv2.BORDER_CONSTANT, value=[0, 0, 0])
                y2 += -min(0, y1)
                y1 += -min(0, y1)
                x2 += -min(0, x1)
                x1 += -min(0, x1)

            finally:
                image = image[x1:x2, y1:y2, ...].copy()
        else:
            image = cv2.resize(image, (224, 224))
        # image, _ = DetectionUtils.resize_cv2(image, 224)  # resize fill blank
        # to torch
        tensor = torch.from_numpy(image.transpose((2, 0, 1)))
        if isinstance(tensor, torch.ByteTensor) or tensor.max() > 1:
            tensor = tensor.float().div(255)
        # normalize
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)

        return tensor

    # def predict(self, images: List[Image.Image]) -> ClassificationResults:
    def predict(self, images: List[numpy.ndarray]) -> ClassificationResults:
        if not self.initialized:
            self.load_model()

        if len(images) == 0:
            return ClassificationResults.empty()
        images_array = []
        start_time = time.time()
        for img in images:
            # res = self.transform(img)
            # cv2.imshow('image', img)
            # cv2.waitKey(0)
            # pilimg = Image.fromarray(img)
            # pilimg.show()
            # res = self.transform(pilimg)
            # transform = transforms.ToPILImage()
            # img0 = transform(res)
            # img0.show()
            res2 = self.transform_cv2(img)
            # transform2 = transforms.ToPILImage()
            # img2 = transform2(res2)
            # img2.show()
            images_array.append(res2)
            # images_array.append(res)

        images_array = torch.stack(images_array, dim=0)
        processing_elapsed_time_ms = 1000 * (time.time() - start_time)
        with torch.no_grad():
            images_array = images_array.to(dev, non_blocking=True)
            predictions = self.model(images_array)

        predicting_elapsed_time_ms = 1000 * (time.time() - start_time) - processing_elapsed_time_ms

        # predictions = predictions.cpu()
        indices = [int(values.argmax()) for values in predictions]
        # predictions_np_array = np.array(predictions)
        indices_top5_sorted = []
        for values in predictions:
            _, ind = values.topk(5)
            indices_top5_sorted.append(ind)
        # indices_top5 = np.array([np.argpartition(values, -5)[-5:].tolist() for values in predictions])
        # indices_top5_sorted = [index[np.argsort(predictions[i][index])][::-1] for i, index in enumerate(indices_top5)]
        # indices_top5_sorted = [index[np.argsort(predictions_np_array[i][index])][::-1] for i, index in enumerate(indices_top5)]
        classes = [self.class_names[index] for index in indices]
        classes_top5 = [[self.class_names[ind] for ind in index] for index in indices_top5_sorted]
        # scores = [float(prediction[index]) for index, prediction in zip(indices, predictions)]
        probs = torch.nn.functional.softmax(predictions, dim=1)
        # probs2 = torch.nn.functional.softmax(predictions)
        indices2 = [int(values.argmax()) for values in probs]
        scores = [float(probs[index]) for index, probs in zip(indices, probs)]
        scores_top5 = [[float(probs[ind]) for ind in index] for index, probs in zip(indices_top5_sorted, probs)]
        all_time_ms = 1000 * (time.time() - start_time)
        logger.debug(f"[TinyViTClassifierFast] Preparing images and classification took {all_time_ms} ms, "
                     f"preparing images {processing_elapsed_time_ms} ms, classification {predicting_elapsed_time_ms} ms.")

        return ClassificationResults(classes, scores, classes_top5, scores_top5)
