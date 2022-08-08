import logging
import os
import time

import tensorflow as tf
import numpy as np

from typing import List

from tensorflow import keras
from PIL.Image import Image

from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.classification.classifiers.LegoClassifier import LegoClassifier
from lego_sorter_server.analysis.classification.toolkit.transformations.simple import Simple

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



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
        self.model_path = model_path
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

    def predict(self, images: List[Image]) -> ClassificationResults:
        if not self.initialized:
            self.load_model()

        if len(images) == 0:
            return ClassificationResults.empty()
        images_array = []
        start_time = time.time()
        for img in images:
            res = self.transform(img)
            images_array.append(res)

        images_array = torch.stack(images_array, dim=0)
        processing_elapsed_time_ms = 1000 * (time.time() - start_time)
        with torch.no_grad():
            images_array = images_array.to(dev, non_blocking=True)
            predictions = self.model(images_array)

        predicting_elapsed_time_ms = 1000 * (time.time() - start_time) - processing_elapsed_time_ms

        # predictions = predictions.cpu()
        indices = [int(values.argmax()) for values in predictions]
        classes = [self.class_names[index] for index in indices]
        # scores = [float(prediction[index]) for index, prediction in zip(indices, predictions)]
        probs = torch.nn.functional.softmax(predictions, dim=1)
        # probs2 = torch.nn.functional.softmax(predictions)
        indices2 = [int(values.argmax()) for values in probs]
        scores = [float(probs[index]) for index, probs in zip(indices, probs)]
        scores_calc_time_ms = 1000 * (time.time() - start_time) - predicting_elapsed_time_ms

        logging.info(f"[KerasClassifier] Preparing images took {processing_elapsed_time_ms} ms, "
                     f"predicting took {predicting_elapsed_time_ms} ms, "
                     f"when calculating scores took {scores_calc_time_ms} ms.")

        return ClassificationResults(classes, scores)
