import logging
import os
import time
import config

import onnxruntime as rt
import numpy as np
from pathlib import Path

from typing import List

from PIL.Image import Image

from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.classification.classifiers.LegoClassifier import LegoClassifier
from lego_sorter_server.analysis.classification.toolkit.transformations.simple import Simple



class KerasOnnxLabelClassifier(LegoClassifier):
    def __init__(self, model_path=os.path.join("lego_sorter_server", "analysis", "classification", "models",
                                               "keras_onnx_model", "447_classes_export_tf2onnx.onnx")):
        super().__init__()
        self.model_path = Path(model_path).absolute()
        self.model = None
        self.initialized = False
        self.size = (224, 224)

    def load_model(self):
        if self.initialized:
            raise Exception("KerasOnnxLabelClassifier already initialized")
        
        if not self.model_path.exists():
            logging.error(f"[KerasOnnxLabelClassifier] No model found in {str(self.model_path)}")
            raise RuntimeError(f"[KerasOnnxLabelClassifier] No model found in {str(self.model_path)}")
        
        start_time = time.time()

        # prefer CUDA Execution Provider over CPU Execution Provider
        EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # initialize the model.onnx
        self.model = rt.InferenceSession(str(self.model_path), providers=EP_list)

        elapsed_time = time.time() - start_time

        logging.info("Loading model took {} seconds".format(elapsed_time))
        self.initialized = True

    def predict(self, images: List[Image]) -> ClassificationResults:
        if not self.initialized:
            self.load_model()

        if len(images) == 0:
            return ClassificationResults.empty()

        images_array = []
        start_time = time.time()
        for img in images:
            transformed = Simple.transform(img, self.size[0])
            img_array = np.array(transformed)
            #img_array = np.expand_dims(img_array, axis=0)
            images_array.append(img_array)

        processing_elapsed_time_ms = 1000 * (time.time() - start_time)

        # Get the outputs metadata as a list of :class:`onnxruntime.NodeArg`
        output_name = self.model.get_outputs()[0].name
        # Get the inputs metadata as a list of :class:`onnxruntime.NodeArg`
        input_name = self.model.get_inputs()[0].name
        # Inference run using image_data as the input to the model 
        predictions = self.model.run([output_name], {input_name: images_array})

        predicting_elapsed_time_ms = 1000 * (time.time() - start_time) - processing_elapsed_time_ms

        logging.info(f"[KerasOnnxLabelClassifier] Preparing images took {processing_elapsed_time_ms} ms, "
                     f"when predicting took {predicting_elapsed_time_ms} ms.")

        predictions = predictions[0]
        prediction_values = (predictions)[0]
        indices = [np.argpartition(values, -config.CLASSIFICATION_LABEL_COUNT)[-config.CLASSIFICATION_LABEL_COUNT:] for values in predictions][0]
        indices = (indices[np.argsort(prediction_values[indices])[-config.CLASSIFICATION_LABEL_COUNT:][::-1]])
        classes = [self.class_names[index] for index in indices if float(prediction_values[index]) > config.CLASSIFICATION_PROBABILITY_MARGIN]
        scores = [float(prediction_values[index]) for index in indices if float(prediction_values[index]) > config.CLASSIFICATION_PROBABILITY_MARGIN]

        return ClassificationResults(classes, scores)