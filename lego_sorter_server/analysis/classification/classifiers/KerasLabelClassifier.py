import logging
import os
import time
import config

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


class KerasLabelClassifier(LegoClassifier):
    def __init__(self, model_path=os.path.join("lego_sorter_server", "analysis", "classification", "models",
                                               "keras_model", "447_classes.h5")):
        super().__init__()
        self.model_path = model_path
        self.model = None
        self.initialized = False
        self.size = (224, 224)

    def load_model(self):
        self.model = keras.models.load_model(self.model_path)
        self.initialized = True

        #----------------------------------------------------
        # Export the Tensorflow Keras model to ONNX
        # Link: https://onnxruntime.ai/docs/tutorials/tf-get-started.html
        # Link: https://github.com/onnx/tensorflow-onnx
        # Extra: https://medium.com/analytics-vidhya/how-to-convert-your-keras-model-to-onnx-8d8b092c4e4f
        ## Export model to ONNX for further conversion
        #print("---")
        #print("Started conversion from Tensorflow Keras to ONNX")
        #input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name='input_1')]
        #onnx_model_name = '447_classes_export.onnx'
        #onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature, opset=17)
        ##onnx_model = keras2onnx.convert_keras(self.model, self.model.name)
        #onnx.save_model(onnx_model, onnx_model_name)
        #print("Finished conversion")

        #print("---")
        #print("Started conversion from ONNX to PyTorch")
        ## Export the ONNX model to PyTorch
        ##pytorch_model = ConvertModel(onnx_model)
        #pytorch_model = ConvertModel(onnx_model, experimental=True)
        #torch.save(pytorch_model, '447_classes_export.pt')
        #print("Finished conversion")
        #exit()

        #---

        ## Export model to ONNX
        #print("---")
        #print("Started conversion from Tensorflow Keras to ONNX")
        #input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name='input_1')]
        #onnx_model_name = '447_classes_export_tf2onnx.onnx'
        ##onnx_model_name = '447_classes_export_keras2onnx.onnx'
        #onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature, opset=17)
        ##onnx_model = keras2onnx.convert_keras(self.model, self.model.name)
        #onnx.save_model(onnx_model, onnx_model_name)
        #print("Finished conversion")
        #exit()
        
        #----------------------------------------------------

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
            img_array = np.expand_dims(img_array, axis=0)
            images_array.append(img_array)

        processing_elapsed_time_ms = 1000 * (time.time() - start_time)

        predictions = self.model(np.vstack(images_array))

        predicting_elapsed_time_ms = 1000 * (time.time() - start_time) - processing_elapsed_time_ms

        logging.info(f"[KerasClassifier] Preparing images took {processing_elapsed_time_ms} ms, "
                    f"when predicting took {predicting_elapsed_time_ms} ms.")

        prediction_values = (predictions.numpy())[0]
        indices = [np.argpartition(values, -config.CLASSIFICATION_LABEL_COUNT)[-config.CLASSIFICATION_LABEL_COUNT:] for values in predictions][0]
        indices = (indices[np.argsort(prediction_values[indices])[-config.CLASSIFICATION_LABEL_COUNT:][::-1]])
        classes = [self.class_names[index] for index in indices if float(prediction_values[index]) > config.CLASSIFICATION_PROBABILITY_MARGIN]
        scores = [float(prediction_values[index]) for index in indices if float(prediction_values[index]) > config.CLASSIFICATION_PROBABILITY_MARGIN]

        return ClassificationResults(classes, scores)