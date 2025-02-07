import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)


train_data = object_detector.DataLoader.from_pascal_voc(
  'android_figurine/train',
  'android_figurine/train',
  ['android','pig_android']
)

val_data = object_detector.DataLoader.from_pascal_voc(
  'android_figurine/train',
  'android_figurine/train',
  ['android','pig_android']
)

spec = model_spec.get('efficientdet_lite2')

model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)

model.evaluate(val_data)

model.export(export_dir='.', tflite_filename='testmodel.tflite')

model.evaluate_tflite('testmodel.tflite', val_data)