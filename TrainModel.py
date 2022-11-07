import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.utils.data_utils import Sequence
import autokeras as ak

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

batch_size = 8
img_height = 480
img_width = 640

trainDSPath = "Dataset/train"
validDSPath = "Dataset/valid"
testDSPath = "Dataset/test"


trainDS = tf.keras.utils.image_dataset_from_directory(
  trainDSPath,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

validDS = tf.keras.utils.image_dataset_from_directory(
  trainDSPath,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

testDS = tf.keras.utils.image_dataset_from_directory(
  testDSPath,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

classNames = trainDS.class_names

clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=1
)


history = clf.fit(trainDS)
model=clf.export_model()
model.summary()


AUTOTUNE = tf.data.AUTOTUNE
trainDS = trainDS.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validDS = validDS.cache().prefetch(buffer_size=AUTOTUNE)
testDS = testDS.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
normalizedDS = trainDS.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalizedDS))

numClasses = len(classNames)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(numClasses)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()