from keras.layers import  Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.applications import NASNetLarge, ResNet101V2
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

IMAGE_SIZE = [640, 640]
EPOCHS = 200
Batch = 128
TARGET_SIZE = (224,224)

train_path = '/home/s180193/PythonMy/Biometria/Dataset/Biometria/train'
test_path = '/home/s180193/PythonMy/Biometria/Dataset/Biometria/test'

#Model 1

model = ResNet101V2(input_shape=[224,224,3], weights='imagenet', include_top=False)

for layer in model.layers:
  layer.trainable = False


x = Flatten()(model.output)
prediction = Dense(16, activation='softmax')(x)
model = Model(inputs=model.input, outputs=prediction)
model.summary()

from keras import optimizers


adam = optimizers.Adam(1e-5)
model.compile(loss='binary_crossentropy',
          optimizer=adam,
          metrics=['accuracy'])

train_datagen = ImageDataGenerator(
preprocessing_function=preprocess_input,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

test_datagen = ImageDataGenerator(
preprocessing_function=preprocess_input,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

from datetime import datetime
from keras.callbacks import ModelCheckpoint


train_set = train_datagen.flow_from_directory(train_path,
                                              target_size = TARGET_SIZE,
                                              batch_size = Batch,
                                              class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                          target_size = TARGET_SIZE,
                                          batch_size = Batch,
                                          class_mode = 'categorical')



checkpoint = ModelCheckpoint(filepath='ResNet101V2.h5', 
                              verbose=2, save_best_only=True)


callbacks = [checkpoint]

for layer in model.layers:
  layer.trainable = True


start = datetime.now()

model_history=model.fit(
  train_set,
  validation_data=test_set,
  epochs=EPOCHS,
    callbacks=callbacks)
try:
  pass
except Exception:
  try:
    model.save("ResNet101V2_Save", save_format="tf")
  except:
    model.save("ResNet101V2_Save.h5")

finally:
  duration = datetime.now() - start
  print("Training completed in time: ", duration)

  plt.plot(model_history.history['accuracy'])
  plt.plot(model_history.history['val_accuracy'])
  plt.title('CNN Model accuracy values')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.savefig('modelBIO_2_ResNet101V2.png')
