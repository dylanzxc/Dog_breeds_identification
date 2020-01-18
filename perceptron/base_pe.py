import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import zipfile
#from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from perceptron import init_model

# for some buggy gpus
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2048)])
zf = zipfile.ZipFile('../labels.csv.zip')
label = pd.read_csv(zf.open('labels.csv'))
label['id'] = label['id'].astype(str) + '.jpg'

IMAGE_SIZE = 224

train_image_generator = ImageDataGenerator(rescale=1.0/255,
                                           validation_split=0.3,
                                           rotation_range=15,
                                           width_shift_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)


train_data_gen = train_image_generator.flow_from_dataframe(dataframe=label,
                                                           batch_size=32,
                                                           x_col='id',
                                                           y_col='breed',
                                                           directory='../train',
                                                           shuffle=True,
                                                           target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                           subset='training',
                                                           class_mode='categorical')

val_data_gen = train_image_generator.flow_from_dataframe(dataframe=label,
                                                              batch_size=1,
                                                              x_col='id',
                                                              y_col='breed',
                                                              directory='../train',
                                                              target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                              subset='validation',
                                                              class_mode='categorical')

model = init_model()
#if we use a pre-train model this is to tell kearas not train the pretrain model again
#model.layers[0].trainable = False
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()

history = model.fit_generator(
    train_data_gen,
    epochs=50,
    validation_data=val_data_gen,
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
