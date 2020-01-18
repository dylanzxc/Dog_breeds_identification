'''RUN get_images.py First!!
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50
import os #ADD
import shutil #ADD
from tensorflow.keras.models import load_model #ADD

def init_model():
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg')) #, weights='imagenet'))
    model.add(Dropout(0.5))
    model.add(Dense(120, activation='softmax'))
    return model

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
#from resnet50 import init_model

# for some buggy gpus
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 6144)])
label = pd.read_csv('labels_stanford.csv') #CHANGE
#label['id'] = label['id'].astype(str) + '.jpg' COMMENT THIS OUT

label_test = label.head(8000) #ADD
label = label.iloc[8000:] #ADD

#ADD
for r, d, f in os.walk('Images/train_new'):
	for i in f:
		if label_test['id'].str.contains(i).any():
			shutil.move('Images/train_new/'+i, 'Images/test_new/'+i)
#ADD
for r, d, f in os.walk('Images/test_new'):
	for i in f:
		if label['id'].str.contains(i).any():
			shutil.move('Images/test_new/'+i, 'Images/train_new/'+i)

IMAGE_SIZE = 224

image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.3)
test_generator = ImageDataGenerator(rescale=1./255) #ADD

train_data_gen = image_generator.flow_from_dataframe(dataframe=label,
                                                           x_col='id',
                                                           y_col='breed',
                                                           batch_size=32,
                                                           directory='Images/train_new', #CHANGE
                                                           shuffle=True,
                                                           target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                           class_mode='categorical',
                                                           subset='training')

val_data_gen = image_generator.flow_from_dataframe(dataframe=label,
                                                              batch_size=32,
                                                              x_col='id',
                                                              y_col='breed',
                                                              directory='Images/train_new', #CHANGE
                                                              target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                              class_mode='categorical',
                                                              subset='validation')

test_data_gen = test_generator.flow_from_dataframe(dataframe=label_test,  #ADD THIS GENERATOR
                                                              batch_size=1,
                                                              x_col='id',
                                                              y_col='breed',
                                                              directory='Images/test_new',
                                                              target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                              class_mode='categorical')


model = init_model()
#model.layers[0].trainable = False
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()

history = model.fit_generator(
    train_data_gen,
    epochs=15,
    validation_data=val_data_gen)

model.save('model_r.h5') #ADD

loaded_model = load_model('model_r.h5') #ADD

score = loaded_model.evaluate_generator(val_data_gen, 32)  #ADD

print(score) #ADD

