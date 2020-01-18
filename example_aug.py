import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import h5py #ADDED
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
#from googlenet import init_model
from tensorflow.keras.models import load_model #ADDED
import os #ADDED
import shutil #ADDED
#from get_images import load_stanford
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Activation, Flatten, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.applications import InceptionV3


# for some buggy gpus
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 6144)])

config = tf.compat.v1.ConfigProto() #I need this to run... might need to comment out these three lines
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

#label = load_stanford()

def init_model():
    model = Sequential()
    model.add(InceptionV3(include_top=False, pooling='avg', weights='imagenet'))
    #model.add(GaussianNoise(1))
    model.add(Dropout(0.5))
    model.add(Dense(120, activation='softmax'))
    return model

label = pd.read_csv('labels_stanford.csv')
#label['id'] = label['id'].astype(str) + '.jpg'

label_test = label.head(8000) #ADDED
label = label.iloc[8000:] #ADDED

#label_test = pd.read_csv('labels_test.csv')
print(label_test) #JUST TO CONFIRM
print(label) #JUST TO CONFIRM
IMAGE_SIZE = 299

'''ADD this to get images in correct folder'''
for r, d, f in os.walk('Images/train_new'):
	for i in f:
		if label_test['id'].str.contains(i).any():
			shutil.move('Images/train_new/'+i, 'Images/test_new/'+i)

train_generator = ImageDataGenerator(rescale=1./255, rotation_range = 90, horizontal_flip = True, validation_split = 0.3) #ADD AUGMENTATION HERE
val_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_generator.flow_from_dataframe(dataframe=label,
                                                           x_col='id',
                                                           y_col='breed',
                                                           batch_size=32,
                                                           directory='Images/train_new', #CHANGE
                                                           shuffle=True,
                                                           subset='training',
                                                           target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                           class_mode='categorical')

val_data_gen = train_generator.flow_from_dataframe(dataframe=label,
                                                              batch_size=32,
                                                              x_col='id',
                                                              y_col='breed',
                                                              directory='Images/train_new', #CHANGE
                                                              subset='validation',
                                                              target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                              class_mode='categorical')

test_data_gen = val_generator.flow_from_dataframe(dataframe=label_test, #ADD THIS
                                                              batch_size=1, 
                                                              x_col='id',
                                                              y_col='breed',
                                                              directory='Images/test_new',
                                                              #subset='validation',
                                                              target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                              class_mode='categorical')




model = init_model()
model.layers[0].trainable = False
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit_generator(
    train_data_gen,
    epochs=15,
    validation_data=val_data_gen
)

model.save("model_g.h5") #ADDED
print("Saved model to disk")

loaded_model = load_model("model_g.h5") #ADDED
print("Loaded model from disk")

score = loaded_model.evaluate_generator(generator=test_data_gen, 
steps=32) #ADDED

print(score) #ADDED

