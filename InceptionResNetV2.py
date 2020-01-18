import tensorflow as tf
import gc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for some buggy gpus
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 6144)])
label = pd.read_csv('labels.csv')
label['id'] = label['id'].astype(str) + '.jpg'

BATCH_SIZE = 128
INPUT_SIZE = 224
SEED = 42

#can also use learning finder to define learning rate
def init_model(lr=0.0001, dropout = None):

	model = Sequential()
	base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
	for layer in base.layers:
		layer.trainable = False
	model.add(base)
	model.add(GlobalAveragePooling2D())
	#number of classses here = 120
	model.add(Dense(120, activation = 'softmax'))
	return model
	


#rescale the image and do the augmentations also split the training and validation sets
image_generator = ImageDataGenerator(rescale=1.0/255,
                                           validation_split=0.3,
                                           rotation_range=15,
                                           width_shift_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

train_data_gen = image_generator.flow_from_dataframe(dataframe=label,
                                                           x_col='id',
                                                           y_col='breed',
                                                           batch_size=32,
                                                           directory='train',
                                                           shuffle=True,
                                                           seed=SEED,
                                                           target_size=(INPUT_SIZE, INPUT_SIZE),
                                                           class_mode='categorical',
                                                           subset='training')

val_data_gen = image_generator.flow_from_dataframe(dataframe=label,
                                                              batch_size=32,
                                                              x_col='id',
                                                              y_col='breed',
                                                              directory='train',
                                                              target_size=(INPUT_SIZE, INPUT_SIZE),
                                                              class_mode='categorical',
                                                              subset='validation',
                                                              seed=SEED)

gc.collect()
model = init_model()
#It is said that adam optimizor has better early-stage performance 
#But customized sgd optimizor coudl have better overall performance 
adam = Adam(lr=lr)
#not use momentum here
sgd = SGD(lr=0.1, momentum=0.95, nesterov=False)
model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])

history = model.fit_generator(
    train_data_gen,
    epochs=100,
    validation_data=val_data_gen)

train_set, val_set = train_test_split(train, test_size=0.20, stratify=train['breed'], random_state=SEED)