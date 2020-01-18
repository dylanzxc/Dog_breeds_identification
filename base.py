import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import h5py
from network import Model
from dataProcess import Visualizer
from dataProcess import DataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# for some buggy gpus
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 6144)])

traindf = pd.read_csv('labels.csv')
testdf = pd.read_csv('labels.csv')

traindf['id'] = traindf['id'].astype(str) + '.jpg'
testdf['id'] = testdf['id'].astype(str) + '.jpg'

image_size = 224
datagen = DataGenerator(traindf, testdf, image_size)
train_generator = datagen.gen_train_data()
valid_generator = datagen.gen_validate_data()

num_classes = 120
my_new_model = Model(num_classes, train_generator, valid_generator)
my_new_model.build_model()

# test_datagen=ImageDataGenerator(preprocessing_function = preprocess_input, rescale = 1./255.)
# test_generator=test_datagen.flow_from_dataframe(
#                             dataframe = testdf,
#                             directory = "test",
#                             x_col = "id",
#                             y_col = None,
#                             has_ext = False,
#                             batch_size = 1,
#                             seed = 42,
#                             shuffle = False,
#                             class_mode = None,
#                             target_size = (image_size, image_size))

history = my_new_model.train_my_model()
# my_new_model.valid_generator()
# test_generator.reset()
show_plot = Visualizer(history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss'], 3)
show_plot.visualize_data()

# pred=my_new_model.predict_generator(test_generator,verbose = 1)
# labels = (train_generator.class_indices)
# labels = list(labels.keys())
# df = pd.DataFrame(data=pred,
#                  columns=labels)

# columns = list(df)
# columns.sort()
# df = df.reindex(columns=columns)

# filenames = testdf["id"]
# df["id"]  = filenames

# cols = df.columns.tolist()
# cols = cols[-1:] + cols[:-1]
# df = df[cols]
# df.head(5)

# df.to_csv("submission.csv",index=False)
