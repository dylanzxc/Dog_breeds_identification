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
from tensorflow.keras.applications.inception_v3 import decode_predictions
import csv


config = tf.compat.v1.ConfigProto() #I need this to run... might need to comment out these three lines
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

label = pd.read_csv('labels_stanford.csv')
#label['id'] = label['id'].astype(str) + '.jpg'

label_test = label.head(8000) #ADDED
label = label.iloc[8000:] #ADDED

label_class = label_test.drop_duplicates(subset = "breed")
#print(label_class)
label_class.to_csv('labels_class.csv')

IMAGE_SIZE = 299

#val_generator = ImageDataGenerator(rescale=1./255)

'''test_data_gen = val_generator.flow_from_dataframe(dataframe=label_test,
                                                              batch_size=1,
                                                              x_col='id',
                                                              y_col='breed',
                                                              directory='Images/test_new',
                                                              target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                              class_mode='categorical')'''


#labels = test_data_gen.class_indices
#print(labels)
breed_labels = pd.read_csv('dict.csv')
#print(breed_labels)

loaded_model = load_model("model_g_final.h5")
print("Loaded model from disk")

#with open('dict.csv', 'w', newline="") as csv_file:  
#    writer = csv.writer(csv_file)
#    for key, value in labels.items():
#       writer.writerow([key, value])

import imageio
x = imageio.imread('preds/sam.jpeg') #THIS IS THE PATH TO THE IMAGE
x = x.astype('float32') / 255.
x = np.expand_dims(x, axis=0)

pred = loaded_model.predict_classes(x)
#topK = decode_predictions(pred, top=5)[0]
#print(pred[0])
breed_id = pred[0]

breed_code = breed_labels.loc[breed_labels['breed_id'] == breed_id]
breed_code = breed_code['breed_code'].values[0]
breed = label_class.loc[label_class['breed'] == breed_code]
print(breed['breed_name'].values[0]) 
