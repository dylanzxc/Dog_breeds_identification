import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.applications import InceptionV3

def init_model():
    model = Sequential()
    model.add(InceptionV3(include_top=False, pooling='avg', weights='imagenet'))
    model.add(Dropout(0.5))
    model.add(Dense(120, activation='softmax'))
    return model

