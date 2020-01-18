import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D

def init_model():
	model = Sequential()
	model.add(Conv2D(64, kernel_size=(1,1), activation="relu", input_shape=(224,224,3)))	
	model.add(MaxPooling2D())
	model.add(Conv2D(32, kernel_size=(1,1), activation="relu"))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(512, activation="relu"))
	model.add(Dense(120, activation="softmax"))
	return model
