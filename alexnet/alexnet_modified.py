import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization 
import numpy as np 
np.random.seed(1000)

def init_model():
	model = Sequential()

    # 1st Convolutional Layer(11×11 with stride 4 in (Krizhevsky et al., 2012))
	model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
	model.add(Activation('relu'))
    # Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Normalization to reduce overfitting 
	model.add(BatchNormalization())

    # 2rd Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
    # Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	model.add(BatchNormalization())

    # 3th Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

    # 4th Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

    # 5th Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
    # Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	model.add(BatchNormalization())


	#model.add(Conv2D(filters=128, kernel_size=(7,7), strides=(2,2), padding='valid'))
	#model.add(Activation('relu'))
    # Max Pooling
	#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	#model.add(BatchNormalization())



    # Passing it to a Fully Connected layer
	model.add(Flatten())
    # 1st Fully Connected Layer
	model.add(Dense(4096, input_shape=(224*224*3,)))
	model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
	model.add(Dropout(0.4))
	model.add(BatchNormalization())

    # 2nd Fully Connected Layer
	model.add(Dense(1000))
	model.add(Activation('relu'))
    # Add Dropout
	model.add(Dropout(0.4))
	model.add(BatchNormalization())

    # 3rd Fully Connected Layer
	model.add(Dense(1000))
	model.add(Activation('relu'))

    # Output Layer
#	model.add(Dense(256))
#	model.add(Activation('softmax'))
	return model 