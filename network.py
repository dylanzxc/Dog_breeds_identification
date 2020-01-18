import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Activation, Flatten, Dropout, BatchNormalization


class Model():
    def __init__(self, num_classes, train_generator, valid_generator):
        super().__init__()
        self.num_classes = num_classes
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.model = Sequential()

    def build_model(self):
        self.model.add(
            ResNet50(
                include_top=False,
                pooling='avg',
                weights=
                'imagenet'
            ))
        self.model.add(Dense(512, activation = 'relu'))
        # self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.layers[0].trainable = False
        self.model.compile(optimizer='sgd',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()
        return

    def train_my_model(self):
        self.model.fit_generator(generator=self.train_generator,
                                 steps_per_epoch=self.train_generator.n,
                                 validation_data=self.valid_generator,
                                 validation_steps=self.valid_generator.n,
                                 epochs=1)
        return

    def evaluate_generator(self):
        self.model.evaluate_generator(generator=self.valid_generator)
        return
