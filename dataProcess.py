import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataGenerator():
    def __init__(self, traindf, testdf, image_size):
        super().__init__()
        self.traindf = traindf
        self.testdf = testdf
        self.image_size = image_size
        self.datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rescale=1. / 255.)
        
        self.datagen_val = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rescale=1. / 255.)

    def gen_train_data(self):
        return self.datagen.flow_from_dataframe(dataframe=self.traindf,
                                                directory="train",
                                                x_col="id",
                                                y_col="breed",
                                                class_mode="categorical",
                                                target_size=(self.image_size,
                                                             self.image_size))

    def gen_validate_data(self):
        return self.datagen_val.flow_from_dataframe(dataframe=self.testdf,
                                                directory="val",
                                                x_col="id",
                                                y_col="breed",
                                                class_mode="categorical",
                                                target_size=(self.image_size,
                                                             self.image_size))

class Visualizer():
    def __init__(self, accuracy, val_accuracy, loss, val_loss, epochs):
        super().__init__()
        self.accuracy = accuracy
        self.val_accuracy = val_accuracy
        self.loss = loss
        self.val_loss = val_loss
        self.epochs = epochs

    def visualize_data(self):
        plt.figure(figsize = (8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(range(self.epochs), self.accuracy, label = 'Training Accuracy')
        plt.plot(range(self.epochs), self.val_accuracy, label = 'Validation Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(range(self.epochs), self.loss, label = 'Training Loss')
        plt.plot(range(self.epochs), self.val_loss, label = 'Validation Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()
