## https://github.com/SaumilShah-7/Dog-Breed-Identification-Kaggle/blob/master/Dog_Breed_Identification_Kaggle%20(using%20ResNet50).ipynb
import tensorflow as tf
#print(tf.__version__)

import pandas as pd
import numpy as np
#from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
import shutil
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 6144)])

y_df = pd.read_csv('labels.csv')

# print(y_df['breed'].value_counts())

y_breed = np.asarray(y_df['breed']).reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y_breed)

labels_mapping = {np.argmax(y[i]): y_df.loc[i,'breed'] for i in range(len(y))}

image_height = 224
image_width = 224

datagen = ImageDataGenerator()

x = []
for i in y_df['id']:
  image = Image.open('train/' + i + '.jpg')
  image = np.array(image.resize((image_height, image_width)))
  x.append(image)
  
x = np.asarray(x)
for i in range(len(x)):
  x[i] = datagen.apply_transform(x = x[i], transform_parameters={'theta': 90})

img = Image.fromarray(x[0], 'RGB')
img.save('my.png')
img.show()

#print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7, shuffle=True, stratify=y)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, len(set([np.argmax(i) for i in y_train])), len(set([np.argmax(i) for i in y_test])))

img = Image.fromarray(x_train[0])
#img

RN50 = ResNet50(include_top=False, weights='imagenet', input_shape=(image_height, image_width, 3))

x_train_features = RN50.predict(preprocess_input(x_train))
x_test_features = RN50.predict(preprocess_input(x_test))

model = Sequential()

model.add(Flatten(input_shape=x_train_features.shape[1:]))
model.add(Dropout(0.5))
model.add(Dense(units=240, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=120, activation='softmax'))

print(model.summary())

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

batch_size = 512
nb_epochs = 30

#x_train_features = datagen.fit(x_train_features) NEED STEPS PER EPOCH

history = model.fit(x_train_features, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=[x_test_features, y_test], callbacks=[mc])

#history = model.fit_generator(generator = train_generator, validation_data=[x_test_features, y_test], steps_per_epoch=len(x_train_features) / batch_size, epochs=nb_epochs)

saved_model = load_model('best_model.h5')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

'''
y_test_pred = saved_model.predict(x_test_features)
y_test_pred_id = np.argmax(y_test_pred, axis=1).reshape(-1, 1)
y_test_id = np.argmax(y_test, axis=1).reshape(-1, 1)

df_cm = pd.DataFrame(confusion_matrix(y_test_id, y_test_pred_id), index=[labels_mapping[i] for i in range(len(labels_mapping))], 
                                            columns=[labels_mapping[i] for i in range(len(labels_mapping))])

most_confused = []
for i in range(len(labels_mapping)):
               most_confused.append((df_cm.index[i], df_cm.iloc[i, df_cm.columns!=df_cm.index[i]].idxmax(), df_cm.iloc[i, df_cm.columns!=df_cm.index[i]].max(axis=0), df_cm.iloc[i].sum(axis=0)))
most_confused.sort(key = lambda x: x[2], reverse = True)

# print(sum(x for _,_,_,x in most_confused))
print(most_confused)


test = []
submission_images = []
for filename in os.listdir('test'):
  image = Image.open('/content/test/'+str(filename))
  submission_images.append(filename.split('.')[0])
  test.append(np.array(image.resize((image_height, image_width))))
test = np.asarray(test)

print(test.shape)

test_features = RN50.predict(preprocess_input(test))
submission_ohe = saved_model.predict(test_features)

submission = pd.DataFrame(submission_ohe, columns=[labels_mapping[i] for i in range(len(labels_mapping))])
submission.insert(0, 'id', submission_images)'''
submission.to_csv('submission.csv', index=False)