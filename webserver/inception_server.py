import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Dense, Dropout
from PIL import Image
from flask_cors import CORS
import numpy as np
import flask
import io
import pandas as pd

app = flask.Flask(__name__)
CORS(app)
model = Sequential()

def load_model():
    global model
    model = tf.keras.models.load_model('model_g_final.h5')

def prepare_image(image, target):
    image = image.resize(target)
    image = img_to_array(image)
    image = image.astype('float32') / 255.
    image = np.expand_dims(image, axis=0)

    return image

@app.route('/init', methods=['GET'])
def hello_world():
    return 'Service is up and running.'

@app.route('/inception', methods=['POST'])
def predict_inception():
    data = {'success': False}
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    if flask.request.method == 'POST':
        image = flask.request.files['image'].read()
        image = Image.open(io.BytesIO(image))
        image = prepare_image(image, target=(299, 299))
        
        pred = model.predict(image)
        pred_id = np.argsort(-pred)

        label_class = pd.read_csv('labels_class.csv')
        breed_labels = pd.read_csv('dict.csv')
        
        breed_id = pred_id[0][0:5]
        data['predictions'] = []
        for i in range(5):
            prob = pred[0][pred_id[0][i]]
            breed_code = breed_labels.loc[breed_labels['breed_id'] == breed_id[i]]
            breed_code = breed_code['breed_code'].values[0]
            breed = label_class.loc[label_class['breed'] == breed_code]
            r = {'label': breed['breed_name'].values[0], 'probability': float(prob)}
            data['predictions'].append(r)
        data['success'] = True
    return flask.jsonify(data)

if __name__ == "__main__":
    load_model()
    app.run(host='0.0.0.0')
