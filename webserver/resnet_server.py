import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Dense, Dropout
from PIL import Image
from flask_cors import CORS
import numpy as np
import flask
import io

app = flask.Flask(__name__)
CORS(app)
model = Sequential()

def load_model():
    global model
    model.add(ResNet50(weights='imagenet'))

def prepare_image(image, target):
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image
@app.route('/init', methods=['GET'])
def hello_world():
    return 'Service is up and running.'

@app.route('/resnet', methods=['POST'])
def predict_resnet():
    data = {'success': False}

    if flask.request.method == 'POST':
        image = flask.request.files['image'].read()
        image = Image.open(io.BytesIO(image))

        image = prepare_image(image, target=(224, 224))

        preds = model.predict(image)
        results = imagenet_utils.decode_predictions(preds)
        data['predictions'] = []
        
        for (imagenetID, label, prob) in results[0]:
            r = {'label': label, 'probability': float(prob)}
            data['predictions'].append(r)

        data['success'] = True
    return flask.jsonify(data)

if __name__ == "__main__":
    load_model()
    app.run(host='0.0.0.0')
