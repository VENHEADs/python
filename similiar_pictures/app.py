
from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
import os
import tqdm
import pandas as pd
import pickle
from scipy.spatial.distance import cosine
import re
import base64
import sys
import os

sys.path.append(os.path.abspath("./model"))
from load import *

app = Flask(__name__)
global model
model, dict_vector = init()
#################################
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global name_for_file
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            name_for_file = filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return render_template("index.html")



def preproc_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


@app.route('/')
def index():

    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():

    print("debug")

    x = preproc_image('uploads/' + name_for_file)
    test_vector = model.predict(x)

    def find_distance(vector):
        return cosine(test_vector, vector)

    distance = map(find_distance, dict_vector.values())

    distances = pd.DataFrame(distance)
    distances['img'] = pd.DataFrame(dict_vector.keys())
    distances.rename(columns={0: 'distance'}, inplace=True)

    result = distances.sort_values(by='distance').img.values[0:5]

    string_output = ""
    for i in result:
        string_output+=i+" "
    return string_output



if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

