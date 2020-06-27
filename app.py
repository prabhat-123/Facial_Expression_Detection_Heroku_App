import numpy as np
import tensorflow as tf
import sys
import os


from tensorflow.compat.v1 import ConfigProto
from tensorflow.python.keras.backend import set_session


tf.keras.backend.clear_session()
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


from flask import Flask,redirect,url_for,request,render_template
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define A Flask APP
app = Flask(__name__)

CATEGORIES = ['angry','disgust','fear','happy','neutral','sad','surprise']
# load model
import json
# load json and create model
json_file = open('faceexpression.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model

model.load_weights("faceexpression_weights.h5")
print("Loaded model from disk")

# model = load_model("faceexpression_weights.h5")
model.summary()
model._make_predict_function()


def model_predict(img_path,model):
    img = image.load_img(img_path,color_mode="grayscale",target_size=(48,48))
    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    # Rescaling the image(Preprocessing the image)
    x = x/255.0
    preds = model.predict(x)
    return preds

@app.route('/',methods= ['GET'])
def index():
    # Main page
    return render_template('./index.html')

@app.route('/',methods= ['POST','GET'])
def upload():
    if request.method == "POST":
        f = request.files['file']
        # save path to './uploads'
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path,'uploads',secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path,model)
        # Process your result for human
        pred_class = np.argmax(preds)
        result = CATEGORIES[pred_class]
        return render_template('./prediction.html',result=result)
    else:
        return render_template('./index.html')

if __name__ == '__main__':
    app.run(debug=True)

# Index.html original code
# {% comment %} <img id="bg" width=1200px height=900px src="{{ url_for('video_feed') }}"> {% endcomment %}