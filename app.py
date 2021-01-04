import os
import time
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
import pickle

import autokeras as ak
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2  
import keras 
from shutil import rmtree
import shutil

from skimage.io import imread
from skimage.transform import resize


app = Flask(__name__)


# IMAGE_UPLOADS = 'static/upload'
app.config['IMAGE_UPLOADS'] = 'static/upload'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ________________________________ Loading Models __________________________#

# Loading keras model for Nature Images
keras_model = load_model("./data/Nature_model.h5", custom_objects=ak.CUSTOM_OBJECTS)
labels = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# Loading model for dog cat prediction
model_pet = load_model('./data/cat_dog_predict_model.h5',custom_objects=ak.CUSTOM_OBJECTS)

# categories = ['bougainvillea', 'gardenias', 'garden_roses', 'hibiscus', 'hydrangeas']
# with open('./data/img_model11.pkl','rb') as f:
#     model = pickle.load(f)

# __________________ Creating Functions _______________________________#
# preprocess image
def pre(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, grayscale=False, color_mode="rgb", 
                                                  target_size=(180,180,3), interpolation="nearest")
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    
    predictions = keras_model.predict(input_arr)
    return predictions

def preprocess(path):
    image = tf.keras.preprocessing.image.load_img(path, grayscale=False, color_mode="rgb", 
                                                  target_size=(64,64,3), interpolation="nearest")
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    
    pred = model_pet.predict(input_arr).round(2)
    pred = pred[0][0]
    return pred



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify_image', methods=['POST'])
def classify_image():
    dir = './static/upload'
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image.save(os.path.join(app.config['IMAGE_UPLOADS'], image.filename))
            print("Image saved")
            print(image.filename)
            f = image.filename
            path = './static/upload/'+f
            print("_______",path)
    
        pred = pre(path)
        pred_categories = tf.argmax(pred, axis=1)
        ar = np.asarray(pred_categories)
        ar = ar[0]
        res = labels[ar]
        print("________________________________________________________")
        print("Predicted class is : ",res)
        print("________________________________________________________")
        pr = pred.round(2)*100
        # print(pr[0][0])
        # print(pr[0][1])
        # print(pr[0][2])
        # print(pr[0][3])
        # print(pr[0][4])
        # print(pr[0][5])
        # print(labels[0])
        # print(labels[1])
        # print(labels[2])
        # print(labels[3])
        # print(labels[4])
        # print(labels[5])
        print("________________________________________________________")
        
    return render_template('index.html',result=res,im=f,clas = labels,pr=pr,st=True)

# ___________________________________________________________________________________________________#

@app.route('/classify_image1', methods=['POST'])
def classify_image1():
    dir = './static/upload'
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

    if request.method == "POST":
        if request.files:
            image1 = request.files["image1"]
            image1.save(os.path.join(app.config['IMAGE_UPLOADS'], image1.filename))
            print("Image saved")
            print(image1.filename)
            f = image1.filename
            path1 = './static/upload/'+f
            print("_______",path1)
    


        # print(path1)
        pred = preprocess(path1)
        if pred <= 0.5:
            res = "Cat"
        if pred > 0.5:
            res = "Dog"
        print("________________________________________________________")
        print("Predicted class is : ",res)
        print("Predicted class is : ",pred)
        print("________________________________________________________")
        
    return render_template('index.html',result=res,im=f,pr=pred,st1=True)
    







if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001, threaded=True)