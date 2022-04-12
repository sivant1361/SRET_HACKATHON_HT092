import numpy as np
import matplotlib.pyplot as plt
import random
from flask import Flask, request, render_template, redirect, flash, send_from_directory

import cv2

import tensorflow as tf
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# config = {
#     "DEBUG": True  # run app in debug mode
# }
# app.config.from_mapping(config)
model = tf.keras.models.load_model("./models/emotion_model.h5")

face_detector = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')

emotiondict = {
    "0": "Angry",
    "1": "Disgust",
    "2": "Fear",
    "3": "Happy",
    "4": "Neutral",
    "5": "Sad",
    "6": "Surprise"
}


def fun(img):
    faceCascade = face_detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=1,
        minSize=(10, 10))

    print("Found {0} faces".format(len(faces)))
    pred = dict()

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        tmp = cv2.resize(img[y:y+h, x:x+w], (48, 48),
                         interpolation=cv2.INTER_AREA)
        tpred = model.predict(np.array([tmp]))
        pred[str([x, y, w, h])] = emotiondict[str(np.argmax(tpred[0]))]

    a = dict()
    for i in pred:
        if pred[i] in list(a.keys()):
            a[pred[i]] += 1
        else:
            a[pred[i]] = 1
    plt.bar(list(a.keys()), a.values())
    plt.savefig("static/bar.png")
    return (a)


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/emotion')
def emotion():
    return render_template('emotion.html', Name="Emino")


@app.route('/classroom')
def classroom():
    return render_template('classroom.html', Name="Emino")


@app.route('/emotion_predict', methods=['POST'])
def emotion_predict():

    try:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No file selected")
        if file:
            filename = secure_filename(file.filename)
            print(file)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            face = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            print(face, filename, file)
            gray_frame = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            num_faces = face_detector.detectMultiScale(
                gray_frame, scaleFactor=1.3, minNeighbors=1)

            index_val = 0
            for (x, y, w, h) in num_faces:
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                backtorgb = cv2.cvtColor(roi_gray_frame, cv2.COLOR_GRAY2RGB)
                cropped_img = np.expand_dims(
                    cv2.resize(backtorgb, (48, 48)), 0)

                pred = model.predict(cropped_img)
                index_val = str(np.argmax(pred))
                num = str(random.randrange(1, 10))

            return render_template(
                'emotion_predict.html',
                Name="Emino",
                index=index_val,
                type=emotiondict[index_val],
                myimage=file.filename,
                emotionimg="Emotions/%s/%s " % (emotiondict[index_val], emotiondict[index_val])+num+".jpg")
    except Exception as e:
        print(e)
        return redirect("404")


@app.route('/classroom_predict', methods=['POST'])
def classroom_predict():

    try:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No file selected")
        if file:

            filename = secure_filename(file.filename)
            print(file)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            face = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(face)
            res = fun(face)
            dom = list(res.keys())[np.argmax(res.values())]
            return render_template(
                'classroom_predict.html',
                Name="Emino",
                type=dom,
                myimage=file.filename,
                emotionimg="bar.png")

    except Exception as e:
        print(e)
        return redirect("404")


@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/404")
def page404():
    return render_template('404.html')


@app.route('/display/<filename>')
def send_uploaded_file(filename=''):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
