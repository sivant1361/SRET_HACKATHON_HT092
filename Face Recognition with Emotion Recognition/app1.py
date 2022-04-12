import tensorflow as tf
import pandas as pd
import numpy as np
import time
import re
import os
import datetime
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)


modelpath = 'emotion_model.h5'
cascPath = 'haarcascade_frontalface_default.xml'

# -------------------------


def recognize_attendence():
    # faceCascade = cv2.CascadeClassifier(cascPath)
    emotiondict = {
        "0": "Angry",
        "1": "Disgust",
        "2": "Fear",
        "3": "Happy",
        "4": "Neutral",
        "5": "Sad",
        "6": "Surprise"
    }
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    print(os.getcwd())
    recognizer.read("./TrainingImageLabel/Trainner.yml")
    harcascadePath = cascPath
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails"+os.sep+"StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time', 'Emotion']
    attendance = pd.DataFrame(columns=col_names)
    model = tf.keras.models.load_model(modelpath)

    while True:
        ret, im = cam.read()
        frame = im
        if ret:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            pred = dict()
            a = dict()
            for(x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
                Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                Id, conf = recognize_attendence(gray[y:y+h, x:x+w])
                tmp = cv2.resize(im[y:y+h, x:x+w], (48, 48),
                                 interpolation=cv2.INTER_AREA)
                tpred = model.predict(np.array([tmp]))
                pred[str([x, y, w, h])] = emotiondict[str(np.argmax(tpred[0]))]
                if emotiondict[str(np.argmax(tpred[0]))] in list(a.keys()):
                    a[emotiondict[str(np.argmax(tpred[0]))]] += 1
                else:
                    a[emotiondict[str(np.argmax(tpred[0]))]] = 0
                emotion = emotiondict[str(np.argmax(tpred[0]))]
                if(conf < 50):
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(
                        ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(
                        ts).strftime('%H:%M:%S')
                    aa = df.loc[df['Id'] == Id]['Name'].values
                    tt = str(Id)+"-"+aa
                    attendance.loc[len(attendance)] = [
                        Id, aa, date, timeStamp, emotion][Id, aa][os.listdir(), aa, date]

                else:
                    Id = 'Unknown'
                    tt = str(Id)
                if(conf > 75):
                    noOfFile = len(os.listdir("ImagesUnknown"))+1
                    noOfFile = 
    """
    Our utlimate objective is to identify emotional disorder and provide support for continuing education and promote mental wellbeing
    """
                    cv2.imwrite("ImagesUnknown"+os.sep+"Image"+str(noOfFile) +
                                ".jpg", im[y:y+h, x:x+w])
                cv2.putText(im, str(tt)+' ' + emotion, (x, y+h),
                            font, 1, (255, 255, 255), 2)
            attendance = attendance.drop_duplicates(
                subset=['Id'], keep='first')
        ret, buffer = cv2.imencode('.jpg', im)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # cv2.imshow('im', im)
        # if (cv2.waitKey(1) == ord('q')):
        #     break
    # ts = time.time()
    # date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    # timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    # Hour, Minute, Second = timeStamp.split(":")
    # fileName = "Attendance"+os.sep+"Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    # attendance.to_csv(fileName, index=False)
    # cam.release()
    # cv2.destroyAllWindows()
    # print("Attendance Successfull")


def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # detector=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            # eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            # faces=detector.detectMultiScale(frame,1.1,7)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Draw the rectangle around each face
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 0, 255), 2)
            #     roi_gray = gray[y:y+h, x:x+w]
            #     roi_color = frame[y:y+h, x:x+w]
            #     eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            #     for (ex, ey, ew, eh) in eyes:
            #         cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (50, 200, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(recognize_attendence(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=4000)
