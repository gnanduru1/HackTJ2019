import numpy as np
import cv2 as cv
import cv2
import tensorflow
import imutils
import os

face_cascade = cv.CascadeClassifier("C:/Users/Srikar/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")#cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

name_to_ID = {"ritesh":1,"srikar":2}
ID_to_name = {1:"ritesh",2:"srikar"}

faces_array = []
IDs = []
folders = ['srikar','ritesh']
for folder in folders:
    for filename in os.listdir(folder):
        if '.jpg' not in filename:
            continue
        name = folder + "/" + filename
#        print(name)
        img = cv.imread(name)
        img = imutils.resize(img, width=500)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#        cv.imshow('img',img)
#        cv.waitKey(0)
#        cv.destroyAllWindows()

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            print("No face detected, removing " + str(name))
            os.remove(name)
            continue
        elif len(faces) > 1:
            print("Multiple faces detected in " + str(name))
            continue
        (x,y,w,h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        faces_array.append(roi_gray)
        IDs.append(name_to_ID[folder])
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#        cv.imshow('img',img)
#        cv.waitKey(0)
#        cv.destroyAllWindows()

def predict(recognizer,test_img):
    img = cv.imread(test_img)
    img = imutils.resize(img, width=500)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    (x,y,w,h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    label,confidence = recognizer.predict(roi_gray)
    print(label)
    label_text = ID_to_name[label]
    return label_text

#img = imutils.resize(img, width=600)
face_recognizer.train(faces_array,np.array(IDs))
#face_recognizer.save('recognizer/trainingData.yml')
print(predict(face_recognizer,"8.jpg"))
