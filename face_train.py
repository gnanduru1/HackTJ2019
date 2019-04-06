import numpy as np
import cv2 as cv
import cv2
import tensorflow
import imutils
import os
from random import shuffle

face_cascade = cv.CascadeClassifier("C:/Users/Srikar/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")#cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

name_to_ID = {"ritesh":1,"srikar":2,"lohit":3,"ganesh":4,"kevin":5}
ID_to_name = {}
for name in name_to_ID:
    ID = name_to_ID[name]
    ID_to_name[ID] = name

faces_array = []
IDs = []
folders = ['srikar','ritesh','lohit','ganesh','kevin']
maindir = '../facerec_testdata'
for folder in folders:
    print(folder)
    for filename in os.listdir(maindir + "/" + folder):
        print(filename)
        if '.jpg' not in filename:
            continue
        name = maindir + "/" + folder + "/" + filename
#        print(name)
        img = cv.imread(name)
        img = imutils.resize(img, width=500)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#        cv.imshow('img',img)
#        cv.waitKey(0)
#        cv.destroyAllWindows()

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            print("No face detected in " + str(name))
#            os.remove(name)
            continue
        elif len(faces) > 1:
            print("Multiple faces detected in " + str(name))
            continue
        (x,y,w,h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        faces_array.append((roi_gray,name_to_ID[folder]))
#        IDs.append(name_to_ID[folder])
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv.imshow('img',img)
        cv.waitKey(0)
        cv.destroyAllWindows()

def shuffle_data(faces_data):
    IDs = []
    faces = []
    arr = [i for i in range(len(faces_data))]
    shuffle(arr)
    for a in arr:
        (face,ID) = faces_data[a]
        faces.append(face)
        IDs.append(ID)
    return faces,IDs

#img = imutils.resize(img, width=600)
faces,IDs = shuffle_data(faces_array)
face_recognizer.train(faces,np.array(IDs))
face_recognizer.write('trained_network.xml')
