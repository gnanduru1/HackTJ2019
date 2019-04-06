import cv2 as cv
import cv2
import os
import imutils

face_cascade = cv.CascadeClassifier("C:/Users/Srikar/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")#cv.CascadeClassifier('haarcascade_frontalface_default.xml')
name_to_ID = {"ritesh":1,"srikar":2,"lohit":3,"ganesh":4,"kevin":5}
ID_to_name = {}
for name in name_to_ID:
    ID = name_to_ID[name]
    ID_to_name[ID] = name

def predict(recognizer,test_dir):
    labels = []
    for filename in os.listdir(test_dir):
#        print(filename)
        img = cv.imread(test_dir + filename)
        img = imutils.resize(img, width=500)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        (x,y,w,h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        label_num,confidence = recognizer.predict(roi_gray)
#        print(label_num,confidence)
        label = ID_to_name[label_num]
#        print(label)
        labels.append((filename,label))
    return labels

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("trained_network.xml")
print(face_recognizer)
print(predict(face_recognizer,"test/"))
