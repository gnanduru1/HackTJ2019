import numpy as np
import cv2 as cv
import imutils
face_cascade = cv.CascadeClassifier("C:/Users/Srikar/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")#cv.CascadeClassifier('haarcascade_frontalface_default.xml')
for i in range(8):
    i = i+1
    name = "ritesh/"+str(i)+".jpg"
    img = cv.imread(name)
    img = imutils.resize(img,width = 500)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    img = imutils.resize(img, width=600)
    cv.imshow('img',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
