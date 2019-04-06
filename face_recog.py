import cv2
import os
import numpy as np

subjects = ["", "ritesh", "srikar"]
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def detect_face(img):
#convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('C:/Users/ganes/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/lbpcascade_frontalface.xml')

    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]
def train():
    faces = []
    labels = []
    int = 1
    for stringo in ["ritesh", "srikar"]:
        for file in os.listdir(stringo):
            full_name = stringo + "/" + file
            face, rectangle = detect_face(cv2.imread(full_name))
            if face is not None:
                faces.append(face)
                labels.append(int)
    int += 1
    return faces, labels
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)
    label = face_recognizer.predict(face)[0]
    print(label)
    #predict the image using our face recognizer
    #get name of respective label returned by face recognizer
    label_text = subjects[label]
    return img

    face_recognizer.train(faces, np.array(labels))
    #draw_rectangle(img, rect)
    #draw_text(img, label_text, rect[0], rect[1]-5)
faces, labels = train()
face_recognizer.train(faces, np.array(labels))
#face_recognizer.train(faces, np.array(labels))
test_img1 = cv2.imread("ritesh/1.jpg")
test_img2 = cv2.imread("srikar/2.jpg")

#perform a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
print("Prediction complete")

#display both images
import imutils
img1 = imutils.resize(predicted_img1, width=600)
img2 = imutils.resize(predicted_img2, width=600)

cv2.imshow(subjects[1], img1)
cv2.waitKey(5000)
cv2.destroyAllWindows()

cv2.imshow(subjects[2], img2)
cv2.waitKey(5000)
cv2.destroyAllWindows()
