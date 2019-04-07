import cv2
#from newface_test import findface
import pickle
import time
import face_recognition
import imutils
cascPath = "C:/Users/Srikar/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
x,y = 0,0
video_capture = cv2.VideoCapture(0)

print("[INFO] loading encodings...")
data = pickle.loads(open("encodings.pickle", "rb").read())

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if True:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model = "hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        print(len(encodings))
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)
            names.append(name)

        print(names)
        if len(names)>0:

            cv2.putText(frame, str(names), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imshow('Video', frame)
        cv2.waitKey(1)
video_capture.release()
cv2.destroyAllWindows()