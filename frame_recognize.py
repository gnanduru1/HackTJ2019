import face_recognition
import cv2
import imutils

img1 = cv2.imread("C:/Users/ganes/Pictures/facerec_pics/ganesh/0.jpg")
img2 = cv2.imread("test/ganesh.jpg")
img1 = imutils.resize(img1, width=500)
cv2.imshow("sample", img1)
cv2.imshow("test", img2)
cv2.waitKey(10000)

known_image = face_recognition.load_image_file("C:/Users/ganes/Pictures/facerec_pics/ganesh/0.jpg")
unknown_image = face_recognition.load_image_file("test/ganesh.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
