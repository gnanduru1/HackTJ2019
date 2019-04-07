import face_recognition

known_image = face_recognition.load_image_file("test/joe-biden.jpg")
unknown_image = face_recognition.load_image_file("test/joe-biden2.jpg")
new_image = face_recognition.load_image_file("test/asian.png")
#known_face_locations=[(0, width, height, 0)]

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
new_encoding = face_recognition.api.face_encodings(new_image, known_face_locations=[(0,182,182,0)], num_jitters=1)

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
results2 = face_recognition.compare_faces([new_encoding], unknown_encoding)

print(results)
print(results2)
