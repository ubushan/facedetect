"""
Face Recognition
https://face-recognition.readthedocs.io/en/latest/usage.html
"""

import face_recognition

# Loads the image into numpy array
image = face_recognition.load_image_file(r"faces/people.jpeg")

# Find all the faces in the image
face_locations = face_recognition.face_locations(image)
# print('Faces detected: %s' % len(face_locations))

# Find the facial features in the image
face_landmarks_list = face_recognition.face_landmarks(image)

# for i in face_landmarks_list:
#     print(i)

# Get face encodings for each face in the image
list_of_face_encodings = face_recognition.face_encodings(image)
# print(list_of_face_encodings)

"""
Face encodings can be compared against each other to see if the faces are a match. 
Note: Finding the encoding for a face is a bit slow, so you might want to save 
the results for each image in a database or cache if you need to refer back to it later.
"""

# Results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces()