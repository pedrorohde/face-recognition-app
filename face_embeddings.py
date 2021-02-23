import face_recognition
import numpy as np

import os

face_embeddings = []
face_names = []

for filename in os.listdir('faces'):
    image = face_recognition.load_image_file('faces/'+filename)
    face_embedding = face_recognition.face_encodings(image, model='small')[0]
    face_embeddings.append(face_embedding)
    face_names.append(os.path.splitext(filename)[0])

np.save('embeddings.npy', face_embeddings)
np.save('names.npy', face_names)