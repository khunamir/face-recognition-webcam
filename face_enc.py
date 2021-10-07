from imutils import paths
import cv2
import face_recognition
import pickle
import os

# Get path of each faces in a folder
facePath = list(paths.list_images('Faces'))
knownEncodings = []
knownNames = []

# Loop over image paths
for (i, imgPath) in enumerate(facePath):
    # Extract the name of the person from the path
    name = imgPath.split(os.path.sep)[-2]

    # Load the input image and covert from BGR to RGB
    img = cv2.imread(imgPath)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use face_recognition to locate faces
    face_loc = face_recognition.face_locations(rgb, model='hog')  # Wth is hog?

    # Compute the facial encodings fro the face
    encodings = face_recognition.face_encodings(rgb, face_loc)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

# Save encodings along with their names in data dictionary
data = {'encodings': knownEncodings, "names": knownNames}

# Use pickle to save data for later use
f = open('face_enc', 'wb')
f.write(pickle.dumps(data))
f.close()