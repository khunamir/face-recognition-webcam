import face_recognition
import imutils
import pickle
import time
import cv2
import os
from imutils.video import WebcamVideoStream

# Load the classifier
faceCascade = cv2.CascadeClassifier('haar_face.xml')

# Load the known faces saved in 'face_enc'
data = pickle.loads(open('face_enc', 'rb').read())

print('Streaming started')

# video_capture = cv2.VideoCapture(0)
video_capture = WebcamVideoStream(src=0).start()  # imultils video streaming instead of opencv for better fps

# Loop over frames in the video file stream
while True:
    # Grab the frame from the threaded video stream
    # ret, frame = video_capture.read()
    frame = video_capture.read()  # imutils
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 
                                         scaleFactor=1.1,
                                         minNeighbors=1,
                                         minSize=(60,60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)  # Wth is this flag doing?

    # Convert the input frame from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    names = []

    # Loop over embeddings in case we have multiple embeddings for multiple faces
    for encoding in encodings:
        matches = face_recognition.compare_faces(data['encodings'], encoding)
        # Set name to Unknown if there's no match
        name = 'Unknown'

        # Check to see if we have found a match 
        if True in matches:
            matchedIdxs = [i for (i,b) in enumerate(matches) if b]
            counts = {}

            # Loop over matched indexes and maintain a count for each recognized faces
            for i in matchedIdxs:
                name = data['names'][i]
                counts[name] = counts.get(name, 0) + 1  # Python has a dictionary .get()?

            # Set name which has hoghest count
            name = max(counts, key=counts.get)

        # Update the list of names
        names.append(name)

        # Loop over recognized faces
        for ((x,y,w,h), name) in zip(faces, names):
            # Rescale the face coordinates
            # Draw the predicted face's name on the image
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, name, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,2), 2)

    cv2.imshow('Frane', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#video_capture.release()
cv2.destroyAllWindows()