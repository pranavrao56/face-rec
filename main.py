import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Open the default camera (index 0)
video_capture = cv2.VideoCapture(0)

# Check if camera is opened successfully
if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load face encodings and names
biden_image = face_recognition.load_image_file("photos/biden.jpg")
biden_encoding = face_recognition.face_encodings(biden_image)[0]
 
obama_image = face_recognition.load_image_file("photos/obama.jpg")
obama_encoding = face_recognition.face_encodings(obama_image)[0]
 
trump_image = face_recognition.load_image_file("photos/trump.jpg")
trump_encoding = face_recognition.face_encodings(trump_image)[0]
 
known_face_encoding = [
biden_encoding,
obama_encoding,
trump_encoding
]
 
known_faces_names = [
"Joe Biden",
"Barack Obama",
"Donald Trump"
]

students = known_faces_names.copy()

# Create CSV file for writing attendance
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(current_date+'.csv','a',newline='')
lnwriter = csv.writer(f)

while True:
    # Capture frame from camera
    ret, frame = video_capture.read()

    # Resize frame for faster processing
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
    face_names = []

    for face_encoding in face_encodings:
        # Compare face encoding with known face encodings
        matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
        name = ""
        face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        face_names.append(name)

        if name in known_faces_names:
            # Display name on the frame and write to CSV file
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,100)
            fontScale = 1.5
            fontColor = (255,0,0)
            thickness = 3
            lineType = 2

            cv2.putText(frame, name+' Present',
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

            if name in students:
                students.remove(name)
                print(students)
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, current_date, current_time])  # Include date and time in CSV
                f.flush()  # Add a flush to ensure that the data is written immediately to the file

    # Display the frame
    cv2.imshow("attendance system", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
f.close()