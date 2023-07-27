#face blur
import cv2

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (face) from the frame
        face = frame[y:y+h, x:x+w]
        
        # Apply a blur effect to the surrounding area of the face
        blurred = cv2.GaussianBlur(frame, (99, 99), 0)
        
        # Replace the surrounding area of the face with the blurred frame
        frame[y:y+h, x:x+w] = blurred[y:y+h, x:x+w]

    # Display the result
    cv2.imshow('Face Detection with Blur', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()