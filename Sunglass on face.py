import cv2
import numpy as np
cap= cv2.VideoCapture(0)

# goggles on Multiple face 

# Load the face cascade XML file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the goggles image with transparency
goggles_img = cv2.imread('YO.png', cv2.IMREAD_UNCHANGED)

while True:
    # Read the current frame from the video stream
    status, photo = cap.read()

    if not status:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Calculate the position and size of the face region
        face_roi = photo[y:y + h, x:x + w]

        # Resize the goggles image to match the dimensions of the face region
        goggles_resized = cv2.resize(goggles_img, (w, h))

        # Extract the alpha channel of the goggles image
        alpha = goggles_resized[:, :, 3] / 255.0

        # Create a mask for the goggles region
        mask = alpha.astype(np.uint8)

        # Apply the mask to remove the goggles region from the face
        bg_removed = cv2.bitwise_and(face_roi, face_roi, mask=(1 - mask))

        # Overlay the resized goggles image onto the face region
        output = bg_removed + cv2.bitwise_and(goggles_resized[:, :, :3], goggles_resized[:, :, :3], mask=mask)

        # Replace the face region with the modified output
        photo[y:y + h, x:x + w] = output

        # Draw a rectangle around the face
        #cv2.rectangle(photo, (x, y), (x + w, y + h), [0, 255, 0], 5)

    # Display the modified frame
    cv2.imshow("Video", photo)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()