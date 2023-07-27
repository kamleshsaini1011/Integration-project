import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from keras.models import model_from_json
from cvzone.HandTrackingModule import HandDetector
import pyautogui
import cv2
import copy
import webbrowser
from cvzone.FaceDetectionModule import FaceDetector
from scipy.ndimage import gaussian_filter
import tkinter as tk
import joblib as jb
from keras.preprocessing import image
from PIL import Image
from FaceRecognition import Camera
from LinearReggression import LinearRegg
from Logg import LogRegg
from blurrface import BlurrtheFace
from Emotion import EmotionDetection
from HandDetection import HandDetect


def face():
    cap = cv2.VideoCapture(0)

# goggles on Multiple face

# Load the face cascade XML file for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the goggles image with transparency
    goggles_img = cv2.imread('YO.png', cv2.IMREAD_UNCHANGED)
    while True:
# Read the current frame from the video stream
        status, photo = cap.read()

        
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
        # cv2.rectangle(photo, (x, y), (x + w, y + h), [0, 255, 0], 5)

    # Display the modified frame
            cv2.imshow("Video", photo)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

def DetectDistance():
    REFERENCE_OBJECT_WIDTH = 0.15  # 15 centimeters
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    focal_length = 1000.0
    principal_point = (640, 480)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_width_pixels = w
            distance = (REFERENCE_OBJECT_WIDTH * focal_length) / face_width_pixels

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Distance: {distance:.2f} meters", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == 13:
            break

    cap.release()
    cv2.destroyAllWindows() 

    # Function to launch the website

def launch_website():
    url = "abc.html"  # Replace with the URL of your local website
    webbrowser.open(url, new=2)  # Open the URL in a new browser window or tab

    def main():
        # Set up webcam
        cap = cv2.VideoCapture(0)
        detector = FaceDetector()

        while True:
            success, img = cap.read()
            img, bboxs = detector.findFaces(img)

            if bboxs:
                # Launch the website if a face is detected
                launch_website()
                break

            cv2.imshow("Face Detection", img)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the loop
                break 
        cap.release()
        cv2.destroyAllWindows()

        if __name__ == "__main__":
             main()


def Volume():
    detector = HandDetector(detectionCon=0.8)
    min_volume = 0
    max_volume = 100
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        hands, frame = detector.findHands(frame)
        if hands:

            for hand in hands:
                landmarks = hand["lmList"]
                bbox = hand["bbox"]
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                thumb_index_distance = np.linalg.norm(np.subtract(thumb_tip, index_tip))
                volume = np.interp(thumb_index_distance, [20, 200], [min_volume, max_volume])
                volume = int(max(min(volume, max_volume), min_volume))
                pyautogui.press('volumedown') if volume < 50 else pyautogui.press('volumeup')
                cv2.putText(frame, f"Volume: {volume}%", (bbox[0], bbox[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.imshow("Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def draw_ashoka_chakra(ax, center_x, center_y, radius):
    # Draw the central blue circle
    circle = plt.Circle((center_x, center_y), radius, color='blue', fill=True)
    ax.add_patch(circle)

    # Draw 24 spokes in the Chakra
    for i in range(24):
        angle = i * 15
        start_x = center_x + 0.45 * radius * np.cos(np.deg2rad(angle))
        start_y = center_y + 0.45 * radius * np.sin(np.deg2rad(angle))
        end_x = center_x + radius * np.cos(np.deg2rad(angle))
        end_y = center_y + radius * np.sin(np.deg2rad(angle))
        ax.plot([start_x, end_x], [start_y, end_y], color='blue', linewidth=1)

    # Draw the small blue circles in the Chakra
    small_circle_radius = radius / 7
    for i in range(4):
        for j in range(6):
            angle = (j * 15) + (i % 2) * 7.5
            x = center_x + 0.7 * radius * np.cos(np.deg2rad(angle))
            y = center_y + 0.7 * radius * np.sin(np.deg2rad(angle))
            small_circle = plt.Circle((x, y), small_circle_radius, color='blue', fill=True)
            ax.add_patch(small_circle)

def draw_indian_flag():
    width, height =805,700
    myimage = np.ones((height, width, 3), dtype=np.uint8) * 255
    myimage[1:225] = [51,153,253]         #for row 
    myimage[225:470] = [255,255,255]
    myimage[470:800]=[8,136,19]
    center=(400,350)
    radius=120
    color=[128,0,0]
    thickness=8
    cv2.circle(myimage, center, radius, color, thickness)
    # Calculate the dimensions of the flag
    stripe_height = height //2
    ashoka_chakra_radius = stripe_height //3
    # Draw the Ashoka Chakra (Navy Blue Circle)
    center_x = width //2
    center_y = stripe_height 
    center = (center_x, center_y)
    cv2.circle(myimage, center, ashoka_chakra_radius,[255,255,255],-1)
    num_lines = 24
    angle = 0
    angle_increment = 360 // num_lines
    for _ in range(num_lines):
        end_x = int(center_x + ashoka_chakra_radius * np.cos(np.deg2rad(angle)))
        end_y = int(center_y + ashoka_chakra_radius * np.sin(np.deg2rad(angle)))
        cv2.line(myimage, center, (end_x, end_y), [128,0,0], thickness=3)
        angle += angle_increment
    cv2.imshow("myimage",myimage)
    cv2.waitKey()
    cv2.destroyAllWindows()    

def DogOrCat():
    loaded_classifier = jb.load("DogvsCat.model")
    a=input("Enter name of the image ")
    new_image_path = a+r'.jpg'
    predicted_class = predict_class(new_image_path, loaded_classifier)
    print(f"The predicted class is: {predicted_class}")

def hello(x):
	# only for referece
	print("")

        
def Crop():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Step 4: Capture Video from Webcam (Optional)
    cap = cv2.VideoCapture(0)
    # Step 5: Process the Video Stream or Load an Image
    while True:
        ret, img = cap.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Step 6: Perform Face Detection
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # Step 7: Draw Rectangles around Detected Faces and Display in a Window
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Crop the face region and display it in a separate window
            face_roi = img[y:y + h, x:x + w]
            cv2.imshow('Detected Face', face_roi)
        cv2.imshow('Face Detection', img)
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def BlurrtheSurr():
    cap = cv2.VideoCapture(0)
    face_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    while True:
        ret, photo = cap.read()

        if not ret:
            break

        faces = face_model.detectMultiScale(photo)

        # Apply Gaussian blur to the background
        blurred_photo = blur_background(photo.copy(), faces)

        # Draw green rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(blurred_photo, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Blurred Background", blurred_photo)

        if cv2.waitKey(1) & 0xFF == 13:
            break

    cv2.destroyAllWindows()
    cap.release()

def blur_background(image, faces):
    # Create a mask to separate the face region from the background
    mask = image.copy()
    mask[:] = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
    inverted_mask = cv2.bitwise_not(mask)

    # Apply Gaussian blur only to the background
    blurred_background = cv2.GaussianBlur(image, (23, 23), 30)
    result = cv2.bitwise_and(image, mask) + cv2.bitwise_and(blurred_background, inverted_mask)
    return result

def BlurrtheFace():
    cap = cv2.VideoCapture(0)
    face_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    while True:
        ret, photo = cap.read()

        if not ret:
            break

        faces = face_model.detectMultiScale(photo)

        # Create a copy of the original photo
        blurred_photo = photo.copy()

        # Apply Gaussian blur to the face regions
        blurred_photo = blur_face(blurred_photo, faces)

        cv2.imshow("Blurred Face", blurred_photo)

        if cv2.waitKey(1) & 0xFF == 13:
            break

    cv2.destroyAllWindows()
    cap.release()

def blur_face(image, faces):
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_roi, (23, 23), 30)
        image[y:y+h, x:x+w] = blurred_face
    return image

def execute_selected_task():
    selected_task = int(choice_var.get())
    if selected_task == 1:
        LinearRegg()
    elif selected_task == 2:
        LogRegg()
    elif selected_task == 3:
        HarryPoterCloak()
    elif selected_task == 4:
        BlurrtheFace()
    elif selected_task == 5:
        BlurrtheSurr()
    elif selected_task == 6:
        DetectDistance()
    elif selected_task == 7:
        EmotionDetection()
    elif  selected_task==8:
        draw_indian_flag()
    elif selected_task==9:
        DogOrCat()
    elif selected_task==10:
        HandDetect()
    elif selected_task==11:
        Volume()
    elif selected_task==12:
        Crop()
    elif selected_task==13:
        launch_website()
    elif selected_task==15:
        face()
# Create the Tkinter GUI
root = tk.Tk()
root.title("Task Menu")
window_width = 900
window_height = 1000
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = int((screen_width - window_width) / 2)
y_coordinate = int((screen_height - window_height) / 2)
root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

# Create a label to display the menu options
label = tk.Label(root, text="Team 10 \n Solution Seekers")
label.pack()

# Create a variable to hold the user's choice
choice_var = tk.StringVar()
# Create radio buttons for the user to select a task
tk.Radiobutton(root, text="Detect Distance From Camera", variable=choice_var, value="6").pack(anchor=tk.W)
tk.Radiobutton(root, text="Blurring the Face only ", variable=choice_var, value="4").pack(anchor=tk.W)
##tk.Radiobutton(root, text="SunGlass", variable=choice_var, value="15").pack(anchor=tk.W)
tk.Radiobutton(root, text="Increase Volume UP & Down ", variable=choice_var, value="11").pack(anchor=tk.W)
tk.Radiobutton(root, text="Blurring the Surrounding not face ", variable=choice_var, value="5").pack(anchor=tk.W)
tk.Radiobutton(root, text="Hand Detection ", variable=choice_var, value="10").pack(anchor=tk.W)
tk.Radiobutton(root, text="Open My Website ", variable=choice_var, value="13").pack(anchor=tk.W)
# Create a button to execute the selected task
execute_button = tk.Button(root, text="Execute Task", command=execute_selected_task)
execute_button.pack()

# Start the Tkinter main loop
root.mainloop()