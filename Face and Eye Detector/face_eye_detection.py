# Face Recognition - creating a bounding box around the Face and also around the Eyes

#-----------------------------#
########## Libraries ##########
#-----------------------------#

import cv2 # Import Open CV

#-----------------------------#

#-----------------------------#
######## Load the xmls ########
#-----------------------------#

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Load the xml for the face.
eye_cascade  = cv2.CascadeClassifier('haarcascade_eye.xml') # Load the xml file for the eyes.

#-----------------------------#

#-----------------------------#
########## Functions ##########
#-----------------------------#

"""Create a function that takes as input the image in black and white (gray_img) 
and the original image (color_img), and that will return the same image with the 
detector rectangles."""

def detect(color_img, gray_img):
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.3, minNeighbors = 5) # detect faces in the image
    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(img = color_img, pt1 = (x,y),
                      pt2 = (x+w, y+h), color = (255,100,100), thickness = 3) # draw a rectangle around the face
        roi_gray = gray_img[y:y+h, x:x+w] # region of Interest in gray_img i.e the image within the drawn rectangle
        roi_color = color_img[y:y+h, x:x+w] # region of Interest in color_img i.e the image within the drawn rectangle
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.2, minNeighbors = 20) # We apply the detectMultiScale method to locate one or several eyes in the image.
        for (ex,ey,ew,eh) in eyes: # For each detected eye:
            cv2.rectangle(img = roi_color, pt1 = (ex,ey),
                      pt2 = (ex+ew, ey+eh), color = (100,255,100), thickness = 2) # We paint a rectangle around the eyes, but inside the region of interest selected i.e., the face.
    return color_img

video = cv2.VideoCapture(0) # takes the arg: 0 - Internal webcam (OR) 1 - External webcam

# Call the function to create the bounding boxes around the face and eyes
while True:
    _, color_img = video.read() #(underscore): we wont be using the first element that is returned by video capture
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY) #Converting the last frame to the gray value
    detection = detect(color_img, gray_img) # Get the output of our detect function.
    cv2.imshow('Video', detection) # Display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we press 'q' on the keyboard:
        break # Stop the loop
        
video.release() # Turn the webcam off
cv2.destroyAllWindows() # Destroy all the windows inside which the images were displayed
