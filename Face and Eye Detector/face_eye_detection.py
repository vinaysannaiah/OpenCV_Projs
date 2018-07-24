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
    
    return color_img
