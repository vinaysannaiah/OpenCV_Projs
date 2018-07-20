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
