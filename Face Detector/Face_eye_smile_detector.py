# -*- coding: utf-8 -*-
"""
@author: Vinay Sannaiah
"""
#Libraries
import cv2 # import openCV

#Load the cascades
face_cascade  =  cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Haar cascade file for face detection
eye_cascade   =  cv2.CascadeClassifier('haarcascade_eye.xml') # Haar cascade file for eye detcction
smile_cascade =  cv2.CascadeClassifier('haarcascade_smile.xml') # Haar cascade file for smile detection

#Functions

def detect(color_img, gray_img):
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.3, minNeighbors = 5)
    for (x, y, w, h) in faces:
        #draw the rectangle
        cv2.rectangle(img = color_img, pt1 = (x,y),
                      pt2 = (x+w, y+h), color = (255,100,100), thickness = 3)
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = color_img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.2, minNeighbors = 20)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img = roi_color, pt1 = (ex,ey),
                      pt2 = (ex+ew, ey+eh), color = (100,255,100), thickness = 2)    
        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor = 1.7, minNeighbors = 15) # ideal 22
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(img = roi_color, pt1 = (sx,sy),
                      pt2 = (sx+sw, sy+sh), color = (100,100,255), thickness = 2)    
    return color_img

video = cv2.VideoCapture(0) # takes the arg 0 - Internal webcam and 1 - External webcam
while True:
    _,frame = video.read() #(underscore): we wont be using the first element that is returned by video capture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converting the last frame to gray value
    detection = detect(frame, gray) # call the detect function to do the necessary detections and to add the bounding box around it.
    cv2.imshow('Video', detection) # display the webcam video with the bounding boxes.
    if cv2.waitKey(1) & 0xFF == ord('q'): # close the webcam window when 'q' is pressed.
        break
    
video.release() # release the webcam
cv2.destroyAllWindows() # destroy all the windows
    
