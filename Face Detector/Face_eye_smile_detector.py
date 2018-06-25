# -*- coding: utf-8 -*-
"""
Created on Thu May 10 21:26:19 2018

@author: Vinay Sannaiah
"""
#Libraries
import cv2

#Load the cascades
face_cascade  =  cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade   =  cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade =  cv2.CascadeClassifier('haarcascade_smile.xml')

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
    detection = detect(frame, gray)
    cv2.imshow('Video', detection)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()
    
