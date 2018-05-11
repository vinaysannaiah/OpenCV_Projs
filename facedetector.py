# Face detector

#============================================================================#
#import libraries
#============================================================================#

import cv2

#============================================================================#
#============================================================================#


#============================================================================#
#Load the necessary files
#1. Haar cascade file for frontal face detection
#============================================================================#

Face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#============================================================================#
#============================================================================#


#============================================================================#
# Functions
#============================================================================#

def face_detection(gray,frame):
    
    faces = Face_detect.detectMultiScale(gray, 1.3, 5)
    
    for (x,y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2) # create a rectangle around the face
#        roi_gray = gray[y:y+h, x:x+w]
#        roi_color = frame[y:y+h, x:x+w]
          
    return frame

#===========================================================================#
#===========================================================================#


# Turn on the Camera
video_capture = cv2.VideoCapture(0)

while True: # to run continously until an esc button is pressed
    _,frame = video_capture.read() # to get the most recent frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the frame into a gray scale for simplied processing
    video_display = face_detection(gray, frame) # create a rectange in the frame around the face
    cv2.imshow("Face Detection", video_display) # display the frame with the face detection
    if cv2.waitKey(1) & 0xFF == ord('a'): # keep running until an esc key is pressed
        break
    
    
video_capture.release() # turn off the camera or release the video
cv2.destroyAllWindows() # destroy the window

