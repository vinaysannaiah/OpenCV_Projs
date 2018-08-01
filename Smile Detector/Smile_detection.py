# Smile Detector using Open CV.
"""
Created on Sat Mar 10 21:26:19 2018

@author: Vinay Sannaiah
"""
#Libraries
import cv2 # import Open CV

#Load the cascade files for Face(mandatory), Eyes(optional), Smile(mandatory)
face_cascade  =  cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Face
eye_cascade   =  cv2.CascadeClassifier('haarcascade_eye.xml') #Eyes
smile_cascade =  cv2.CascadeClassifier('haarcascade_smile.xml') #Smile
