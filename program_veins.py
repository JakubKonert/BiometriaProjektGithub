#from skimage.io import imread_collection
#from picamera.array import PiRGBArray
#from picamera import PiCamera
from matplotlib import pyplot as plt
import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
import os
import glob
import fnmatch
a = 0
 
 
cap = cv2.VideoCapture(-1)
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 640,400)
cap.set(3,320)
cap.set(4,200)
while(cap.isOpened()):
    # capture frame-by-frame
    ret, frame = cap.read()
 
    if ret:
        fram = cv2.flip(frame, 0)
        gray = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((7,7),np.uint8)
        equ = cv2.equalizeHist(gray)
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
        cl1 = clahe.apply(equ)
        blur = cv2.GaussianBlur(cl1, (7, 7), 0)
        img_gray = blur
 
 
        #files = next(os.walk("/home/pi/plate/"))#.__next__()
        files = os.listdir("/home/pi/plate/")
        onlyfiles = len(files)
        #print (onlyfiles)
        if onlyfiles > 0: #no. of files in plate
            zx = onlyfiles #range
            #print (zx)
            #for imgx in glob.glob('/home/pi/plate/*.jpg'):
            for i in range(zx):
                #print(i)
 
                #for img in glob.glob("/pi/plate/*.jpg"):
                #template = cv2.imread(imgx)
                template = cv2.imread('/home/pi/plate/pic'+str(i)+'.jpg', 0)
                w, h = template.shape[::-1]
                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
 
                threshold = 0.90
                glitch1 = 0.50
                glitch2 = 0.99
                print(res)
 
                loc = np.where(res >= threshold)
 
 
 
                if  res>=threshold: #and glitch2>=res>=glitch1:
                    font= cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.50
                    font_thickness = 1
                    cv2.putText(img_gray,'FOUND',(0,15),font,font_scale, (200,255,155),font_thickness)
                    #cv2.putText(img_gray,"template",(0,20),font,font_scale, (200,255,155),font_thickness)

                cv2.rectangle(img_gray, (55,0),(250,200),(255,0,0),2)       
                cv2.rectangle(img_gray, (0,20),(53,240),(0,0,0),-2,4)
                cv2.rectangle(img_gray, (252,0),(320,240),(0,0,0),-2,4)
                cv2.imshow("Frame", img_gray)
        #else:  
        cv2.rectangle(img_gray, (55,0),(250,200),(255,0,0),2)       
        cv2.rectangle(img_gray, (0,20),(53,240),(0,0,0),-2,4)
        cv2.rectangle(img_gray, (252,0),(320,240),(0,0,0),-2,4)
        #cv2.rectangle(img_gray, (55,5),(250,195),(255,0,0),2)
        cv2.imshow("Frame", img_gray)
        key = cv2.waitKey(0)