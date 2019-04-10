# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:55:39 2019

@author: Daniel
"""

import cv2
import numpy as np
import pyautogui as pag
import time

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
#eye_cascadel = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
host = 'http://192.168.178.48:8080/shot.jpg'
cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1280)

counter = 0
maxFaceHeight = 0
mean = 0
std = 0
calibration = 0
cx = np.ones([4], dtype=float)
cy = np.ones([4], dtype=float)
rcx = np.zeros([4], dtype=float)
rcy = np.zeros([4], dtype=float)
sw,sh = pag.size()
px = 1
py = 1
AscentArea = 2
cutImagex = 450
cutImagey = 150
Eyey = 35
Eyex = 60
disx = 5
disy = 20

#topright = []
#topmid = []
#topleft = []
#toprightbet = []
#topleftbet = []
#midright = []
#midmid = []
#midleft = []
#botrightbet = []
#botleftbet = []
#bottright = []
#bottmid = []
#bottleft = []

markers = 100 * np.ones(shape=[1080,1920,1], dtype=np.uint8)
cv2.circle(markers, (0,0), 2, (255,255,255),5)
cv2.circle(markers, (1920,0), 2, (255,255,255),5)
cv2.circle(markers, (0,1080), 2, (255,255,255),5)
cv2.circle(markers, (1920,1080), 2, (255,255,255),5)
cv2.circle(markers, (int(1920/2),int(1080/2)), 2, (255,255,255),5)
cv2.circle(markers, (int(1920/2),0), 2, (255,255,255),5)
cv2.circle(markers, (0,int(1080/2)), 2, (255,255,255),5)
cv2.circle(markers, (int(1920/2),1080), 2, (255,255,255),5)
cv2.circle(markers, (1920,int(1080/2)), 2, (255,255,255),5)
cv2.circle(markers, (int(1920/4),int(1080*0.75)), 2, (255,255,255),5)
cv2.circle(markers, (int(1920*0.75),int(1080/4)), 2, (255,255,255),5)
cv2.circle(markers, (int(1920/4),int(1080/4)), 2, (255,255,255),5)
cv2.circle(markers, (int(1920*0.75),int(1080*0.75)), 2, (255,255,255),5)
cv2.namedWindow("Markers", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Markers", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

t = time.clock()
while True:
    re, img = cap.read()
#    stream = urlopen(host)
#    imgNp = np.array(bytearray(stream.read()), dtype=np.uint8)
#    img = cv2.imdecode(imgNp,-1)
    img = img[cutImagey:img.shape[0]-cutImagey,cutImagex:img.shape[1]-cutImagex]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(9,9),1)
    clahe = cv2.createCLAHE(clipLimit = 4.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    face = face_cascade.detectMultiScale(img)
    if type(face) is not tuple:
        for (x,y,w,h) in face:
            rightface = gray[0:y+h,0:int(x+w*0.45)]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        
#    t = time.clock() - t
#    print(t)
    
    eyes = eye_cascade.detectMultiScale(rightface)
    if type(eyes) is not tuple:
        for(x,y,w,h) in eyes:
#            cv2.rectangle(rightface, (x,y),(x+w,y+h),(0,255,0),2)
            yf = y
            xf = x
            hf = h
            wf = w
            reye = img[int(y+disy):int(y+disy+Eyey),int(x+disx):int(x+disx+Eyex)]
#            cv2.rectangle(img, (int(x+disx),int(y+disy)), (int(x+disx+Eyex),int(y+disy+Eyey)), (0,255,0))
            #0.5 0.9 0.2 0.9
        
#    cv2.circle(reye,(x,y), 1, (255,255,255))
    cx[3] = int(x+xf+0.2*wf)
    cy[3] = int(y+yf+0.5*hf)
#    cv2.circle(img,(int(x+xf),int(y+yf+0.3*hf)), 2, (0,0,255))
    cv2.normalize(reye, reye, 0,255,cv2.NORM_MINMAX)
    scaled = cv2.resize(reye, (reye.shape[1]*6,reye.shape[0]*6), interpolation = cv2.INTER_CUBIC)
    cv2.imshow('Eye',scaled)
    cv2.imshow("Markers",markers)
    
    k = cv2.waitKey(20) & 0xff
    
    if k == 27:
        break
    elif k == ord('7'):
        print(calibration)
    if (time.clock() - t) >= 1:
        botrightbet.append(reye)
    

cap.release()
cv2.destroyAllWindows()