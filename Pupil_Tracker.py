# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:02:17 2019

"""

import cv2
import numpy as np
from numba import jit
import pyautogui as pag
import time

@jit(nopython=True,cache=True,fastmath=True)
def gradx(pic,grad,mean,std):
    if len(grad[:,:]) > 0:
        for y in range(len(grad[:,0])):
            for x in range(len(grad[0,:])-2):
                grad[y,x+1] = (float(pic[y,x+2]) - float(pic[y,x])) / 2.0
                
        mean = np.mean(grad)
        std = np.std(grad)
        
        for y in range(len(grad[:,0])):
            for x in range(len(grad[0,:])):
                if grad[y,x] < 0.3 * mean + std and grad[y,x] > - 0.3 * mean - std:
                    grad[y,x] = 0
                if grad[y,x] > 0:
                    grad[y,x] = 1
                elif grad[y,x] < 0:
                    grad[y,x] = -1
    return grad

@jit(nopython=True,cache=True,fastmath=True)
def evaluate(x,y,gradix,gradiy,func):
    if len(gradix[:,:]) > 0:
        for cy in range(len(gradix[:,0])):
            for cx in range(len(gradix[0,:])):
                if y != cy and x != cx:
                    dy = float(cy - y)
                    dx = float(cx - x)
                    norm = np.linalg.norm(np.array([dx,dy]))#np.sqrt(np.square(dx)+np.square(dy))
                    dy = dy/norm
                    dx = dx/norm
                    func[cy,cx] = gradiy[cy,cx] * dy + gradix[cy,cx] * dx
                    if func[cy,cx] < 0:
                        func[cy,cx] = 0
    #                func[cy,cx] = np.square(func[cy,cx])
    return np.mean(func)

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

while True:
    re, img = cap.read()
#    stream = urlopen(host)
#    imgNp = np.array(bytearray(stream.read()), dtype=np.uint8)
#    img = cv2.imdecode(imgNp,-1)
    img = img[cutImagey:img.shape[0]-cutImagey,cutImagex:img.shape[1]-cutImagex]
    t = time.clock()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(13,13),4)
#    clahe = cv2.createCLAHE(clipLimit = 1.0, tileGridSize=(8,8))
#    gray = clahe.apply(gray)
    
    face = face_cascade.detectMultiScale(img)
    if type(face) is not tuple:
        for (x,y,w,h) in face:
            rightface = gray[0:y+h,0:int(x+w*0.45)]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        
    t = time.clock() - t
    print(t)
    
    eyes = eye_cascade.detectMultiScale(rightface)
    if type(eyes) is not tuple:
        for(x,y,w,h) in eyes:
#            cv2.rectangle(rightface, (x,y),(x+w,y+h),(0,255,0),2)
            yf = y
            xf = x
            hf = h
            wf = w
            cv2.rectangle(img, (int(x),int(y+15)), (int(x+60),int(y+55)), (0,255,0))
            reye = rightface[int(y+h*0.3):int(y+h*0.5+30),int(x):int(x+w*0.2+50)]
            #0.5 0.9 0.2 0.9
    
    gradix = np.zeros_like(reye, dtype=float)
    gradx(reye,gradix,mean,std)
    
    gradiy = np.zeros_like(np.transpose(reye), dtype=float)
    gradx(np.transpose(reye),gradiy,mean,std)
    gradiy = np.transpose(gradiy)
    
    
    func = np.zeros_like(reye, dtype=float)
    means = np.zeros_like(reye,dtype=float)
    y = int(reye.shape[0]/2)
    x = int(reye.shape[1]/2)
    loop = 0
    if reye.shape[0] > 0 and reye.shape[1] > 0:
        while True:
#            if y-1 >= 0 and y+2 <= reye.shape[1] and x-1 >= 0 and x+2 <= reye.shape[0]: 
            if y-AscentArea >= 0:
                ymin = y-AscentArea
            else:
                ymin = 0
                
            if y+AscentArea <= reye.shape[0]:
                ymax = y+AscentArea
            else:
                ymax = reye.shape[0]
                
            if x-AscentArea >= 0:
                xmin = x-AscentArea
            else:
                xmin = 0
            
            if x+AscentArea <= reye.shape[1]:
                xmax = x+AscentArea
            else:
                xmax = reye.shape[1]
                
            contin = 0
            for i in np.arange(ymin, ymax):
                for j in np.arange(xmin, xmax):
                    if means[i,j] < 10:
                        means[i,j] = (255-reye[i,j]) * evaluate(j,i,gradix,gradiy,func)
                        if means[i,j] > means[y,x]:
                            contin = 1
                            y = i
                            x = j
            loop = loop + 1
            if contin == 0 or loop == 10:
                break
    
    cv2.circle(reye,(x,y), 1, (255,255,255))
    cx[3] = int(x+xf+0.2*wf)
    cy[3] = int(y+yf+0.5*hf)
    cv2.circle(img,(int(x+xf),int(y+yf+0.3*hf)), 2, (0,0,255))
    scaled = cv2.resize(reye, (reye.shape[1]*6,reye.shape[0]*6), interpolation = cv2.INTER_CUBIC)
    cv2.imshow('Cam',img)
    cv2.imshow('Eye',255-scaled)
    
    k = cv2.waitKey(20) & 0xff
    
    if k == 27:
        break
    elif k == ord('7'):
        print(calibration)
#    elif k == ord('8'):
#    elif k == ord('9'):
#    elif k == ord('4'):
#    elif k == ord('5'):
#    elif k == ord('6'):
#    elif k == ord('1'):
#    elif k == ord('2'):
#    elif k == ord('3'):
    

cap.release()
cv2.destroyAllWindows()
