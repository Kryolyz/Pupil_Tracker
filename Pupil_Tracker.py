# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:02:17 2019

"""

import cv2
import numpy as np
import time

class PupilDetector:
    """
    @brief      A class for detecting the pupil in a video stream.
    
    This class detects the pupil in a video stream by using OpenCV's Haar cascades
    to detect the eyes and then applying a gradient ascent algorithm to find the center of the pupil.
    """
    def __init__(self):
        """
        @brief      Constructs a new instance of the PupilDetector class.
        
        This constructor initializes the Haar cascade classifier for eye detection and opens a video stream from the default camera. It also sets some parameters for the gradient ascent algorithm.
        """
        # Initialize Haar cascade classifier for eye detection
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
        # Open video stream from default camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3,1920)
        self.cap.set(4,1280)
        # Set parameters for gradient ascent algorithm
        self.AscentArea = 2
        self.maxGradientAscentSteps = 15

    def get_gradients(self, im):
        """
        @brief      Calculates the gradients of an image.
        
        @param      im    The image.
        
        This function calculates the gradients of an image using numpy's gradient function.
        It then normalizes the gradients and sets small values to zero.
        """
        grad = np.zeros_like(im, dtype=float)
        if grad.size > 0:
            # Calculate gradients
            gradients = np.gradient(im)
            for grad in gradients:
                # Normalize gradients
                mean = np.mean(grad)
                std = np.std(grad)
                grad[(grad < 0.3 * mean + std) & (grad > -0.3 * mean - std)] = 0
                grad[grad > 0] = 1
                grad[grad < 0] = -1
        self.gradient_x = gradients[1]
        self.gradient_y = gradients[0]
    
    def evaluate(self, x, y, reward_function):
        """
        @brief      Evaluates the reward function at a given position.
        
        @param      x               The x coordinate of the position.
        @param      y               The y coordinate of the position.
        @param      reward_function  The reward function.
        
        @return     The mean value of the reward function.
        
        This function evaluates the reward function at a given position by calculating the dot product
        between the gradient field and the direction from the current position to all other positions.
        It then sets negative values to zero and returns the mean value of the reward function.
        """
        if self.gradient_x.size > 0:
            # Calculate direction from current position to all other positions
            cy, cx = np.indices(self.gradient_x.shape)
            dy = (cy - y).astype(float)
            dx = (cx - x).astype(float)
            norm = np.linalg.norm(np.stack([dx, dy], axis=-1), axis=-1)
            dy[norm != 0] /= norm[norm != 0]
            dx[norm != 0] /= norm[norm != 0]
            
            # Calculate dot product between gradient field and direction
            reward_function = self.gradient_y * dy + self.gradient_x * dx
            # Set negative values to zero
            reward_function[reward_function < 0] = 0
        return np.mean(reward_function)
    
    def gradient_ascent_step(self, x, y, eye):
        """
        @brief      Performs a single step of gradient ascent.
        
        @param      x     The x coordinate of the current position.
        @param      y     The y coordinate of the current position.
        @param      eye   The eye image.
        
        @return     A tuple containing a boolean indicating whether to continue gradient ascent and the new x and y coordinates.
        
        This function performs a single step of gradient ascent to find the center of the pupil in an eye image.
        It evaluates the reward function in a local area around the current position and moves to the position with the highest reward.
        """
        
        # Determine the local area around the current position
        if y - self.AscentArea >= 0:
            ymin = y - self.AscentArea
        else:
            ymin = 0
            
        if y + self.AscentArea <= eye.shape[0]:
            ymax = y + self.AscentArea
        else:
            ymax = eye.shape[0]
            
        if x - self.AscentArea >= 0:
            xmin = x - self.AscentArea
        else:
            xmin = 0
        
        if x + self.AscentArea <= eye.shape[1]:
            xmax = x+self.AscentArea
        else:
            xmax = eye.shape[1]
        
        # Evaluate the reward function in the local area
        continueGradientAscent = False
        for i in np.arange(ymin, ymax):
            for j in np.arange(xmin, xmax):
                if self.means[i,j] < 10:
                    self.means[i,j] = (255-eye[i,j]) * self.evaluate(j,i,self.reward_function)
                    # Move to the position with the highest reward
                    if self.means[i,j] > self.means[y,x]:
                        continueGradientAscent = True
                        y = i
                        x = j
        return continueGradientAscent, x, y
        
    
    def detect_pupil(self):
        """
        @brief      Detects the pupil in a video stream.
        
        This function detects the pupil in a video stream by using OpenCV's Haar cascades to detect
        the eyes and then applying a gradient ascent algorithm to find the center of the pupil.
        """
        while True:
            # Read a frame from the video stream
            re, img = self.cap.read()
            if not re:
                continue
            
            # Time the entire thing
            t = time.time()
            
            # Convert the image to grayscale and apply Gaussian blur
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(13,13),4)
            
            # Detect eyes using Haar cascades
            eyes = self.eye_cascade.detectMultiScale(gray)
            eye_position = []
            if type(eyes) is not tuple:
                if (len(eyes) < 2):
                    x,y,w,h = eyes[0]
                else:
                    # Choose the leftmost eye in the video (which most likely is your right eye)
                    if eyes[0][0] < eyes[1][0]:
                        x,y,w,h = eyes[0]
                    else:
                        x,y,w,h = eyes[1]
                        
                # Store eye position and draw a rectangle around it
                eye_position = [x,y,w,h]
                eye_position[1] += 0.3*h
                cv2.rectangle(img, (int(x),int(y+h*0.3)), (int(x+w),int(y+h)), (0,255,0))
                eye = gray[int(y+h*0.3):int(y+h),int(x):int(x+w)]
            else:
                continue
            
            # Calculate gradients
            self.get_gradients(eye)
            
            # Initialize variables for gradient ascent
            y = int(eye.shape[0]/2)
            x = int(eye.shape[1]/2)
            loop = 0
                
            self.reward_function = np.zeros_like(eye, dtype=float)
            self.means = np.zeros_like(eye,dtype=float)
            
            if eye.shape[0] > 0 and eye.shape[1] > 0:
                # Perform gradient ascent to find the center of the pupil
                while True:
                    continueGradientAscent, x, y = self.gradient_ascent_step(x, y, eye)
                    loop = loop + 1
                    if not continueGradientAscent or loop == self.maxGradientAscentSteps:
                        break
            
            # Print time
            t = time.time() - t
            print(t)
            
            # Draw a circle around the detected pupil center
            cv2.circle(eye,(x,y), 1, (255,255,255))
            cv2.circle(img,(int(eye_position[0]+x),int(eye_position[1]+y)), 2, (0,0,255))
            scaled = cv2.resize(eye, (eye.shape[1]*6,eye.shape[0]*6), interpolation = cv2.INTER_CUBIC)
            cv2.imshow('Cam',img)
            cv2.imshow('Eye',255-scaled)
            
            
            k = cv2.waitKey(1) & 0xff
            
            if k == 27:
                self.cap.release()
                cv2.destroyAllWindows()
                break
            
if __name__ == '__main__':
    pupil_detector = PupilDetector()
    pupil_detector.detect_pupil()
