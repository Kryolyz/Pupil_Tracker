# Pupil_Tracker
A gaze/pupil tracker based on an algorithm published by Timm Fabian and Erhardt Barth. The paper can be found at https://www.inb.uni-luebeck.de/fileadmin/files/PUBPDFS/TiBa11b.pdf . It basically uses the color gradient field produced by the usually much darker pupil to find the center.

# Speed/Efficiency
My implementation uses haarcascades to find the eyes and then applies the algorithm to the extracted picture of the eye. To increase speed Gradient Ascent is used to find the optimum without having to calculate the reward function value of every pixel. The time for each frame depends somewhat on the dimensions of the extracted eye image, but the impact is reduced by using numpy vectorization. At a distance of about 40 centimeters with an HD-webcam each frame needs about 0.015 seconds on my machine, which is not particularly strong.

# Goal
My original goal was to create a gaze tracker accurately enough to control the computer cursor with. Unfortunately the resolution of the webcam seems to be too low. Also a fixed reference point on the face is missing to calculate the relative position of the pupil to the eye/face, but that problem should be solvable by using either a fixed matrix to track a certain point or a small easily distinguishable object on the face (like a small frame with a clear color).
