# Pupil_Tracker
A gaze/pupil tracker based on an algorithm published by Timm Fabian and Erhardt Barth. The paper can be found at https://www.inb.uni-luebeck.de/fileadmin/files/PUBPDFS/TiBa11b.pdf . It basically uses the color gradient field produced by the usually much darker pupil to find the center.

# Speed/Efficiency
My implementation uses haarcascades to find the face and the right eye and then applies the algorithm to the extracted picture of the eye. Numba is being used for automatic parallel processing. To further increase speed gradient descent is used to find the optimum without having to calculate the value of every pixel. The time for each frame depends heavily on the dimensions of the extracted picture of the eye. At a distance of about 40 centimeters with a HD-webcam each frame needs about 0.02 seconds on an intel-i5 with 3.2 GH and 4 real cores.
