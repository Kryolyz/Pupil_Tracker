# Pupil_Tracker
A gaze/pupil tracker based on an algorithm published by Timm Fabian and Erhardt Barth. The paper can be found at https://www.inb.uni-luebeck.de/fileadmin/files/PUBPDFS/TiBa11b.pdf . It basically uses the color gradient field produced by the usually much darker pupil to find the center. Don't mind the "CNN" folder, it's just files from testing neural networks to accomplish a similar thing.

# Speed/Efficiency
My implementation uses haarcascades to find the face and the right eye and then applies the algorithm to the extracted picture of the eye. Numba is being used for automatic parallel processing. To further increase speed gradient descent is used to find the optimum without having to calculate the value of every pixel. The time for each frame depends heavily on the dimensions of the extracted picture of the eye. At a distance of about 40 centimeters with an HD-webcam each frame needs about 0.02 seconds on an intel-i5 with 3.2 GHz and 4 real cores.

# Goal
My original goal was to create a gaze tracker accurately enough to control the computer cursor with. Unfortunately the resolution of the webcam seems to be too low. Also a fixed reference point on the face is missing to calculate the relative position of the pupil to the eye/face, but that problem should be solvable by using either a fixed matrix to track a certain point or a small easily distinguishable object on the face (like a small frame with a clear color).

I also attempted to use a convulutional neural network to determine the gaze-direction but in order to find the optimal shape of such a network a meta-learning algorithm is required and my gpu needs literal hours to even train a single small convulutional one, so the only option is to rent clouds and i'd prefer to not spend too much money on hobby projects.
