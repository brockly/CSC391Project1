'''
Alex Brockman
Part 3 Implementation
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image of Ima and resize by 1/4
path = '/Users/Alex/Dropbox/Documents_WakeForest/Junior/CSC 391/Projects/Project #1 - Spatial and Frequency Filtering/VisionLab1/Part3/'
img = cv2.imread(path + 'dog_noise50.JPG')
small = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
windowLen = 3 #default is 3
try:
    windowLen = int(input("Enter a box filter size \n"))
except ValueError:
    print("Not a number")
w = windowLen

# Create noisy image as g = f + sigma * noise
# with noise scaled by sigma = .2 max(f)/max(noise)
noise = np.random.randn(small.shape[0], small.shape[1])
smallNoisy = np.zeros(small.shape, np.float64)
sigma = 0.2 * small.max()/noise.max()
# Color images need noise added to all channels (1, 2 and 3)
if len(small.shape) == 2:
    smallNoisy = small + sigma * noise
else:
    smallNoisy[:, :, 0] = small[:, :, 0] + sigma * noise
    smallNoisy[:, :, 1] = small[:, :, 1] + sigma * noise
    smallNoisy[:, :, 2] = small[:, :, 2] + sigma * noise

# Calculate the index of the middle column in the image
col = int(smallNoisy.shape[1]/2)
# Obtain the image data for this column
colData = smallNoisy[0:smallNoisy.shape[0], col, 0]

# Plot the column data as a stem plot of xvalues vs colData
xvalues = np.linspace(0, len(colData) - 1, len(colData))
markerline, stemlines, baseline = plt.stem(xvalues, colData, 'b')
plt.setp(markerline, 'markerfacecolor', 'b')
plt.setp(baseline, 'color', 'r', 'linewidth', 0.5)
plt.title('Pre Filter')
plt.show()

# Create a 1-D filter of length 6: [1/6, 1/6, 1/6] and apply to column data before displaying it
w = np.ones(windowLen, 'd')  # vector of ones
w = w / w.sum()
y = np.convolve(w, colData, mode='valid')

# Plot the filtered column data as a stem plot
xvalues = np.linspace(0, len(y)-1, len(y))
markerline, stemlines, baseline = plt.stem(xvalues, y, 'g')
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','r', 'linewidth', 0.5)
plt.title('Post Filter')
plt.show()

# Create a 2-D box filter of size 6 x 6 and scale so that sum adds up to 1
w = np.ones((windowLen, windowLen), np.float32)
w = w / w.sum()
noised = np.zeros(smallNoisy.shape, np.float64)  # array for Noised image
# Apply the filter to each channel
noised[:, :, 0] = cv2.filter2D(smallNoisy[:, :, 0], -1, w)
noised[:, :, 1] = cv2.filter2D(smallNoisy[:, :, 1], -1, w)
noised[:, :, 2] = cv2.filter2D(smallNoisy[:, :, 2], -1, w)

print(noised.min())
print(noised.max())

#Show original image
cv2.imshow("Dog", small)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Show spatial filter
noised = noised / smallNoisy.max()  # scale to [min/max, 1]
noised = noised * 255 # scale to [min/max*255, 255]
cv2.imshow('Spatial Filter', noised.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

#Save images in folder
cv2.imwrite(path + 'prefilter_dog.jpg', small)
print("Pre Filtered photo saved!")
cv2.imwrite(path + 'spatialfilter_dog.jpg', smallNoisy)
print("Spatial Filtered photo saved!")

#Gaussian Blur
gaussian = cv2.GaussianBlur(small, (windowLen, windowLen), 0)
cv2.imshow('Gaussian Blur', gaussian)
cv2.imwrite(path + 'gaussian_dog.jpg', gaussian)
print("Gaussian Blur photo saved!")
cv2.waitKey(0)
cv2.destroyAllWindows()

#Median Blur
median = cv2.medianBlur(small, windowLen)
cv2.imshow('Median Blur', median)
cv2.imwrite(path + 'median_blur_dog.jpg', median)
print("Median Blur photo saved!")
cv2.waitKey(0)
cv2.destroyAllWindows()

#Edge Detection
edges = cv2.Canny(small, 100, 200)
cv2.imshow('Edge Detection', edges)
cv2.imwrite(path + 'edge_detection_dog.jpg', edges)
print("Edge Detection photo saved!")
cv2.waitKey(0)

