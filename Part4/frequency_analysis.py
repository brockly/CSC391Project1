'''
Alex Brockman
Part 4 Frequency Analysis
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

#First make it grayscale RGB TO GRAY
path = '/Users/Alex/Dropbox/Documents_WakeForest/Junior/CSC 391/Projects/Project #1 - Spatial and Frequency Filtering/VisionLab1/Part4/'
img = cv2.imread(path + 'dog_noise50.JPG')
small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('Gray Image', small)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create noisy image as g = f + sigma * noise
# with noise scaled by sigma = .2 max(f)/max(noise)
noise = np.random.randn(small.shape[0], small.shape[1])
smallNoisy = np.zeros(small.shape, np.float64)
sigma = 0.2 * small.max()/noise.max()
# Color images need noise added to all channels
if len(small.shape) == 2:
    smallNoisy = small + sigma * noise
else:
    smallNoisy[:, :, 0] = small[:, :, 0] + sigma * noise
    smallNoisy[:, :, 1] = small[:, :, 1] + sigma * noise
    smallNoisy[:, :, 2] = small[:, :, 2] + sigma * noise

# Calculate the index of the middle column in the image
col = int(smallNoisy.shape[1]/2)
# Obtain the image data for this column
colData = small[0:smallNoisy.shape[0], col, 0]

# Plot the column data as a function
xvalues = np.linspace(0, len(colData) - 1, len(colData))
plt.plot(xvalues, colData, 'b')
plt.savefig(path + 'columndata.png', bbox_inches='tight')
plt.show()

# Compute the 1-D Fourier transform of colData
F_colData = np.fft.fft(colData)

# Plot the magnitude of the Fourier coefficients as a stem plot
# Notice the use off fftshift() to center the low frequency coefficients around 0
xvalues = np.linspace(0, len(colData), len(colData))
markerline, stemlines, baseline = plt.stem(xvalues, (np.abs(F_colData)), 'g')
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','r', 'linewidth', 0.5)
plt.savefig(path + 'stemplot.png', bbox_inches='tight')
plt.show()

# Convert small image to grayscale so that a 3D plot is easier to make
graySmall = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:graySmall.shape[0], 0:graySmall.shape[1]]
# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, graySmall, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
plt.savefig(path + '3dgraph.png', bbox_inches='tight')
plt.show()

# Take the 2-D DFT and plot the magnitude of the corresponding Fourier coefficients
F2_graySmall = np.fft.fft2(graySmall)
fig = plt.figure()
ax = fig.gca(projection='3d')
Y = (np.linspace(-int(graySmall.shape[0]/2), int(graySmall.shape[0]/2)-1, graySmall.shape[0]))
X = (np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1]))
X, Y = np.meshgrid(X, Y)

for k in range(0, 150):
    # Truncate frequencies and then plot the resulting function in real space
    Trun_F_colData = F_colData.copy()
    Trun_F_colData[k+1:len(Trun_F_colData)-k] = 0
    trun_colData = np.fft.ifft(Trun_F_colData)
    # Plot
    xvalues = np.linspace(0, len(trun_colData) - 1, len(trun_colData))
    plt.plot(xvalues, colData, 'b')
    plt.plot(xvalues, trun_colData, 'r')
    plt.title('k = 0 : ' + str(k))
    plt.savefig(path + 'plot.png', bbox_inches='tight')
    k = k + 25
    plt.show()
    plt.clf()

# Plot the magnitude and the log(magnitude + 1) as images (view from the top)
magnitudeImage = np.fft.fftshift(np.abs(F2_graySmall))
magnitudeImage = magnitudeImage / magnitudeImage.max()   # scale to [0, 1]
magnitudeImage = ski.img_as_ubyte(magnitudeImage)
cv2.imshow('Magnitude plot', magnitudeImage)
cv2.imwrite(path + 'magImg.jpg', magnitudeImage, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
logMagnitudeImage = np.fft.fftshift(np.log(np.abs(F2_graySmall)+1))
logMagnitudeImage = logMagnitudeImage / logMagnitudeImage.max()   # scale to [0, 1]
logMagnitudeImage = ski.img_as_ubyte(logMagnitudeImage)
cv2.imshow('Log Magnitude plot', logMagnitudeImage)
cv2.imwrite(path + 'logMagImg.jpg', logMagnitudeImage, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
cv2.waitKey(0)

