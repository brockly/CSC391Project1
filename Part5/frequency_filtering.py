'''
Alex Brockman
Part 5 Frequency Filtering
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

#import image and make it gray
path = '/Users/Alex/Dropbox/Documents_WakeForest/Junior/CSC 391/Projects/Project #1 - Spatial and Frequency Filtering/VisionLab1/Part5/'
img = cv2.imread(path + 'dog.JPG')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('Gray Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

Y = (np.linspace(-int(gray_image.shape[0]/2), int(gray_image.shape[0]/2)-1, gray_image.shape[0]))
X = (np.linspace(-int(gray_image.shape[1]/2), int(gray_image.shape[1]/2)-1, gray_image.shape[1]))
X, Y = np.meshgrid(X, Y)

# Explore the Butterworth filter
# U and V are arrays that give all integer coordinates in the 2-D plane
#  [-m/2 , m/2] x [-n/2 , n/2].
# Use U and V to create 3-D functions over (U,V)
U = (np.linspace(-int(gray_image.shape[0]/2), int(gray_image.shape[0]/2)-1, gray_image.shape[0]))
V = (np.linspace(-int(gray_image.shape[1]/2), int(gray_image.shape[1]/2)-1, gray_image.shape[1]))
U, V = np.meshgrid(U, V)
# The function over (U,V) is distance between each point (u,v) to (0,0)
D = np.sqrt(X*X + Y*Y)
# create x-points for plotting
xval = np.linspace(-int(gray_image.shape[1]/2), int(gray_image.shape[1]/2)-1, gray_image.shape[1])
# Specify a frequency cutoff value as a function of D.max()
D0 = 0.25 * D.max()

# The ideal lowpass filter makes all D(u,v) where D(u,v) <= 0 equal to 1
# and all D(u,v) where D(u,v) > 0 equal to 0
idealLowPass = D <= D0

# Filter our small grayscale image with the ideal lowpass filter
# 1. DFT of image
print(gray_image.dtype)
FTgray_image = np.fft.fft2(gray_image.astype(float))
# 2. Butterworth filter is already defined in Fourier space
# 3. Elementwise product in Fourier space (notice fftshift of the filter)
FTgray_imageFiltered = FTgray_image * np.fft.fftshift(idealLowPass)
# 4. Inverse DFT to take filtered image back to the spatial domain
gray_imageFiltered = np.abs(np.fft.ifft2(FTgray_imageFiltered))

# Save the filter and the filtered image (after scaling)
idealLowPass = ski.img_as_ubyte(idealLowPass / idealLowPass.max())
gray_imageFiltered = ski.img_as_ubyte(gray_imageFiltered / gray_imageFiltered.max())
cv2.imwrite(path + "idealLowPass.jpg", idealLowPass)
cv2.imwrite(path + "grayImageIdealLowpassFiltered.jpg", gray_imageFiltered)

# Plot the ideal filter and then create and plot Butterworth filters of order
# n = 1, 2, 3, 4
plt.plot(xval, idealLowPass[int(idealLowPass.shape[0]/2), :], 'c--', label='ideal')
colors='brgkmc'
for n in range(1, 5):
    # Create Butterworth filter of order n
    H = 1.0 / (1 + (np.sqrt(2) - 1)*np.power(D/D0, 2*n))
    # Apply the filter to the grayscaled image
    FTgray_imageFiltered = FTgray_image * np.fft.fftshift(H)
    gray_imageFiltered = np.abs(np.fft.ifft2(FTgray_imageFiltered))
    gray_imageFiltered = ski.img_as_ubyte(gray_imageFiltered / gray_imageFiltered.max())
    cv2.imwrite(path + "grayImageButterworth-n" + str(n) + ".jpg", gray_imageFiltered)
    H = ski.img_as_ubyte(H / H.max())
    cv2.imwrite(path + "butter-n" + str(n) + ".jpg", H)
    # Get a slice through the center of the filter to plot in 2-D
    slice = H[int(H.shape[0]/2), :]
    plt.plot(xval, slice, colors[n-1], label='n='+str(n))
    plt.legend(loc='upper left')

plt.savefig(path + 'butterworthFilters.jpg', bbox_inches='tight')