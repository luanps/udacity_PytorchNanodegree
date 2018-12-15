import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import sys
image = cv2.imread(sys.argv[1])
# Convert to grayscale for filtering
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')

# 3x3 array for edge detection
sobel_y = np.array([[ -1, -2, -1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]])

sobel_x = np.array([[ -1, 0, 1],
                   [ -2, 0, 2],
                   [ -1, 0, 1]])

newFilter = np.array([[ -2, -1, 0, 1, 2],
                    [ -2, -1, 0, 1, 2],
                    [ -3, -2, 0, 2, 3],
                    [ -2, -1, 0, 1, 2],
                    [ -2, -1, 0, 1, 2]])

filtered = cv2.filter2D(gray, -1, newFilter)
filtered_x = cv2.filter2D(gray, -1, sobel_x)
filtered_y = cv2.filter2D(gray, -1, sobel_y)

cv2.imshow('new 5x5 filter',filtered)
cv2.imshow( 'sobel_x',filtered_x)
cv2.imshow( 'sobel_y',filtered_y)
cv2.waitKey(0)
