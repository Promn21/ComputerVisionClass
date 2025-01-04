import cv2
import numpy as np

# Read image
img = cv2.imread('mewtwo.png')
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

### Apply different edge detection methods ###



# Sobel
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobelx, sobely)
sobel_combined = sobel_combined.astype(np.uint8)
_, threshold_img = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY_INV)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

anime = cv2.bitwise_and(img, img, mask=threshold_img)

#Canny
canny = cv2.Canny(blurred, threshold1=100, threshold2=200)

# Laplacian
#laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

#Contors
_, threshold_img2 = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(threshold_img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

#Morphological operations
kernel = np.ones((5,5), np.uint8) # Define kernel for morphological operations
dilation = cv2.dilate(threshold_img2, kernel, iterations=1) # Perform dilation
erosion = cv2.erode(threshold_img2, kernel, iterations=1) # Perform erosion
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) # Perform opening (erosion followed by dilation)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) # Perform closing (dilation followed by erosion)

# Convert to uint8 and normalize for display
#sobel_combined = np.uint8(np.absolute(sobel_combined))
#laplacian = np.uint8(np.absolute(laplacian))

# Display results
# cv2.imshow('Blur', blurred)
# cv2.imshow('Sobel_x', sobelx)
# cv2.imshow('Sobel_y', sobely)
# cv2.imshow('Sobel', sobel_combined.astype(np.uint8))
# cv2.imshow('Canny', canny)
# cv2.imshow('Anime', anime)
# cv2.imshow('Edge', threshold_img2)
# cv2.imshow('Contours', result)
# cv2.imshow('Laplacian', laplacian)
cv2.imshow('Original', img)
cv2.imshow('Dilation', dilation)
cv2.imshow('Erosion', erosion)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()