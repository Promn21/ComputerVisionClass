import cv2 as cv

image1 = cv.imread("mewtwo.png")
image2 = cv.imread("mew.png")

#greyscale
gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
gray_image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
#threshold
_, binary_image1 = cv.threshold(gray_image1, 150, 255, cv.THRESH_BINARY)
_, binary_image2 = cv.threshold(gray_image2, 150, 255, cv.THRESH_BINARY)

#draw a black circle
cv.circle(gray_image1, (gray_image1.shape[1] // 2, gray_image1.shape[0] // 2), 200, (1, 1, 1), 2)

#canvas
height, width = gray_image1.shape
top_row = cv.hconcat([gray_image1, binary_image1]) 
bottom_row = cv.hconcat([gray_image2, binary_image2]) 
canvas = cv.vconcat([top_row, bottom_row]) 

#images placing
#mewtwo
canvas[0:height, 0:width] = gray_image1  # Top1
canvas[0:height, width:width * 2] = binary_image1  # Top2
#mew
canvas[height:height * 2, 0:width] = gray_image2  # Bottom1
canvas[height:height * 2, width:width * 2] = binary_image2  # Bottom2

cv.imshow("Mewone + Mewtwo = Mewthree", canvas)

cv.waitKey(0)