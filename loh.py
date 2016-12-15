import cv2
import numpy as np
import os
import imutils
import copy

image = cv2.imread("Maki/ripe-unripe/ru-03.jpg")
clone_img = copy.copy(image)            
b, g, r = cv2.split(image);
bl, gr, re = cv2.split(clone_img);
x = cv2.subtract(r,g)
x2 = cv2.subtract(g,r)                                        
_, output = cv2.threshold(x, 45, 255, cv2.THRESH_BINARY)
_, output2 = cv2.threshold(x2, 45, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)
output = cv2.erode(output, kernel, iterations = 7)
output = cv2.dilate(output, kernel, iterations = 3)

output2 = cv2.erode(output2, kernel, iterations = 2)
output2 = cv2.dilate(output2, kernel, iterations = 2)
i, j = output.shape

for a in range(i):
	for b in range(j):
		if(output[a][b] == 0):
			image[a][b][0] = 0
			image[a][b][1] = 0
			image[a][b][2] = 0
			
		if(output2[a][b] == 0):
			clone_img[a][b][0] = 0
			clone_img[a][b][1] = 0
			clone_img[a][b][2] = 0
			
	
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clone_img = cv2.cvtColor(clone_img, cv2.COLOR_BGR2GRAY)

cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

for c in cnts:
	# compute the center of the contour
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
 
	# draw the contour and center of the shape on the image
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
	cv2.rectangle(image, (cX-30, cY+30), (cX+30, cY-30), (255, 255, 255), 3)
	cv2.putText(image, "center", (cX - 20, cY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		
cnts2 = cv2.findContours(clone_img.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts2 = cnts2[0] if imutils.is_cv2() else cnts2[1]

for c in cnts2:
	# compute the center of the contour
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
 
	# draw the contour and center of the shape on the image
	cv2.drawContours(clone_img, [c], -1, (0, 255, 0), 2)
	cv2.circle(clone_img, (cX, cY), 7, (255, 255, 255), -1)
	cv2.rectangle(clone_img, (cX-30, cY+30), (cX+30, cY-30), (255, 255, 255), 3)
	cv2.putText(clone_img, "center", (cX - 20, cY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		
# show the image
cv2.imshow("Image", image)
cv2.imshow("Clone", clone_img)
cv2.waitKey(0)
'''
circles = cv2.HoughCircles(output, cv2.HOUGH_GRADIENT, 1.2, 100)
 
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
 
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
	# show the output image
	cv2.imshow("output", np.hstack([image, output]))
	cv2.waitKey(0)
	'''
'''	
cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("output", output)

cv2.namedWindow("original", cv2.WINDOW_AUTOSIZE)
cv2.imshow("original", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

output = image.copy()
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect circles in the image
ORANGE_MIN = np.array([5, 50, 50],np.uint8)
ORANGE_MAX = np.array([15, 255, 255],np.uint8)

hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

frame_threshed = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)


circles = cv2.HoughCircles(frame_threshed, cv2.HOUGH_GRADIENT, 1.2, 100)
 
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
 
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
	# show the output image
	cv2.imshow("output", np.hstack([image, output]))
	cv2.waitKey(0)

cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("output", frame_threshed)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''