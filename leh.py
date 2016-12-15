import numpy as np
import cv2
import os
import imutils
import copy

image = cv2.imread("Maki/ripe-unripe/11228.jpg")
clone_img = copy.copy(image)
clone_img2 = copy.copy(image)
clone_img3 = copy.copy(image)
b, g, r = cv2.split(image);
bl, gr, re = cv2.split(clone_img);

x = cv2.subtract(r,g)
x2 = cv2.subtract(r,g)
                                        
_, output = cv2.threshold(x, 45, 255, cv2.THRESH_BINARY)
_, output2 = cv2.threshold(x2, 45, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)
output = cv2.erode(output, kernel, iterations = 3)
output = cv2.dilate(output, kernel, iterations = 2)

output2 = cv2.erode(output2, kernel, iterations = 3)
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

k, l = image.shape
for c in range(k):
	for d in range(l):
		if(image[c][d] == 0 and clone_img[c][d] == 0):
			clone_img2[c][d][0] = 0
			clone_img2[c][d][1] = 0
			clone_img2[c][d][2] = 0
			
clone_img2 = cv2.cvtColor(clone_img2, cv2.COLOR_BGR2GRAY)
cnts = cv2.findContours(clone_img2.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
idx = 0
for c in cnts:
	
	idx += 1
	x,y,w,h = cv2.boundingRect(c)
	roi=clone_img[y:y+h,x:x+w]
	#cv2.imwrite(str(idx) + '.jpg', roi)
	
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

	cv2.drawContours(clone_img2, [c], -1, (0, 255, 0), 2)
	
	#cv2.circle(clone_img3, (cX, cY), 7, (255, 255, 255), -1)
	cv2.rectangle(clone_img3, (cX-30, cY+30), (cX+30, cY-30), (255, 255, 255), 3)
	#cv2.putText(clone_img3, "center", (cX - 20, cY - 20),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		
cv2.imshow("Clone", clone_img3)
cv2.waitKey(0)
