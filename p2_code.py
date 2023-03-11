import cv2
import numpy as np
import matplotlib.pyplot as plt

img = np.ones([250, 600, 3])*255

bloat_grid = np.ones([11,11,3])*225
bloat_grid[:,:,1:3] = 0

# Drawing the rectangular obstacles
img[0:100, 100:150, :] = 0
img[-100:-1, 100:150, :] = 0

# Polygon corner points coordinates
hex_pts = np.array([[300, 200], [300+65, 125+32],
                	[300+65, 125-32], [300, 50],
                	[300-65, 125-32], [300-65, 125+32]], np.int32)
 
 
tri_pts = np.array([[460, 225], [510, 125], [460, 25]])

color = (0, 0, 0)
# img = cv2.polylines(img, np.array([hex_pts]), True, color, 5)
img = cv2.fillPoly(img, np.array([hex_pts]), color)
img = cv2.fillPoly(img, np.array([tri_pts]), color)

for i in range(5,img.shape[0]-5):
	for j in range(5,img.shape[1]-5):
		if np.sum(img[i,j]) == 0:
			# print(i,j)
			# print(img[i-5:i+5, j-5:j+5].shape)
			# print(bloat_grid.shape)
			img[i-5:i+6, j-5:j+6] = cv2.bitwise_and(img[i-5:i+6, j-5:j+6, :], bloat_grid)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()