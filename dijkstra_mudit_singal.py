import cv2
import numpy as np
import matplotlib.pyplot as plt

open_list = []
closed_list = []

# x = input("Please enter start x position")
# y = input("Please enter start y position")
# x_i = [x,y]
# x = input("Please enter goal x position")
# y = input("Please enter goal y position")
# x_g = [y,x]
x_i = [15,15]
x_g = [230, 580]
visualize_start_n_goal = True
n_iter = 1000

# Variable that will keep count of iterations
iters = 0 


class node:
	def __init__(self, data):
		self.data = data
		self.children = []
		self.parent = None
		self.ctc = np.inf

	def append_child(self, child):
		child.parent = self
		self.children.append(child)
		child.ctc = self.ctc + 1

	def update_parent(parent):
		self.parent = parent
		self.ctc = parent.ctc + 1


def move_up(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	
	if img[y+1, x, 1] == 255:
		return [x, y+1], 1.0, True
	else:
		return [], np.inf, False


def move_down(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if img[y-1, x, 1] == 255:
		return [x, y-1], 1.0, True
	else:
		return [], np.inf, False


def move_left(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if img[y, x-1, 1] == 255:
		return [x-1, y], 1.0, True
	else:
		return [], np.inf, False


def move_right(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if img[y, x+1, 1] == 255:
		return [x+1, y], 1.0, True
	else:
		return [], np.inf, False


def move_up_left(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if img[y+1, x-1, 1] == 255:
		return [x-1, y+1], 1.4, True
	else:
		return [], np.inf, False

def move_up_right(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if img[y+1, x+1, 1] == 255:
		return [x+1, y+1], 1.4, True
	else:
		return [], np.inf, False


def move_down_left(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if img[y-1, x-1, 1] == 255:
		return [x-1, y-1], 1.4, True
	else:
		return [], np.inf, False


def move_down_right(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if img[y-1, x+1, 1] == 255:
		return [x+1, y-1], 1.4, True
	else:
		return [], np.inf, False


moves_list = [move_up, move_down, move_left, move_right, move_up_left, move_up_right, move_down_left, move_down_right]
node_arr = np.empty(shape=[250, 600, 2])
node_arr[:, :, 0] = None
node_arr[:, :, 1] = np.inf

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

if visualize_start_n_goal:
	start_color = [0, 0, 255]
	goal_color = [0, 255, 0]
	cv2.circle(img, (x_i[0], x_i[1]) , color=start_color, radius=3, thickness=-1)
	cv2.circle(img, (x_g[0], x_g[1]) , color=goal_color, radius=3, thickness=-1)
	img = cv2.flip(img, flipCode=0)
	print(img.shape)

# curr = start_node
# while curr != goal and iters < n_iter:
# 	for move in moves_list:
# 		[x_new, y_new], success = move(curr)
# 		if success:
# 			if [x_new, y_new] not in open_list:
# 				temp_node = node([x_new, y_new])

# 			if curr.ctc + get_relative_ctc(curr, temp_node) < temp_node.ctc:
# 				temp_node.ctc = curr.ctc + get_relative_ctc(curr, temp_node)
# 				curr.append_child(temp_node)



cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()