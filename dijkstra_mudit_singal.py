import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
from queue import PriorityQueue
import time

start = time.time()

open_list = []
path_list = []
# open_list = PriorityQueue()
# closed_list = []

# x = input("Please enter start x position")
# y = input("Please enter start y position")
# x_i = [x,y]
# x = input("Please enter goal x position")
# y = input("Please enter goal y position")
# x_g = [y,x]

visualize_start_n_goal = True
n_iter = 10000

# Variable that will keep count of iterations
iters = 0 

x_min = 0
x_max = 600

y_min = 0
y_max = 250

class node:
	def __init__(self, data):
		self.data = data
		self.children = []
		self.parent = None
		self.ctc = np.inf

	def append_child(self, child):
		child.parent = self
		self.children.append(child)


	def update_parent(self, parent, next_ctc):
		self.parent = parent
		self.ctc = parent.ctc + next_ctc


def move_up(node):
	curr_px = node.data # (y,x)
	y = curr_px[0]
	x = curr_px[1]

	if y > y_max:
		return [], np.inf, False
	
	if img[y+1, x, 1] == 255:
		return np.array([y+1, x]), 1.0, True
	else:
		return [], np.inf, False


def move_down(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if y < y_min:
		return [], np.inf, False

	if img[y-1, x, 1] == 255:
		return np.array([y-1, x]), 1.0, True
	else:
		return [], np.inf, False


def move_left(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if x < x_min:
		return [], np.inf, False

	if img[y, x-1, 1] == 255:
		return np.array([y, x-1]), 1.0, True
	else:
		return [], np.inf, False


def move_right(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if x > x_max:
		return [], np.inf, False

	if img[y, x+1, 1] == 255:
		return np.array([y, x+1]), 1.0, True
	else:
		return [], np.inf, False


def move_up_left(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if y > y_max and x < x_min:
		return [], np.inf, False

	if img[y+1, x-1, 1] == 255:
		return np.array([y+1, x-1]), 1.4, True
	else:
		return [], np.inf, False

def move_up_right(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if y > y_max and x > x_max:
		return [], np.inf, False

	if img[y+1, x+1, 1] == 255:
		return np.array([y+1, x+1]), 1.4, True
	else:
		return [], np.inf, False


def move_down_left(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if y < y_min or x < x_min:
		return [], np.inf, False

	if img[y-1, x-1, 1] == 255:
		return np.array([y-1, x-1]), 1.4, True
	else:
		return [], np.inf, False


def move_down_right(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if y < y_min and x > x_max:
		return [], np.inf, False

	if img[y-1, x+1, 1] == 255:
		return np.array([y-1, x+1]), 1.4, True
	else:
		return [], np.inf, False

x_i = np.array([15,16])
x_g = np.array([35, 582])
# x_g = np.array([235, 580])
moves_list = [move_up, move_down, move_left, move_right, move_up_left, move_up_right, move_down_left, move_down_right]
node_arr = np.empty(shape=[250, 600, 4], dtype=object)
node_arr[:, :, 0] = None		# Node 
node_arr[:, :, 1] = np.inf		# Cost to come of current pixel
node_arr[:, :, 2] = False		# In closed list
node_arr[:, :, 3] = False		# In open list

start_node = node(x_i)
start_node.ctc = 0
goal_node = node(x_g)

# Initializing the start node cost to some as 0
node_arr[x_i[0], x_i[1], 1] = 0
node_arr[x_i[0], x_i[1], 0] = start_node

img = np.ones([250, 600, 3])*255

bloat_grid = np.ones([11,11,3])*255
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

img[:, 0:6] = cv2.bitwise_and(img[:, 0:6], np.array([255, 0, 0]) )
img[:, -5:] = cv2.bitwise_and(img[:, -5:], np.array([255, 0, 0]) )
img[0:6, :] = cv2.bitwise_and(img[0:6, :], np.array([255, 0, 0]) )
img[-5:, :] = cv2.bitwise_and(img[-5:, :], np.array([255, 0, 0]) )


if visualize_start_n_goal:
	start_color = [0, 0, 255]
	goal_color = [0, 255, 0]
	cv2.circle(img, (x_i[1], x_i[0]) , color=start_color, radius=3, thickness=-1)
	cv2.circle(img, (x_g[1], x_g[0]) , color=goal_color, radius=3, thickness=-1)
	img = cv2.flip(img, flipCode=0)
	print(img.shape)

curr_node = start_node
# heapq.heappush(open_list, [0, start_node.data])
# open_list.put([0, start_node])
open_list.append([0, start_node])

while np.count_nonzero(curr_node.data - goal_node.data) != 0 and len(open_list) != 0 :
	# [curr_ctc, curr_node_data] = heapq.heappop(open_list)
	[curr_ctc, curr_node] = open_list.pop(0)
	# heapq.heappush(closed_list, [curr_ctc, curr_node_data])
	node_arr[curr_node.data[0], curr_node.data[1], 2] = True

	iters += 1


	for move in moves_list:
		# [x_new, y_new], incr_ctc, success = move(curr)
		new_coords, incr_ctc, success = move(curr_node)
		# cv2.circle(img, (new_coords[1]+3, new_coords[0]+3) , color=start_color, radius=1, thickness=-1)

		if success:
			new_node_params = node_arr[new_coords[0], new_coords[1]]
			if new_node_params[2] == False:
				if new_node_params[3] == False or new_node_params[1] == np.inf:
					temp_node = node(new_coords)
					temp_node.ctc = curr_node.ctc + incr_ctc
					node_arr[new_coords[0], new_coords[1], 1] = curr_node.ctc + incr_ctc

					curr_node.append_child(temp_node)
					node_arr[new_coords[0], new_coords[1], 0] = temp_node

					# Adding the new node in open list and marking it as open in the array of nodes
					# open_list.put([1, temp_node])
					open_list.append([curr_node.ctc + incr_ctc, temp_node])
					open_list.sort(key=lambda x:x[0])
					# heapq.heappush(open_list, [curr_node.ctc + incr_ctc, new_coords])
					node_arr[new_coords[0], new_coords[1], 3] = True


				elif node_arr[new_coords[0], new_coords[1], 1] > curr_node.ctc + incr_ctc:
					node_arr[new_coords[0], new_coords[1], 0].parent = curr_node
					node_arr[new_coords[0], new_coords[1], 0].ctc = curr_node.ctc + incr_ctc
					node_arr[new_coords[0], new_coords[1], 1] = curr_node.ctc + incr_ctc



print(curr_node.data)
back_parent = curr_node
while back_parent != None:
	path_list.append(back_parent.data)
	back_parent = back_parent.parent

path_list = path_list[::-1]	

for i in range(len(path_list)):
	# print(path_list[i, 0])
	img[y_max - path_list[i][0], path_list[i][1]] = np.array([0, 0, 0])


print(iters)

end = time.time()

print("Total time taken: ", end - start)


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()