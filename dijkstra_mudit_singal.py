# Importing all the necessary libraries
import cv2
import numpy as np
import time

# Creating 2 empty lists for storing the list of open nodes and list of nodes in path from start position to goal position
open_list = []
path_list = []

# Reading the start and goal position from the user
# x = int(input("Please enter start x position: "))
# y = int(input("Please enter start y position: "))
# x_i = np.array([y,x])
# x = int(input("Please enter goal x position: "))
# y = int(input("Please enter goal y position: "))
# x_g = np.array([y,x])

# Start recording the time to check time of execution
start = time.time()

# Test Start and goal position
x_i = np.array([15,16])
x_g = np.array([35, 582])
# x_g = np.array([235, 580])

# Flag to enable start and goal position visualization
visualize_start_n_goal = True

# Threshold on the number of iterations, not used currently, ignore
n_iter = 10000

# Variable that will keep count of iterations
iters = 0 

# Limits on x and y positions of the pixel nodes to prevent out-of-index access 
x_min = 0
x_max = 600

y_min = 0
y_max = 250

# Checking if start node and goal node are in image space
if x_i[1] < x_min or x_i[1] > x_max or x_i[0] < y_min or x_i[0] > y_max or x_g[1] < x_min or x_g[1] > x_max or x_g[0] < y_min or x_g[0] > y_max:
	print("Out of bound position. Not valid!")
	print("Exiting program")
	exit()

# Creating video codecs with XVID format writer to write 2 videos 
video_writer_fourcc = cv2.VideoWriter_fourcc(*'XVID') 
video_writer_fourcc2 = cv2.VideoWriter_fourcc(*'XVID') 
# Creating a VideoWriter object

# Creating 2 video writers, one for path animation and the second for storing the explored nodes
video = cv2.VideoWriter('path_animation.avi', video_writer_fourcc, 30, (600, 250))
explore_video = cv2.VideoWriter('explore_animation.avi', video_writer_fourcc2, 60, (600, 250))

# Creating a node class to store all node parameters, including the data (node position), list of children, parent of the node, and cost to come for it
class node:
	def __init__(self, data):
		self.data = data
		self.children = []
		self.parent = None
		self.ctc = np.inf

	def append_child(self, child):
		child.parent = self
		self.children.append(child)

# Function to check if move up is valid and return the next pixel location, relative cost to come from previous node and if the next move is valid 
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

# Function to check if move down is valid and return the next pixel location, relative cost to come from previous node and if the next move is valid
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

# Function to check if move left is valid and return the next pixel location, relative cost to come from previous node and if the next move is valid
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

# Function to check if move right is valid and return the next pixel location, relative cost to come from previous node and if the next move is valid
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

# Function to check if move up-left is valid and return the next pixel location, relative cost to come from previous node and if the next move is valid
def move_up_left(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if y > y_max-2 or x < x_min+1:
		return [], np.inf, False

	if img[y+1, x-1, 1] == 255:
		return np.array([y+1, x-1]), 1.4, True
	else:
		return [], np.inf, False

# Function to check if move up-right is valid and return the next pixel location, relative cost to come from previous node and if the next move is valid
def move_up_right(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if y > y_max-2 or x > x_max-2:
		return [], np.inf, False

	if img[y+1, x+1, 1] == 255:
		return np.array([y+1, x+1]), 1.4, True
	else:
		return [], np.inf, False

# Function to check if move down-left is valid and return the next pixel location, relative cost to come from previous node and if the next move is valid
def move_down_left(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if y < y_min+1 or x < x_min+1:
		return [], np.inf, False

	if img[y-1, x-1, 1] == 255:
		return np.array([y-1, x-1]), 1.4, True
	else:
		return [], np.inf, False

# Function to check if move down-right is valid and return the next pixel location, relative cost to come from previous node and if the next move is valid
def move_down_right(node):
	curr_px = node.data # (x,y)
	y = curr_px[0]
	x = curr_px[1]
	if y < y_min+1 or x > x_max-2:
		return [], np.inf, False

	if img[y-1, x+1, 1] == 255:
		return np.array([y-1, x+1]), 1.4, True
	else:
		return [], np.inf, False

# Creating a list of all the move functions for easy iteration and reducing code length
moves_list = [move_up, move_down, move_left, move_right, move_up_left, move_up_right, move_down_left, move_down_right]

# Creating an array of nodes with 4 parameters: The node at that location, cost-to-come to that pixel from start node, flag that indicates if the node is in closed list, flag that indicates if the node is in the open list
node_arr = np.empty(shape=[250, 600, 4], dtype=object)
node_arr[:, :, 0] = None		# Node 
node_arr[:, :, 1] = np.inf		# Cost to come of current pixel
node_arr[:, :, 2] = False		# In closed list
node_arr[:, :, 3] = False		# In open list

# Creating the start and goal nodes. Initializing the start node's cost-to-come as 0
start_node = node(x_i)
start_node.ctc = 0
goal_node = node(x_g)

# Setting start node's cost-to-come in node array as 0 and also, the node at the starting position as the start node
node_arr[x_i[0], x_i[1], 1] = 0
node_arr[x_i[0], x_i[1], 0] = start_node

# Initializing the blank canvas using numpy ones function
img = np.ones([250, 600, 3])*255

# Creating a bloat grid that will be used to detect pixels that are in obstacle space, and color the pixels blue that represent the bloated region
bloat_grid = np.ones([11,11,3])*255
bloat_grid[:,:,1:3] = 0

# Drawing the obstacles in image space as black pixels, first line is for rectangles, 2nd one is for triangle, 3rd is for hexagon
x_px = 0
y_px = 0
while x_px < x_max:
	y_px = 0
	while y_px < y_max:
		if ((y_px>=0 and y_px<=100 and x_px>=100 and x_px<=150) or (y_px >= 150 and y_px <= 250 and x_px>=100 and x_px<=150) or 
		   (y_px >= 2*x_px - 895 and y_px <= -2*x_px + 1145 and x_px >= 460) or
		   ((x_px >= 235) and (x_px <= 365) and ((0.6538*x_px + y_px) >= 246.1538) and ((-0.6538*x_px + y_px) >= -146.1538) and ((-0.6538*x_px + y_px) <= 3.8461) and ((0.6538*x_px + y_px) <= 396.1538) ) ):
			img[y_px, x_px] = np.array([0,0,0], dtype=np.uint8)

		y_px += 1

	x_px += 1

# Running the bloat grid throughout the canvas and taking bitwise and to get the bloated space
for i in range(5,img.shape[0]-5):
	for j in range(5,img.shape[1]-5):
		if np.sum(img[i,j]) == 0:
			img[i-5:i+6, j-5:j+6] = cv2.bitwise_and(img[i-5:i+6, j-5:j+6, :], bloat_grid)

# Adding the 5px bloated regions at the edges of canvas that can't be taken care of from the obstacle space logic
img[:, 0:6] = cv2.bitwise_and(img[:, 0:6], np.array([255, 0, 0]) )
img[:, -5:] = cv2.bitwise_and(img[:, -5:], np.array([255, 0, 0]) )
img[0:6, :] = cv2.bitwise_and(img[0:6, :], np.array([255, 0, 0]) )
img[-5:, :] = cv2.bitwise_and(img[-5:, :], np.array([255, 0, 0]) )
img = img.astype(np.uint8)

# Checking if start node and goal node are in valid locations
if img[x_i[0], x_i[1], 1] == 0:
	print("Start position not valid!")
	print("Exiting program")
	exit()

if img[x_g[0], x_g[1], 1] == 0:
	print("Goal position not valid!")
	print("Exiting program")
	exit()

# If visualization of start and goal nodes is turned on, visualize the start and goal nodes using circles in image canvas
if visualize_start_n_goal:
	start_color = [0, 0, 255] 	# Start node as red color
	goal_color = [0, 255, 0] 	# Goal node as Green color
	cv2.circle(img, (x_i[1], x_i[0]) , color=start_color, radius=3, thickness=-1)
	cv2.circle(img, (x_g[1], x_g[0]) , color=goal_color, radius=3, thickness=-1)
	img = cv2.flip(img, flipCode=0)

# Starting the dijkstra algorithm by assiging the start node as current node and appending the start node in open list
curr_node = start_node
open_list.append([0, start_node])

# Taking copy of image for writing to the exploration video submission
og_img = img.copy()
explore_video.write(og_img)

# Main loop of the dijkstra's algorithm that continues until the current node is pointing to the goal node and until the open list is not empty
while np.count_nonzero(curr_node.data - goal_node.data) != 0 and len(open_list) != 0 :
	# Read the node with least cost-to-come in open list 
	[curr_ctc, curr_node] = open_list.pop(0)
	# Marking the node as is_in_opened_list in the node array
	node_arr[curr_node.data[0], curr_node.data[1], 2] = True

	iters += 1

	# Check new node for every possible move
	for move in moves_list:
		new_coords, incr_ctc, success = move(curr_node)

		# If the move is possible do the following, else check other moves
		if success:
			# Read the node parameters at the location generated from the above move
			new_node_params = node_arr[new_coords[0], new_coords[1]]

			# Do the following if the node is not in closed list
			if new_node_params[2] == False:
				# Do the following if the node is not in open list or the cost-to-come of current node is infinity
				if new_node_params[3] == False or new_node_params[1] == np.inf:
					# Modifying the image that is used to record the explored nodes and writing the modified frames to the explored-nodes video
					og_img[y_max - new_coords[0], new_coords[1]] = np.array([255, 255, 0])
					explore_video.write(og_img)

					# Create a new node object at the coordinates returned by the move function and assign it the correct cost-to-come
					temp_node = node(new_coords)
					temp_node.ctc = curr_node.ctc + incr_ctc
					node_arr[new_coords[0], new_coords[1], 1] = curr_node.ctc + incr_ctc

					# Appending the new node as child of current node and also assigning the parent of new node as the current node
					curr_node.append_child(temp_node)
					node_arr[new_coords[0], new_coords[1], 0] = temp_node

					# Adding the new node in open list and marking it as open in the array of nodes
					open_list.append([curr_node.ctc + incr_ctc, temp_node])

					# Sorting the open-list with respect to cost-to-come
					open_list.sort(key=lambda x:x[0]) 
					node_arr[new_coords[0], new_coords[1], 3] = True

				# If the old cost-to-come is greater than cost-to-come of parent node + the next cost-to-come, update the next node's parent and cost-to-come
				elif node_arr[new_coords[0], new_coords[1], 1] > curr_node.ctc + incr_ctc:
					# Updating the parent of next-node
					node_arr[new_coords[0], new_coords[1], 0].parent = curr_node

					# Updating the cost-to-come of next-node
					node_arr[new_coords[0], new_coords[1], 0].ctc = curr_node.ctc + incr_ctc
					node_arr[new_coords[0], new_coords[1], 1] = curr_node.ctc + incr_ctc

# Running back-tracking from the end-node to the start node
back_parent = curr_node
while back_parent != None:
	path_list.append(back_parent.data)
	back_parent = back_parent.parent

# Reverse the path list to get a sorted list from start-node to end-node
path_list = path_list[::-1]	
img = img.astype(np.uint8)

# Draw the path as black pixels on the image canvas
for i in range(len(path_list)):
	img[y_max - path_list[i][0], path_list[i][1]] = np.array([0, 0, 0])
	video.write(img)

video.release()
explore_video.release()

# Print the number of iterations taken
print("Number of iterations taken: ", iters)

# Finding and printing the total time taken by the code to generate the output and the related files
end = time.time()
print("Total time taken: ", end - start)

# Display the final image with path
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
