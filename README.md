# ENPM661_project2
Github repo for project 2 of ENPM661 - Planning for Autonomous Robots course

### General note:
I have used a hash map to store the node and their parameters as per the pixels in the canvas. This reduces the time required to check if an element is in open list from O(n) to O(1), which improves performance by upto 20 times. The data structure used in an 3D array where the first 2 dimensions represent the y and x coordinates of the image pixel locations, and the 3rd dimension represents the parameters of each node element, such as ctc, is_opened, is_closed and the node object itself.

### Libraries:
OpenCV, Numpy, time

### Instructions to run:
1. Run the file using console/terminal or IDE that supports user input.
2. The code will prompt the user 4 times for 4 inputs corresponding to x and y pixel values for start and goal nodes.
3. Enter x, y coordinates for start and goal nodes. If these are invalid, i.e. in obstacle space or out of bounds, then the code will print that the input is invalid and the program will exit.
4. After entering valid coordinates, the code will compute the path from start node to goal node, generate 2 videos, one for path animation and another for node exploration.
5. The code will display the final image containing the start node, goal node, and a valid path between them. Press any key on image window to exit the program.
6. (Optional) You can disable the displaying of start and goal nodes by setting the 'visualize_start_n_goal' as False.
