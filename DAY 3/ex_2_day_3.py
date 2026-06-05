# exercise_2_day3.py

import numpy as np
import cv2

# Create a blank image
board = np.zeros((400, 400, 3), dtype="uint8")

# Draw a checkerboard pattern
square_size = 50

for y in range(0, 400, square_size):
    for x in range(0, 400, square_size):

        # Check if this is an "even" square
        if (x // square_size + y // square_size) % 2 == 0:
            board[y:y + square_size, x:x + square_size] = (255, 255, 255)
        else:
            board[y:y + square_size, x:x + square_size] = (0, 0, 0)

# Display the checkerboard
cv2.imshow("Checkerboard", board)
cv2.waitKey(0)

# Save the image
cv2.imwrite("checkerboard.png", board)

cv2.destroyAllWindows()

print("Saved checkerboard.png")