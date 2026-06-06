# exercise_3_chessboard.py

import numpy as np
import cv2

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create 8x8 chessboard on a 400x400 canvas
square_size = 50
board = np.zeros((400, 400, 3), dtype="uint8")

for row in range(8):
    for col in range(8):

        # Calculate pixel coordinates
        y_start = row * square_size
        y_end = (row + 1) * square_size
        x_start = col * square_size
        x_end = (col + 1) * square_size

        # Alternate colors
        if (row + col) % 2 == 0:
            board[y_start:y_end, x_start:x_end] = WHITE
        else:
            board[y_start:y_end, x_start:x_end] = BLACK

# Add labels
cv2.putText(
    board, "a", (15, 390),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    (128, 128, 128), 1
)

cv2.putText(
    board, "b", (65, 390),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    (128, 128, 128), 1
)

cv2.putText(
    board, "h", (360, 390),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    (128, 128, 128), 1
)

# Display image
cv2.imshow("Chessboard", board)
cv2.waitKey(0)

# Save image
cv2.imwrite("chessboard.png", board)

# Close windows
cv2.destroyAllWindows()

print("Chessboard saved as 'chessboard.png'")