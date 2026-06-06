# exercise_2_target.py

import numpy as np
import cv2

WHITE = (255, 255, 255)

canvas = np.zeros((600, 600, 3), dtype="uint8")

center = (300, 300)

# Colors for target rings (outer to inner)
colors = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (0, 255, 255),    # Yellow
    (255, 255, 0),    # Cyan
    (255, 0, 255)     # Magenta
]

# Draw rings
for i, color in enumerate(colors):
    radius = 250 - (i * 40)
    thickness = 10 if i < 5 else -1
    cv2.circle(canvas, center, radius, color, thickness)

# Add crosshairs
cv2.line(
    canvas,
    (center[0] - 50, center[1]),
    (center[0] + 50, center[1]),
    WHITE,
    2
)

cv2.line(
    canvas,
    (center[0], center[1] - 50),
    (center[0], center[1] + 50),
    WHITE,
    2
)

cv2.imshow("Target Board", canvas)
cv2.waitKey(0)

cv2.imwrite("target.png", canvas)

cv2.destroyAllWindows()

print("Target saved as 'target.png'")