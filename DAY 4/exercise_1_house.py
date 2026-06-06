# exercise_1_house.py

import numpy as np
import cv2

# Create canvas
canvas = np.ones((500, 500, 3), dtype="uint8") * 255

# Draw the house body
cv2.rectangle(canvas, (150, 250), (350, 450), (0, 0, 0), 3)

# Draw the roof
cv2.line(canvas, (150, 250), (250, 120), (0, 0, 255), 3)
cv2.line(canvas, (250, 120), (350, 250), (0, 0, 255), 3)

# Draw the door
cv2.rectangle(canvas, (220, 350), (280, 450), (255, 0, 0), -1)

# Draw windows
cv2.rectangle(canvas, (170, 280), (220, 330), (0, 255, 0), -1)
cv2.rectangle(canvas, (280, 280), (330, 330), (0, 255, 0), -1)

# Draw the sun
cv2.circle(canvas, (400, 80), 40, (0, 255, 255), -1)

# Add title
cv2.putText(
    canvas,
    "My Dream House",
    (120, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 0, 0),
    2
)

cv2.imshow("House", canvas)
cv2.waitKey(0)

cv2.imwrite("house.png", canvas)

cv2.destroyAllWindows()

print("House saved as 'house.png'")