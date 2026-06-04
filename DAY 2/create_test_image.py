import numpy as np
import cv2

# Create a colorful 400x400 image
image = np.zeros((400, 400, 3), dtype="uint8")

# Draw a red rectangle
cv2.rectangle(image, (50, 50), (200, 200), (0, 0, 255), -1)

# Draw a green circle
cv2.circle(image, (300, 300), 80, (0, 255, 0), -1)

# Draw a blue line
cv2.line(image, (0, 0), (400, 400), (255, 0, 0), 10)

# Save the image
cv2.imwrite("test_image.png", image)
print("Created test_image.png")

# Also save a grayscale version
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("test_image_gray.png", gray)
print("Created test_image_gray.png")