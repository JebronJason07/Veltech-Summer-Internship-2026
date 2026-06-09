# exercise_4_color_splash.py
import cv2
import numpy as np
image = cv2.imread("test_image.png")
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
# Create a mask for a specific region (circle in center)
mask = np.zeros(image.shape[:2], dtype="uint8")
center = (image.shape[1]//2, image.shape[0]//2)
radius = min(image.shape[0], image.shape[1]) // 4
cv2.circle(mask, center, radius, 255, -1)
# Apply mask: color inside mask, grayscale outside
mask_inv = cv2.bitwise_not(mask)
# Inside mask: keep color
inside = cv2.bitwise_and(image, image, mask=mask)
# Outside mask: grayscale
outside = cv2.bitwise_and(gray_color, gray_color, mask=mask_inv)
# Combine
splash = cv2.add(inside, outside)
# Draw circle boundary
cv2.circle(splash, center, radius, (0, 255, 0), 3)
cv2.imshow("Color Splash Effect", splash)
cv2.waitKey(0)
cv2.imwrite("color_splash.png", splash)
cv2.destroyAllWindows()