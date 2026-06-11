# exercise_3_color_isolation.py
import cv2
import numpy as np
image = cv2.imread("test_image.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Define ranges for different colors
color_ranges = {
 "Red": [([0, 50, 50], [10, 255, 255]), ([160, 50, 50], [179, 255,
255])],
 "Green": [([40, 50, 50], [80, 255, 255])],
 "Blue": [([100, 50, 50], [130, 255, 255])],
 "Yellow": [([20, 50, 50], [30, 255, 255])],
 "Purple": [([130, 50, 50], [160, 255, 255])]
}
for color_name, ranges in color_ranges.items():
 # Create mask for this color
    mask = np.zeros(image.shape[:2], dtype="uint8")
    for lower, upper in ranges:
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
 
 # Keep only this color, others become grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
 
        result = np.where(mask[:, :, np.newaxis], image, gray_3ch)
 
        cv2.imshow(f"Keep Only {color_name}", result)
        cv2.waitKey(0)
        cv2.imwrite(f"color_isolation_{color_name.lower()}.png", result)
cv2.destroyAllWindows()