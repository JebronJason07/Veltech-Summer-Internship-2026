# exercise_4_smart_crop.py
import cv2
import numpy as np
import imutils
image = cv2.imread("test_pattern.png") # Use pattern image
# Convert to grayscale and find edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
if contours:
 # Find largest contour by area
    largest = max(contours, key=cv2.contourArea)
 
 # Get bounding box
    x, y, w, h = cv2.boundingRect(largest)
 
 # Crop to bounding box with padding
    padding = 20
    y_start = max(0, y - padding)
    y_end = min(image.shape[0], y + h + padding)
    x_start = max(0, x - padding)
    x_end = min(image.shape[1], x + w + padding)
 
    cropped = image[y_start:y_end, x_start:x_end]
 
 # Draw bounding box on original
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0),3)
 
    cv2.imshow("Original with Bounding Box", image)
    cv2.imshow("Smart Cropped Object", cropped)
    cv2.waitKey(0)
    cv2.imwrite("smart_crop.png", cropped)
cv2.destroyAllWindows()