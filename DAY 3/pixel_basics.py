# pixel_basics.py
# Learning to access and manipulate individual pixels
from __future__ import print_function
import argparse
import cv2
import numpy as np
print("=" * 50)
print("DAY 3: PIXEL BASICS")
print("=" * 50)
# ----- PART A: SETUP AND LOAD IMAGE -----
print("\n[Step 1] Loading image...")
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
 help="Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
if image is None:
 print("ERROR: Could not load image!")
 exit()
print("Image loaded successfully!")
print("Image shape (height, width, channels):", image.shape)
# Display original
cv2.imshow("Original Image", image)
cv2.waitKey(0)
# ----- PART B: ACCESSING A SINGLE PIXEL -----
print("\n[Step 2] Accessing individual pixels...")
# Access pixel at top-left corner (0, 0)
(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - B:{}, G:{}, R:{}".format(b, g, r))
# Access pixel at center
height = image.shape[0]
width = image.shape[1]
center_y = height // 2
center_x = width // 2
(b, g, r) = image[center_y, center_x]
print("Pixel at center ({}, {}) - B:{}, G:{}, R:{}".format(
 center_x, center_y, b, g, r))
# Access pixel at bottom-right
(b, g, r) = image[height-1, width-1]
print("Pixel at bottom-right ({}, {}) - B:{}, G:{}, R:{}".format(
 width-1, height-1, b, g, r))
cv2.waitKey(0)
# ----- PART C: MODIFYING A SINGLE PIXEL -----
print("\n[Step 3] Modifying individual pixels...")
# Create a copy for modification
modified = image.copy()
# Change the top-left pixel to RED (in BGR, that's 0,0,255)
print("Changing top-left pixel to RED...")
modified[0, 0] = (0, 0, 255) # B=0, G=0, R=255
# Check the change
(b, g, r) = modified[0, 0]
print("After change - B:{}, G:{}, R:{}".format(b, g, r))
# Change the center pixel to GREEN
print("Changing center pixel to GREEN...")
modified[center_y, center_x] = (0, 255, 0) # B=0, G=255, R=0
# Change the bottom-right pixel to BLUE
print("Changing bottom-right pixel to BLUE...")
modified[height-1, width-1] = (255, 0, 0) # B=255, G=0, R=0
cv2.imshow("Modified - Single Pixels Changed", modified)
cv2.waitKey(0)
# ----- PART D: MODIFYING REGIONS (SLICING) -----
print("\n[Step 4] Modifying regions using NumPy slicing...")
# Create another copy
region_modified = image.copy()
# Draw a 50x50 green square in the top-left
print("Drawing a 50x50 GREEN square at top-left...")
region_modified[0:50, 0:50] = (0, 255, 0)
cv2.imshow("Green Square", region_modified)
cv2.waitKey(0)
# Draw a 100x100 blue square in the center
print("Drawing a 100x100 BLUE square at center...")
# Calculate square boundaries
start_y = (height // 2) - 50
end_y = (height // 2) + 50
start_x = (width // 2) - 50
end_x = (width // 2) + 50
region_modified[start_y:end_y, start_x:end_x] = (255, 0, 0)
cv2.imshow("Green + Blue Squares", region_modified)
cv2.waitKey(0)
# ----- PART E: EXTRACTING REGIONS (CROPPING) -----
print("\n[Step 5] Extracting regions (cropping)...")
# Crop a 100x100 region from top-left
crop = image[0:100, 0:100]
print("Cropped shape:", crop.shape)
cv2.imshow("Cropped - Top Left 100x100", crop)
cv2.waitKey(0)
# Crop the center of the image
center_crop = image[
 height//4 : 3*height//4,
 width//4 : 3*width//4
]
cv2.imshow("Cropped - Center Region", center_crop)
cv2.waitKey(0)
# ----- PART F: UNDERSTANDING BGR VS RGB -----
print("\n[Step 6] Demonstrating BGR vs RGB...")
# Create a small test image to understand color order
test = np.zeros((200, 600, 3), dtype="uint8")
# Left section - Pure Blue (BGR: 255,0,0)
test[0:200, 0:200] = (255, 0, 0)
cv2.putText(test, "BGR: (255,0,0) = BLUE", (10, 100),
 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
# Middle section - Pure Green (BGR: 0,255,0)
test[0:200, 200:400] = (0, 255, 0)
cv2.putText(test, "BGR: (0,255,0) = GREEN", (210, 100),
 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
# Right section - Pure Red (BGR: 0,0,255)
test[0:200, 400:600] = (0, 0, 255)
cv2.putText(test, "BGR: (0,0,255) = RED", (410, 100),
 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.imshow("BGR Color Demonstration", test)
cv2.waitKey(0)
print("\n" + "=" * 50)
print("DAY 3 COMPLETE!")
print("=" * 50)
cv2.destroyAllWindows()