# day5_transformations.py
# Complete guide to image transformations
from __future__ import print_function
import numpy as np
import argparse
import cv2
import imutils # Our custom utility module
print("=" * 70)
print("DAY 5: IMAGE TRANSFORMATIONS")
print("=" * 70)
# -----------------------------------------------------------
# SECTION 1: LOAD THE IMAGE
# -----------------------------------------------------------
print("\n[Section 1] Loading the image...")
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
 help="Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
if image is None:
 print("ERROR: Could not load image!")
 exit()
height, width = image.shape[:2]
print(f"Loaded image: {width} x {height} pixels")
cv2.imshow("Original Image", image)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 2: TRANSLATION (SHIFTING)
# -----------------------------------------------------------------
print("\n[Section 2] Translation - Shifting the image...")
print(
    "Theory: Translation moves the image along X and Y axes."
    )
print("Matrix: [[1, 0, tx], [0, 1, ty]] where tx=right/left, ty=down/up")
# Method 1: Manual translation (to understand what happens)
print("\n--- Method 1: Manual translation ---")
# Shift right 50 pixels, down 100 pixels
M = np.float32([[1, 0, 50], [0, 1, 100]])
shifted_manual = cv2.warpAffine(image, M, (width, height))
cv2.imshow("Manual: Shifted Right 50, Down 100",
shifted_manual)
cv2.waitKey(0)
# Shift left 50 pixels, up 100 pixels
M = np.float32([[1, 0, -50], [0, 1, -100]])
shifted_manual = cv2.warpAffine(image, M, (width, height))
cv2.imshow("Manual: Shifted Left 50, Up 100", shifted_manual)
cv2.waitKey(0)
print("\n--- Method 2: Using our convenience function ---")
# Using imutils.translate
shifted = imutils.translate(image, 75, 50)
cv2.imshow("Convenience: Shifted Right 75, Down 50", shifted)
cv2.waitKey(0)
# Shift the image significantly to see what happens at edges
print("\n--- Edge Effects ---")
shifted_far = imutils.translate(image, 200, 150)
cv2.imshow("Shifted Far (notice black fill)", shifted_far)
cv2.waitKey(0)
print("Observation: Empty areas are filled with black (0)")
# -----------------------------------------------------------------
# SECTION 3: ROTATION
# -----------------------------------------------------------------
print("\n[Section 3] Rotation - Turning the image...")
print("Theory: Rotation spins the image around a center point.")
print("Positive angle = counter-clockwise, Negative = clockwise")
# Manual rotation
print("\n--- Manual Rotation ---")
# Get center point
center = (width // 2, height // 2)
# Rotate by 45 degrees
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_45 = cv2.warpAffine(image, M, (width, height))
cv2.imshow("Manual: Rotated 45 Degrees CCW", rotated_45)
cv2.waitKey(0)
# Rotate by -90 degrees (clockwise 90)
M = cv2.getRotationMatrix2D(center, -90, 1.0)
rotated_90 = cv2.warpAffine(image, M, (width, height))
cv2.imshow("Manual: Rotated -90 Degrees (Clockwise)",
rotated_90)
cv2.waitKey(0)
print("\n--- Using Convenience Function ---")
# Using imutils.rotate
rotated_180 = imutils.rotate(image, 180)
cv2.imshow("Convenience: Rotated 180 Degrees", rotated_180)
cv2.waitKey(0)
# Rotate by small angles
for angle in [15, 30, 45, 60]:
 rotated = imutils.rotate(image, angle)
 cv2.imshow(f"Rotated {angle} Degrees", rotated)
 cv2.waitKey(0)
print("\n--- Rotation Around Different Points ---")
# Rotate around top-left corner
top_left = (0, 0)
M = cv2.getRotationMatrix2D(top_left, 30, 1.0)
rotated_tl = cv2.warpAffine(image, M, (width, height))
cv2.imshow("Rotated Around Top-Left Corner", rotated_tl)
cv2.waitKey(0)
# Rotate around bottom-right corner
bottom_right = (width - 1, height - 1)
M = cv2.getRotationMatrix2D(bottom_right, 30, 1.0)
rotated_br = cv2.warpAffine(image, M, (width, height))
cv2.imshow("Rotated Around Bottom-Right Corner", rotated_br)
cv2.waitKey(0)
print("Observation: Rotation center dramatically changes result!")
# -----------------------------------------------------------------
# SECTION 4: RESIZING
# -----------------------------------------------------------------
print("\n[Section 4] Resizing - Changing dimensions...")
print("Theory: Important to maintain aspect ratio to avoid distortion!")
# Wrong way - distorting the image
print("\n--- WRONG: Distorting aspect ratio ---")
distorted = cv2.resize(image, (300, 300))
cv2.imshow("DISTORTED! (300x300 doesn't match original ratio)", distorted)
cv2.waitKey(0)
print("\n--- CORRECT: Maintaining aspect ratio ---")
# Resize by width (manually)
width_new = 200
ratio = width_new / width
height_new = int(height * ratio)
resized_manual = cv2.resize(image, (width_new, height_new))
print("Original: {width}x{height} → Resized: {width_new}x{height_new}")
cv2.imshow(f"Resized to width = {width_new} pixels",resized_manual)
cv2.waitKey(0)
# Resize by height (manually)
height_new = 150
ratio = height_new / height
width_new = int(width * ratio)
resized_manual = cv2.resize(image, (width_new, height_new))
print("Original: {width}x{height} → Resized: {width_new}x{height_new}")
cv2.imshow(f"Resized to height = {height_new} pixels",
resized_manual)
cv2.waitKey(0)
print("\n--- Using Convenience Function ---")
# Using imutils.resize
resized_width = imutils.resize(image, width=300)
cv2.imshow("Resized to width=300 (aspect ratio preserved)",
resized_width)
cv2.waitKey(0)
resized_height = imutils.resize(image, height=200)
cv2.imshow("Resized to height=200 (aspect ratio preserved)",
resized_height)
cv2.waitKey(0)
# Create multiple resized versions
print("\n--- Size Comparison ---")
sizes = [100, 200, 300, 400, 500]
resized_variants = []
for size in sizes:
 resized = imutils.resize(image, width=size)
 resized_variants.append(resized)
 cv2.imshow(f"Width = {size} pixels", resized)
 cv2.waitKey(0)
print("Observation: Smaller images lose detail but load faster!")
# -----------------------------------------------------------------
# SECTION 5: FLIPPING
# -----------------------------------------------------------------
print("\n[Section 5] Flipping - Mirroring the image...")
print("Theory: Flip codes: 1=horizontal, 0=vertical, -1=both")
# Horizontal flip (mirror left-right)
flipped_h = cv2.flip(image, 1)
cv2.imshow("Horizontal Flip (Mirror)", flipped_h)
cv2.waitKey(0)
# Vertical flip (mirror top-bottom)
flipped_v = cv2.flip(image, 0)
cv2.imshow("Vertical Flip (Upside Down)", flipped_v)
cv2.waitKey(0)
# Both flips (rotate 180 - different from rotation!)
flipped_both = cv2.flip(image, -1)
cv2.imshow("Both Flips (Horizontal + Vertical)",
flipped_both)
cv2.waitKey(0)
print("\n--- Using Convenience Function ---")
flipped = imutils.flip(image, 'horizontal')
cv2.imshow("Using imutils.flip - Horizontal", flipped)
cv2.waitKey(0)
flipped = imutils.flip(image, 'vertical')
cv2.imshow("Using imutils.flip - Vertical", flipped)
cv2.waitKey(0)
flipped = imutils.flip(image, 'both')
cv2.imshow("Using imutils.flip - Both", flipped)
cv2.waitKey(0)
# Create a split-screen comparison
print("\n--- Comparison View ---")
top_row = np.hstack([image, flipped_h])
cv2.imshow("Original | Horizontal Flip", top_row)
cv2.waitKey(0)
bottom_row = np.hstack([flipped_v, flipped_both])
combined = np.vstack([top_row, bottom_row])
cv2.imshow("All Flips: Original | Horizontal | Vertical | Both", combined)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 6: CROPPING
# -----------------------------------------------------------------
print("\n[Section 6] Cropping - Extracting regions...")
print("Theory: Use NumPy slicing: image[start_y:end_y, start_x:end_x]")
print("Remember: (y, x) order! First rows, then columns.")
# Crop the top-left 100x100 region
cropped_tl = image[0:100, 0:100]
cv2.imshow("Cropped: Top-Left 100x100", cropped_tl)
cv2.waitKey(0)
# Crop the center region
center_y = height // 2
center_x = width // 2
crop_size = 150
cropped_center = image[
 center_y - crop_size//2 : center_y + crop_size//2,
 center_x - crop_size//2 : center_x + crop_size//2
]
cv2.imshow(f"Cropped: Center {crop_size}x{crop_size}",
cropped_center)
cv2.waitKey(0)
# Crop a specific object area (example)
print("\n--- Find and Crop Interesting Regions ---")
# Show image with grid to help find coordinates
grid_image = image.copy()
for i in range(0, width, 50):
 cv2.line(grid_image, (i, 0), (i, height), (0, 255, 0), 1)
for i in range(0, height, 50):
 cv2.line(grid_image, (0, i), (width, i), (0, 255, 0), 1)
cv2.imshow("Image with Grid (50px spacing)", grid_image)
cv2.waitKey(0)
print("\n--- Using Convenience Function ---")
# Using imutils.crop
cropped = imutils.crop(image, 100, 300, 50, 250)
cv2.imshow("Using imutils.crop", cropped)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 7: COMBINING TRANSFORMATIONS
# -----------------------------------------------------------------
print("\n[Section 7] Combining Multiple Transformations...")
# Start with original
result = image.copy()
# 1. Crop to focus on center
result = imutils.crop(result,
 height//4, 3*height//4,
 width//4, 3*width//4)
# 2. Resize to standard size
result = imutils.resize(result, width=400)
# 3. Rotate slightly for artistic effect
result = imutils.rotate(result, 5)
# 4. Add a border to show final result
cv2.rectangle(result, (0, 0), (result.shape[1]-1,
result.shape[0]-1),
 (0, 0, 255), 5)
cv2.imshow("Combined: Crop → Resize → Rotate → Border",
result)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 8: PRACTICE CHALLENGE
# -----------------------------------------------------------------
print("\n[Section 8] Challenge: Create a Collage!")
print("Follow these steps to create your own image collage:")
# Create a blank canvas for collage
collage = np.zeros((600, 800, 3), dtype="uint8")
# 1. Take the original image, resize to 300x200
img1 = imutils.resize(image, width=300)
img1 = imutils.crop(img1, 0, 200, 0, 300) # Ensure exact 
size
# 2. Flip horizontally
img2 = imutils.flip(image, 'horizontal')
img2 = imutils.resize(img2, width=300)
# 3. Rotate 90 degrees
img3 = imutils.rotate(image, 90)
img3 = imutils.resize(img3, width=300)
# 4. Crop the center
img4 = imutils.crop(image, height//3, 2*height//3, width//3,2*width//3)
img4 = imutils.resize(img4, width=300)
# Place images on collage
collage[0:200, 0:300] = img1
collage[0:200, 350:650] = img2
collage[250:450, 0:300] = img3
collage[250:450, 350:650] = img4
# Add labels
cv2.putText(collage, "Original", (100, 30),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv2.putText(collage, "Horizontal Flip", (430, 30),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv2.putText(collage, "Rotated 90", (100, 280),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv2.putText(collage, "Center Crop", (430, 280),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv2.imshow("MY COLLAGE - Day 5 Challenge Complete!", collage)
cv2.waitKey(0)
# Save your work
cv2.imwrite("collage_result.png", collage)
print("\nCollage saved as 'collage_result.png'")
print("\n" + "=" * 70)
print("DAY 5 COMPLETE!")
print("=" * 70)
cv2.destroyAllWindows()