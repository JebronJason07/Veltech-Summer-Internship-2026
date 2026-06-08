# exercise_5_image_alignment.py
import cv2
import numpy as np
import imutils
# Load an image
image = cv2.imread("test_image.png")
# Simulate a misaligned image by rotating and shifting
misaligned = image.copy()
misaligned = imutils.rotate(misaligned, -8) # Tilted
misaligned = imutils.translate(misaligned, 30, 20) # Shifted
# Your task: Try to align the misaligned image back to original
# Step 1: Detect edges or features
# Step 2: Find rotation angle needed
# Step 3: Find translation needed
# Step 4: Apply inverse transformations
# Solution approach (for learning)
print("CHALLENGE: Can you align the misaligned image?")
print("Hint: Try rotating by 8 degrees and translating by -30, -20")
# Try to fix it
fixed = misaligned.copy()
fixed = imutils.rotate(fixed, 8) # Counter-rotate
fixed = imutils.translate(fixed, -30, -20) # Counter-shift
# Compare
comparison = np.hstack([image, misaligned, fixed])
cv2.imshow("Original | Misaligned | Fixed", comparison)
cv2.waitKey(0)
cv2.imwrite("alignment_result.png", comparison)
cv2.destroyAllWindows()