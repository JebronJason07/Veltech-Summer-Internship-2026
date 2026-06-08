# exercise_2_panorama_prep.py
import numpy as np
import cv2
import imutils
# Create two shifted versions of an image to simulate panorama stitching
image = cv2.imread("test_image.png") # Use your image
# Create left and right shifted versions
left_shift = imutils.translate(image, -80, 0)
right_shift = imutils.translate(image, 80, 0)
# Create overlay to show overlap
overlay = image.copy()
overlay[0:image.shape[0], 0:80] =left_shift[0:image.shape[0], image.shape[1]-80:image.shape[1]]
cv2.putText(overlay, "Overlap Region", (10, 50),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
# Create side-by-side comparison
comparison = np.hstack([left_shift, image, right_shift])
cv2.imshow("Panorama Preparation (Left | Original | Right)",
comparison)
cv2.waitKey(0)
cv2.imwrite("panorama_prep.png", comparison)
cv2.destroyAllWindows()