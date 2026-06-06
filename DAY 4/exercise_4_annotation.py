# exercise_4_annotate.py

import argparse
import cv2

# Colors (BGR format)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

if image is None:
    print("Error: Could not load image")
    exit()

height, width = image.shape[:2]

# Draw a rectangle around the center region
center_x, center_y = width // 2, height // 2
rect_size = 100

cv2.rectangle(
    image,
    (center_x - rect_size, center_y - rect_size),
    (center_x + rect_size, center_y + rect_size),
    RED,
    3
)

# Draw a circle marking the exact center
cv2.circle(image, (center_x, center_y), 10, GREEN, -1)

# Add text annotations
cv2.putText(
    image,
    "CENTER REGION",
    (center_x - 80, center_y - rect_size - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    WHITE,
    2
)

cv2.putText(
    image,
    f"Size: {width}x{height}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    YELLOW,
    2
)

cv2.imshow("Annotated Image", image)
cv2.waitKey(0)

cv2.imwrite("annotated_image.jpg", image)

cv2.destroyAllWindows()

print("Saved annotated_image.jpg")