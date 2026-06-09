# exercise_5_heart_mask.py
import cv2
import numpy as np
import math
def create_heart_mask(size):
    """Create a heart-shaped mask"""
    mask = np.zeros((size, size), dtype="uint8")
    center = size // 2

    for y in range(size):
        for x in range(size):
            x_norm = (x - center) / center
            y_norm = (y - center) / center

            x2 = x_norm * x_norm
            y2 = y_norm * y_norm
            value = (x2 + y2 - 0.8) ** 3 - x2 * y_norm ** 3

            if value < 0:
                mask[y, x] = 255

    return mask

# Load images
image1 = cv2.imread("test_image.png")
image2 = cv2.imread("test_pattern.png")
# Resize to same square size
size = min(image1.shape[0], image1.shape[1], image2.shape[0],
image2.shape[1])
size = min(size, 400) # Limit size
image1 = cv2.resize(image1, (size, size))
image2 = cv2.resize(image2, (size, size))
# Create heart mask
heart = create_heart_mask(size)
heart_inv = cv2.bitwise_not(heart)
# Apply mask
result = cv2.bitwise_and(image1, image1, mask=heart)
result2 = cv2.bitwise_and(image2, image2, mask=heart_inv)
final = cv2.add(result, result2)
# Draw heart outline
contours, _ = cv2.findContours(heart, cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(final, contours, -1, (0, 255, 0), 2)
cv2.imshow("Heart Mask", heart)
cv2.imshow("Heart Blended Result", final)
cv2.waitKey(0)
cv2.imwrite("heart_blend.png", final)
cv2.destroyAllWindows()