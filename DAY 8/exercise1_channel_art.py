# exercise_1_channel_art.py

import cv2
import numpy as np

image = cv2.imread("test_image.png")

# Split channels
b, g, r = cv2.split(image)

# Create artistic effects
effects = {
    "Cyanotype": cv2.merge([b, b, b]),  # Blue only
    "Sepia": cv2.merge([b, g, (r * 0.7).astype("uint8")]),
    "Magenta Dream": cv2.merge([
        (b * 0.5).astype("uint8"),
        (g * 0.3).astype("uint8"),
        r
    ]),
    "Green Screen": cv2.merge([
        (b * 0.2).astype("uint8"),
        g,
        (r * 0.2).astype("uint8")
    ]),
}

for name, img in effects.items():
    cv2.imshow(f"Effect: {name}", img)
    cv2.waitKey(0)
    cv2.imwrite(f"art_{name.lower()}.png", img)

cv2.destroyAllWindows()