# exercise_1_directional_edges.py

import cv2
import numpy as np

def analyze_edges(image):
    """Analyze edge directions using Sobel"""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # Compute magnitude and direction
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

    # Normalize for display
    magnitude = np.uint8(
        np.clip(magnitude / np.max(magnitude) * 255, 0, 255)
    )

    # Create direction color map
    direction_vis = np.zeros_like(image)
    mask = magnitude > 30

    # Color code directions
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            if mask[y, x]:
                angle = direction[y, x]

                # Map angle to color
                if -22.5 < angle <= 22.5:  # Horizontal
                    direction_vis[y, x] = (0, 0, 255)  # Red

                elif 22.5 < angle <= 67.5:  # Diagonal right
                    direction_vis[y, x] = (0, 255, 255)  # Yellow

                elif 67.5 < angle <= 112.5:  # Vertical
                    direction_vis[y, x] = (0, 255, 0)  # Green

                elif 112.5 < angle <= 157.5:  # Diagonal left
                    direction_vis[y, x] = (255, 0, 0)  # Blue

                else:  # Horizontal
                    direction_vis[y, x] = (0, 0, 255)  # Red

    return direction_vis, magnitude


image = cv2.imread("test_image.png")

if image is not None:
    direction_vis, magnitude = analyze_edges(image)

    # Convert magnitude to BGR for stacking
    magnitude_bgr = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

    comparison = np.hstack([
        cv2.resize(image, (350, 250)),
        cv2.resize(magnitude_bgr, (350, 250)),
        cv2.resize(direction_vis, (350, 250))
    ])

    cv2.imshow("Edge Direction Analysis", comparison)
    cv2.waitKey(0)

    cv2.imwrite("edge_directions.png", direction_vis)

cv2.destroyAllWindows()