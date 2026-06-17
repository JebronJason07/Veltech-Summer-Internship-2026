# exercise_2_auto_canny.py

import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
    """
    Automatically determine Canny thresholds using median
    """
    # Compute median of pixel intensities
    v = np.median(image)

    # Apply automatic Canny
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(image, lower, upper)

    return edges, lower, upper


# Load and process image
image = cv2.imread("test_image.png")

if image is None:
    print("Error: Could not load image!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Try different sigma values
sigmas = [0.2, 0.33, 0.5, 0.7]

print("Automatic Canny with different sigma values:")

for sigma in sigmas:
    edges, lower, upper = auto_canny(blurred, sigma)

    print(f"Sigma={sigma:.2f}: lower={lower}, upper={upper}")

    # Add label
    display = edges.copy()

    cv2.putText(
        display,
        f"sigma={sigma:.2f} ({lower},{upper})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        255,
        1
    )

    cv2.imshow(f"Auto Canny - sigma={sigma}", display)
    cv2.waitKey(0)

# Find best sigma (default = 0.33)
edges_best, _, _ = auto_canny(blurred, 0.33)

# Compare with manual Canny
manual_edges = cv2.Canny(blurred, 50, 150)

# Convert to BGR for display
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
manual_bgr = cv2.cvtColor(manual_edges, cv2.COLOR_GRAY2BGR)
auto_bgr = cv2.cvtColor(edges_best, cv2.COLOR_GRAY2BGR)

comparison = np.hstack([
    cv2.resize(gray_bgr, (250, 200)),
    cv2.resize(manual_bgr, (250, 200)),
    cv2.resize(auto_bgr, (250, 200))
])

cv2.putText(comparison, "Original", (80, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.putText(comparison, "Manual (50,150)", (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.putText(comparison, "Auto Canny", (550, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.imshow("Manual vs Auto Canny", comparison)

cv2.waitKey(0)
cv2.destroyAllWindows()