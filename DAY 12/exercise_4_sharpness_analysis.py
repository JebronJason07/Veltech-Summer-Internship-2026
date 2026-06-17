# exercise_4_sharpness_analysis.py

import cv2
import numpy as np


def analyze_sharpness(image):
    """
    Analyze image sharpness using Laplacian variance
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Compute variance
    variance = laplacian.var()

    # Create visualization
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    laplacian_vis = cv2.cvtColor(laplacian_abs, cv2.COLOR_GRAY2BGR)

    # Determine sharpness status
    if variance < 100:
        status = "BLURRY"
        color = (0, 0, 255)
    elif variance < 500:
        status = "MODERATE"
        color = (0, 255, 255)
    else:
        status = "SHARP"
        color = (0, 255, 0)

    cv2.putText(
        laplacian_vis,
        f"Sharpness Score: {variance:.0f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    cv2.putText(
        laplacian_vis,
        f"Status: {status}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    return laplacian_vis, variance, status


# Load image
image = cv2.imread("test_image.png")

if image is None:
    print("Error: Could not load image!")
    exit()

# Create blurry versions
blurry = cv2.GaussianBlur(image, (15, 15), 0)
very_blurry = cv2.GaussianBlur(image, (31, 31), 0)

# Analyze all versions
results = []

for img, name in [
    (image, "Original"),
    (blurry, "Blurry"),
    (very_blurry, "Very Blurry")
]:
    laplacian_vis, variance, status = analyze_sharpness(img)
    results.append((img, laplacian_vis, name, variance, status))

# Create comparison grid
row_height = 130
display = np.zeros((row_height * len(results), 900, 3), dtype="uint8")

for i, (img, lap, name, var, status) in enumerate(results):

    y_start = i * row_height
    y_end = y_start + row_height

    display[y_start:y_end, 0:300] = cv2.resize(img, (300, row_height))
    display[y_start:y_end, 300:600] = cv2.resize(lap, (300, row_height))

    cv2.putText(
        display,
        name,
        (10, y_start + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        display,
        f"Score: {var:.0f}",
        (10, y_start + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1
    )

    cv2.putText(
        display,
        f"Status: {status}",
        (10, y_start + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1
    )

cv2.imshow("Image Sharpness Analysis", display)
cv2.imwrite("sharpness_analysis.png", display)

cv2.waitKey(0)
cv2.destroyAllWindows()