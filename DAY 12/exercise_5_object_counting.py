# exercise_5_object_counting.py

import cv2
import numpy as np

def count_objects(image, min_area=100):
    """
    Count objects using edge detection and contours
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edges
    edges = cv2.Canny(blurred, 30, 90)

    # Dilate to close gaps
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        edges_dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter by area
    valid_contours = [
        c for c in contours
        if cv2.contourArea(c) > min_area
    ]

    # Draw results
    result = image.copy()
    cv2.drawContours(result, valid_contours, -1, (0, 255, 0), 2)

    # Label objects
    for i, contour in enumerate(valid_contours):
        M = cv2.moments(contour)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.putText(
                result,
                str(i + 1),
                (cx - 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

    return result, len(valid_contours), edges


# Create test image with multiple objects
objects = np.zeros((400, 500, 3), dtype="uint8")

# Draw random shapes
np.random.seed(42)

for _ in range(12):
    x = np.random.randint(20, 480)
    y = np.random.randint(20, 380)
    r = np.random.randint(15, 35)

    color = np.random.randint(100, 250, 3).tolist()

    if np.random.random() > 0.5:
        cv2.circle(objects, (x, y), r, color, -1)
    else:
        cv2.rectangle(
            objects,
            (x - r, y - r),
            (x + r, y + r),
            color,
            -1
        )

# Add some noise
noise = np.random.randint(
    0, 30,
    objects.shape,
    dtype="uint8"
)

objects = cv2.add(objects, noise)

cv2.imshow("Objects to Count", objects)
cv2.waitKey(0)

# Count objects
result, count, edges = count_objects(objects)

# Show pipeline
pipeline = np.hstack([
    cv2.resize(objects, (250, 200)),
    cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (250, 200)),
    cv2.resize(result, (250, 200))
])

cv2.putText(
    pipeline,
    "Original",
    (80, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 255, 255),
    1
)

cv2.putText(
    pipeline,
    "Edges",
    (330, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 255, 255),
    1
)

cv2.putText(
    pipeline,
    f"Found: {count} objects",
    (520, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 255, 0),
    1
)

cv2.imshow("Object Counting Pipeline", pipeline)

cv2.waitKey(0)

cv2.imwrite("object_counting.png", result)

cv2.destroyAllWindows()

print(f"CHALLENGE COMPLETE! Found {count} objects.")
