import cv2
import numpy as np

def extract_license_plate(image):
    """
    Simulate license plate extraction using thresholding
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    methods = {}

    # Method 1: Simple threshold
    _, methods["Simple (T=127)"] = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY
    )

    # Method 2: Otsu
    _, methods["Otsu"] = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Method 3: Adaptive
    methods["Adaptive"] = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        5
    )

    # Method 4: Adaptive with larger block
    methods["Adaptive (Large)"] = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8
    )

    return methods


# Create a simulated license plate image
plate = np.zeros((150, 400, 3), dtype="uint8")

# Background
plate[:, :] = (200, 200, 200)

# Plate text
cv2.putText(
    plate,
    "ABC-1234",
    (80, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    2,
    (0, 0, 0),
    5
)

# Add noise and shadow
noise = np.random.randint(0, 50, plate.shape, dtype="uint8")
plate_noisy = cv2.add(plate, noise)

# Shadow on right side
plate_noisy[:, 250:] = (plate_noisy[:, 250:] * 0.6).astype(np.uint8)

cv2.imshow("Simulated License Plate", plate_noisy)
cv2.waitKey(0)

# Extract plate using different methods
results = extract_license_plate(plate_noisy)

# Compare all methods
display_images = [plate_noisy]
method_names = ["Original"]

for name, result in results.items():
    display_images.append(
        cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    )
    method_names.append(name)

# Create grid
rows = []
row = []

for img, name in zip(display_images, method_names):

    img_resized = cv2.resize(img, (250, 100))

    cv2.putText(
        img_resized,
        name,
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1
    )

    row.append(img_resized)

    if len(row) == 2:
        rows.append(np.hstack(row))
        row = []

if row:
    # Fill last row if needed
    blank = np.zeros_like(row[0])
    row.append(blank)
    rows.append(np.hstack(row))

grid = np.vstack(rows)

cv2.imshow(
    "License Plate Extraction - Method Comparison",
    grid
)

cv2.waitKey(0)

print("\nCHALLENGE: Which method best extracts the license plate text?")
print("Answer: Adaptive thresholding handles shadows best!")

cv2.destroyAllWindows()