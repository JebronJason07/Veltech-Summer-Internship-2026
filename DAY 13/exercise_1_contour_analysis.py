import cv2
import numpy as np

def analyze_contours(image_path):
    """Analyze contours and print statistics"""

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print("Contour Analysis Report")
    print("-" * 50)

    areas = []
    perimeters = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        areas.append(area)
        perimeters.append(perimeter)

        print(
            f"Contour {i+1}: Area={area:.0f}, "
            f"Perimeter={perimeter:.1f}"
        )

    print("-" * 50)
    print(f"Total contours: {len(contours)}")

    if len(areas) > 0:
        print(f"Total area: {sum(areas):.0f} pixels")
        print(f"Average area: {np.mean(areas):.0f}")
        print(f"Largest: {max(areas):.0f}")
        print(f"Smallest: {min(areas):.0f}")
    else:
        print("No contours found.")

    return contours, areas, perimeters


# Run analysis
analyze_contours("test_image.png")

cv2.destroyAllWindows()