# exercise_4_advanced_shape_detection.py

import cv2
import numpy as np


def advanced_shape_detection(image_path):
    """Detect shapes with advanced filtering"""

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours_info = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours_info[-2]

    result = image.copy()
    shape_data = {}

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 50:
            continue

        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        vertices = len(approx)

        # Additional metrics
        circularity = (
            4 * np.pi * area / (perimeter ** 2)
            if perimeter > 0 else 0
        )

        x, y, w, h = cv2.boundingRect(contour)

        # Shape classification
        if vertices <= 4:

            if circularity > 0.8 and vertices > 6:
                shape = "Circle"
                confidence = circularity

            elif vertices == 3:
                shape = "Triangle"
                confidence = 0.90

            elif vertices == 4:
                aspect_ratio = w / h if h > 0 else 1

                if 0.9 <= aspect_ratio <= 1.1:
                    shape = "Square"
                    confidence = 0.95
                else:
                    shape = "Rectangle"
                    confidence = 0.90

            else:
                shape = "Unknown"
                confidence = 0.50

        elif vertices <= 6:

            if circularity > 0.6:
                shape = f"{vertices}-gon"
                confidence = 0.70
            else:
                shape = f"{vertices}-sided"
                confidence = 0.50

        else:

            if circularity > 0.7:
                shape = "Circle"
                confidence = circularity
            else:
                shape = "Complex"
                confidence = 0.30

        # Store statistics
        if shape not in shape_data:
            shape_data[shape] = {
                'count': 0,
                'areas': [],
                'confidence': []
            }

        shape_data[shape]['count'] += 1
        shape_data[shape]['areas'].append(area)
        shape_data[shape]['confidence'].append(confidence)

        # Draw contour
        color = np.random.randint(0, 256, 3).tolist()

        cv2.drawContours(
            result,
            [contour],
            -1,
            color,
            2
        )

        # Label
        M = cv2.moments(contour)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.putText(
                result,
                shape,
                (cx - 20, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )

    # Print statistics
    print("\nAdvanced Shape Detection Results")
    print("-" * 50)

    for shape, data in shape_data.items():

        avg_area = np.mean(data['areas'])
        avg_conf = np.mean(data['confidence'])

        print(
            f"{shape}: {data['count']} objects, "
            f"avg area={avg_area:.0f}, "
            f"confidence={avg_conf:.2f}"
        )

    cv2.imshow("Advanced Shape Detection", result)
    cv2.imwrite("advanced_shapes.png", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run
advanced_shape_detection("test_image.png")