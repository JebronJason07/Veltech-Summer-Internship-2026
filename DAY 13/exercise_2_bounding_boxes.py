import cv2
import numpy as np

def draw_bboxes_and_labels(image_path):
    """Draw bounding boxes and labels for all objects"""

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

    result = image.copy()

    for i, contour in enumerate(contours):

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Draw rectangle
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Draw contour
        cv2.drawContours(
            result,
            [contour],
            -1,
            (255, 0, 0),
            1
        )

        # Add label with area
        area = cv2.contourArea(contour)

        label = f"#{i+1} ({area:.0f})"

        cv2.putText(
            result,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1
        )

    cv2.imshow("Bounding Boxes and Labels", result)
    cv2.waitKey(0)

    cv2.imwrite("bbox_labels.png", result)
    print("Saved: bbox_labels.png")


draw_bboxes_and_labels("test_image.png")

cv2.destroyAllWindows()