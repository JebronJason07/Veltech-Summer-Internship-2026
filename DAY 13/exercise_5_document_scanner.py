import cv2
import numpy as np


def order_points(pts):
    """Order points as: top-left, top-right, bottom-right, bottom-left"""

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def scan_document(image_path):
    """Find and extract document from image using contours"""

    image = cv2.imread(image_path)

    if image is None:
        print("Could not load image")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(
        edges_dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print("No contours found!")
        return

    # Largest contour
    largest = max(contours, key=cv2.contourArea)

    # Polygon approximation
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    print(f"Document corners found: {len(approx)}")

    result = image.copy()

    cv2.drawContours(result, [approx], -1, (0, 255, 0), 3)

    # Draw corner points
    for point in approx:
        x, y = point[0]
        cv2.circle(result, (x, y), 8, (0, 0, 255), -1)

    cv2.imshow("Document Detected", result)

    # Perspective transform if 4 corners found
    if len(approx) == 4:

        pts = approx.reshape(4, 2).astype("float32")
        pts = order_points(pts)

        (tl, tr, br, bl) = pts

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(pts, dst)

        warped = cv2.warpPerspective(
            image,
            M,
            (maxWidth, maxHeight)
        )

        cv2.imshow("Scanned Document", warped)

        cv2.imwrite(
            "scanned_document.png",
            warped
        )

        print("Scanned document saved as scanned_document.png")

    else:
        print("Document is not a quadrilateral. Cannot perform scan.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run document scanner
scan_document("test_image.png")