# exercise_3_document_edges.py

import cv2
import numpy as np

def find_document_edges(image):
    """
    Find the edges of a document in an image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to connect edges
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        edges_dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Find largest contour (should be the document)
    if contours:
        largest = max(contours, key=cv2.contourArea)

        # Approximate polygon
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        # Draw on image
        result = image.copy()
        cv2.drawContours(result, [approx], -1, (0, 255, 0), 3)

        return result, approx

    return image, None


# Create a document simulation
doc = np.ones((500, 700, 3), dtype="uint8") * 220  # Light background

cv2.rectangle(doc, (100, 50), (600, 450), (180, 180, 180), -1)

cv2.putText(
    doc,
    "DOCUMENT EDGE DETECTION",
    (150, 150),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 0),
    2
)

cv2.putText(
    doc,
    "Find the document boundaries!",
    (150, 200),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 0, 0),
    2
)

# Add some background noise
background = np.random.randint(0, 50, doc.shape, dtype="uint8")
doc = cv2.add(doc, background)

cv2.imshow("Document Image", doc)
cv2.waitKey(0)

# Find document edges
result, approx = find_document_edges(doc)

if approx is not None:
    cv2.imshow("Document Edges Found", result)
    print(f"Found document with {len(approx)} corners")
else:
    print("Document not found!")

cv2.waitKey(0)
cv2.destroyAllWindows()