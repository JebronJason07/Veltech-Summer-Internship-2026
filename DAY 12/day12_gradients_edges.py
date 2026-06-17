# day12_gradients_edges.py
# Complete guide to gradients and edge detection
from __future__ import print_function
import numpy as np
import argparse
import cv2
print("=" * 70)
print("DAY 12: GRADIENTS AND EDGE DETECTION")
print("=" * 70)
# -----------------------------------------------------------------
# SECTION 1: LOAD IMAGE
# -----------------------------------------------------------------
print("\n[Section 1] Loading image...")
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
if image is None:
    print("ERROR: Could not load image!")
    exit()
height, width = image.shape[:2]
print(f"Loaded: {width} x {height} pixels")
# Convert to grayscale (edges work on single channel)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original Image", image)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)
# Create a test image with distinct edges for demonstration
print("\nCreating test image with distinct edges...")
test_edges = np.zeros((300, 500), dtype="uint8")
# Draw various shapes with different edge types
cv2.rectangle(test_edges, (20, 20), (120, 280), 255, -1) # Sharp rectangle
cv2.circle(test_edges, (250, 150), 80, 255, -1) # Circle
cv2.ellipse(test_edges, (400, 150), (60, 90), 30, 0, 360, 255, -1) # Ellipse
cv2.line(test_edges, (0, 0), (500, 300), 255, 10) # Diagonal line
# Add gradient edges
for x in range(200, 300):
    intensity = int(255 * (x - 150) / 150)
    test_edges[150:300, x] = np.clip(intensity, 0, 255)
cv2.imshow("Test Image (Distinct Edges)", test_edges)
cv2.waitKey(0)
print("\nImage features: rectangle (sharp edges), circle (curved edges),")
print("ellipse (angled edges), diagonal line, gradient edge")
# -----------------------------------------------------------------
# SECTION 2: THE LAPLACIAN OPERATOR
# -----------------------------------------------------------------
print("\n[Section 2] Laplacian Operator - Second Derivative")
print("-" * 50)
print("The Laplacian computes the second derivative of the image.")
print("It highlights regions of rapid intensity change.")
# Important: Use floating point to capture negative values!
print("\nImportant: Using float64 to capture both positive and negative gradients.")
# Apply Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
print(f"Laplacian output min: {laplacian.min()}, max: {laplacian.max()}")
# Convert to absolute values for display (edges are intensity changes)
laplacian_abs = np.uint8(np.absolute(laplacian))
cv2.imshow("Original Grayscale", gray)
cv2.imshow("Laplacian (Absolute)", laplacian_abs)
cv2.waitKey(0)
# Test Laplacian on the test image
laplacian_test = cv2.Laplacian(test_edges, cv2.CV_64F)
laplacian_test_abs = np.uint8(np.absolute(laplacian_test))
cv2.imshow("Test Image", test_edges)
cv2.imshow("Laplacian on Test Image", laplacian_test_abs)
cv2.waitKey(0)
print("Observation: Laplacian finds edges in ALL directions.")
print(" - Sharp edges = bright lines in output")
print(" - Gradual edges = dimmer lines")
print(" - Flat areas = black")
# Demonstrate the importance of floating point
print("\n--- Why float64 matters ---")
# Using uint8 (incorrect)
laplacian_uint8 = cv2.Laplacian(gray, cv2.CV_8U)
cv2.imshow("Laplacian with uint8 (missing half the edges!)",laplacian_uint8)
cv2.waitKey(0)
print("With uint8, negative gradients become 0 → MISSING EDGES!")
print("Always use float64 (CV_64F) for gradient computation!")
# -----------------------------------------------------------------
# SECTION 3: SOBEL OPERATOR - DIRECTIONAL EDGES
# -----------------------------------------------------------------
print("\n[Section 3] Sobel Operator - Directional Gradients")
print("-" * 50)
print("The Sobel operator computes gradients in specific directions.")
print(" - dx = 1, dy = 0: Horizontal edges (vertical changes)")
print(" - dx = 0, dy = 1: Vertical edges (horizontal changes)")
# Compute Sobel gradients
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # x-direction (vertical edges)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1) # y-direction (horizontal edges)
print(f"Sobel X min: {sobel_x.min():.2f}, max: {sobel_x.max():.2f}")
print(f"Sobel Y min: {sobel_y.min():.2f}, max: {sobel_y.max():.2f}")
# Convert to absolute values for display
sobel_x_abs = np.uint8(np.absolute(sobel_x))
sobel_y_abs = np.uint8(np.absolute(sobel_y))
cv2.imshow("Original Grayscale", gray)
cv2.imshow("Sobel X (Vertical Edges)", sobel_x_abs)
cv2.imshow("Sobel Y (Horizontal Edges)", sobel_y_abs)
cv2.waitKey(0)
# Test on test image
sobel_x_test = cv2.Sobel(test_edges, cv2.CV_64F, 1, 0)
sobel_y_test = cv2.Sobel(test_edges, cv2.CV_64F, 0, 1)
sobel_x_test_abs = np.uint8(np.absolute(sobel_x_test))
sobel_y_test_abs = np.uint8(np.absolute(sobel_y_test))
cv2.imshow("Test Image", test_edges)
cv2.imshow("Sobel X on Test (Vertical edges)", sobel_x_test_abs)
cv2.imshow("Sobel Y on Test (Horizontal edges)", sobel_y_test_abs)
cv2.waitKey(0)
print("\nObservation:")
print(" - Sobel X finds VERTICAL edges (top/bottom of rectangle)")
print(" - Sobel Y finds HORIZONTAL edges (left/right of rectangle)")
# Combine Sobel gradients
print("\n--- Combining Sobel X and Y ---")
# Method 1: Add (simple but can cause saturation)
sobel_combined_add = cv2.add(sobel_x_abs, sobel_y_abs)
cv2.imshow("Combined: Addition", sobel_combined_add)
cv2.waitKey(0)
# Method 2: Bitwise OR (recommended)
sobel_combined_or = cv2.bitwise_or(sobel_x_abs, sobel_y_abs)
cv2.imshow("Combined: Bitwise OR", sobel_combined_or)
cv2.waitKey(0)
# Method 3: Calculate magnitude (more accurate)
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_magnitude = np.uint8(np.clip(magnitude, 0, 255))
cv2.imshow("Combined: Magnitude", sobel_magnitude)
cv2.waitKey(0)
print("All methods work. Bitwise OR is often used for simplicity.")
# -----------------------------------------------------------------
# SECTION 4: CANNY EDGE DETECTOR - THE GOLD STANDARD
# -----------------------------------------------------------------
print("\n[Section 4] Canny Edge Detector - Gold Standard")
print("-" * 50)
print("Canny is a multi-stage algorithm:")
print(" 1. Gaussian blur to reduce noise")
print(" 2. Compute gradients (Sobel)")
print(" 3. Non-maximum suppression (thin edges)")
print(" 4. Hysteresis thresholding (track edges)")
# Apply Canny with different parameters
print("\n--- Finding optimal Canny parameters ---")
# Try different thresholds
threshold_pairs = [
 (30, 90), # Low, high
 (50, 100), 
 (50, 150),
 (100, 200),
 (150, 250)
]
print("Testing different threshold pairs (low, high):")
for low, high in threshold_pairs:
    edges = cv2.Canny(gray, low, high)
    cv2.putText(edges, f"T1={low}, T2={high}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow(f"Canny - T1={low}, T2={high}", edges)
    cv2.waitKey(0)
print("\nObservation:")
print(" - Lower thresholds = more edges (including noise)")
print(" - Higher thresholds = fewer edges (only strong edges)")
print(" - Ratio of high/low = 2:1 is a good starting point")
# Demonstrate the importance of blurring
print("\n--- Why blur before Canny? ---")
# Add noise to image
noise = np.random.normal(0, 25, gray.shape).astype("uint8")
noisy_gray = cv2.add(gray, noise)
# Canny without blur
edges_noisy = cv2.Canny(noisy_gray, 50, 150)
# Canny with blur
blurred = cv2.GaussianBlur(noisy_gray, (5, 5), 0)
edges_blurred = cv2.Canny(blurred, 50, 150)
comparison = np.hstack([
 cv2.resize(noisy_gray, (200, 150)),
 cv2.resize(edges_noisy, (200, 150)),
 cv2.resize(edges_blurred, (200, 150))
])
cv2.putText(comparison, "Noisy Image", (40, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
cv2.putText(comparison, "Canny (no blur)", (220, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
cv2.putText(comparison, "Canny (with blur)", (420, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
cv2.imshow("Blurring Before Canny - Critical!", comparison)
cv2.waitKey(0)
print("CONCLUSION: ALWAYS BLUR BEFORE CANNY!")
print(" - Without blur: many false edges from noise")
print(" - With blur: clean, meaningful edges")
# Test Canny on test image
print("\n--- Canny on Test Image ---")
canny_test = cv2.Canny(test_edges, 30, 90)
cv2.imshow("Test Image", test_edges)
cv2.imshow("Canny on Test Image", canny_test)
cv2.waitKey(0)
print("Canny produces clean, thin, continuous edges!")
# -----------------------------------------------------------------
# SECTION 5: CANNY PARAMETER EXPLORER
# -----------------------------------------------------------------
print("\n[Section 5] Interactive Canny Parameter Explorer")
print("-" * 50)
def interactive_canny(image_gray):
    """Interactive tool to find optimal Canny parameters"""
 # Apply initial blur (always recommended)
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
 
    cv2.namedWindow("Canny Explorer")
 
 # Create trackbars
    cv2.createTrackbar("Low Threshold", "Canny Explorer", 50, 255, lambda x: None)
    cv2.createTrackbar("High Threshold", "Canny Explorer", 150, 255, lambda x: None)
    cv2.createTrackbar("Blur Size", "Canny Explorer", 5, 21, lambda x: None)
 
    print("\nCanny Explorer Controls:")
    print(" - Low Threshold: edges below this are ignored")
    print(" - High Threshold: edges above this are definitely edges")
    print(" - Blur Size: noise reduction (must be odd)")
    print(" - Press ESC to exit")
    print("\nRule of thumb: High = 2× to 3× Low")
 
    while True:
        low = cv2.getTrackbarPos("Low Threshold", "Canny Explorer")
        high = cv2.getTrackbarPos("High Threshold", "Canny Explorer")
        blur_size = cv2.getTrackbarPos("Blur Size", "Canny Explorer")
 
 # Ensure odd number
        if blur_size % 2 == 0:
            blur_size += 1
 
 # Apply blur
        blurred = cv2.GaussianBlur(image_gray, (blur_size, blur_size), 0)
 
 # Apply Canny
        edges = cv2.Canny(blurred, low, high)
 
 # Show results
        display = np.hstack([
        cv2.resize(image_gray, (300, 250)),
        cv2.resize(edges, (300, 250))
        ])
 
 # Add info
        cv2.putText(display, f"Low={low}, High={high}, Blur={blur_size}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
 
        cv2.imshow("Canny Explorer", display)
 
        if cv2.waitKey(30) & 0xFF == 27: # ESC
            break
 
    cv2.destroyAllWindows()
# Run the interactive explorer
interactive_canny(gray)
# -----------------------------------------------------------------
# SECTION 6: PRACTICAL APPLICATION - EDGE-BASED OBJECT DETECTION
# -----------------------------------------------------------------
print("\n[Section 6] Practical Application: Edge Detection Pipeline")
print("-" * 50)
def edge_detection_pipeline(image, blur_size=5, low_thresh=50, high_thresh=150):
    """
    Complete edge detection pipeline
    """
 # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
 # Step 2: Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
 
 # Step 3: Apply Canny
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
 
 # Step 4: Dilate edges to make them more visible (optional)
    kernel = np.ones((2, 2), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
 
    return edges, edges_dilated
# Apply pipeline
edges, edges_dilated = edge_detection_pipeline(image, 5, 50, 150)
# Create visualization
visualization = np.hstack([
 cv2.resize(image, (250, 200)),
 cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (250, 200)),
 cv2.resize(cv2.cvtColor(edges_dilated, cv2.COLOR_GRAY2BGR), (250,
200))
])
cv2.putText(visualization, "Original", (80, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(visualization, "Canny Edges", (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(visualization, "Dilated Edges", (530, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.imshow("Edge Detection Pipeline", visualization)
cv2.waitKey(0)
print("\nEdge detection pipeline complete!")
print(" - Dilated edges are thicker and easier to see")
print(" - Use for visualization or contour detection prep")
# -----------------------------------------------------------------
# SECTION 7: COMPARING ALL METHODS
# -----------------------------------------------------------------
print("\n[Section 7] Comparing All Edge Detection Methods")
print("-" * 50)
# Apply all methods to the same image
methods = {
 "Laplacian": lambda: np.uint8(np.absolute(cv2.Laplacian(gray,cv2.CV_64F))),
 "Sobel X": lambda: np.uint8(np.absolute(cv2.Sobel(gray, cv2.CV_64F,1, 0))),
 "Sobel Y": lambda: np.uint8(np.absolute(cv2.Sobel(gray, cv2.CV_64F,0, 1))),
 "Sobel Combined": lambda: cv2.bitwise_or(
 np.uint8(np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))),
 np.uint8(np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1)))
 ),
 "Canny (50,150)": lambda: cv2.Canny(gray, 50, 150),
 "Canny + Blur": lambda: cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0),50, 150)
}
# Create comparison grid
grid_images = []
current_row = []
grid_images.append(cv2.resize(gray, (200, 150))) # Original first
for name, func in methods.items():
    result = func()
    if len(result.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result_resized = cv2.resize(result, (200, 150))
    cv2.putText(result_resized, name[:12], (5, 20),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    grid_images.append(result_resized)
# Create 3x3 grid
grid = np.vstack([
 np.hstack(grid_images[0:3]),
 np.hstack(grid_images[3:6]),
 np.hstack(grid_images[6:9])
])
cv2.imshow("ALL EDGE DETECTION METHODS COMPARED", grid)
cv2.waitKey(0)
print("\nComparison Summary:")
print(" - Laplacian: Fast, but noisy")
print(" - Sobel: Directional, good for specific edges")
print(" - Canny: Best quality, requires parameter tuning")
print(" - Canny + Blur: Cleanest results (recommended for most applications)")
# -----------------------------------------------------------------
# SECTION 8: CREATE EDGE DETECTION REFERENCE GUIDE
# -----------------------------------------------------------------
print("\n[Section 8] Creating Edge Detection Reference Guide")
reference = np.zeros((600, 800, 3), dtype="uint8")
ref_height, ref_width = reference.shape[:2]
# Title
cv2.putText(reference, "EDGE DETECTION REFERENCE GUIDE", (ref_width//2 - 230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
# Use test image for demonstration
demo_img = cv2.resize(test_edges, (150, 100))
laplacian_demo = cv2.resize(np.uint8(np.absolute(cv2.Laplacian(test_edges, cv2.CV_64F))),(150, 100))
sobel_x_demo = cv2.resize(np.uint8(np.absolute(cv2.Sobel(test_edges, cv2.CV_64F, 1, 0))), (150, 100))
sobel_y_demo = cv2.resize(np.uint8(np.absolute(cv2.Sobel(test_edges, cv2.CV_64F, 0, 1))), (150, 100))
canny_demo = cv2.resize(cv2.Canny(test_edges, 30, 90), (150, 100))
# Place images in reference
reference[60:160, 30:180] = cv2.cvtColor(demo_img, cv2.COLOR_GRAY2BGR)
reference[60:160, 210:360] = cv2.cvtColor(laplacian_demo,cv2.COLOR_GRAY2BGR)
reference[60:160, 390:540] = cv2.cvtColor(sobel_x_demo,cv2.COLOR_GRAY2BGR)
reference[60:160, 570:720] = cv2.cvtColor(sobel_y_demo,cv2.COLOR_GRAY2BGR)
reference[180:280, 30:180] = cv2.cvtColor(canny_demo, cv2.COLOR_GRAY2BGR)
# Labels
cv2.putText(reference, "Original", (70, 180), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
cv2.putText(reference, "Laplacian", (260, 180), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
cv2.putText(reference, "Sobel X", (440, 180), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
cv2.putText(reference, "Sobel Y", (620, 180), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
cv2.putText(reference, "Canny", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 1)
# Method descriptions
cv2.putText(reference, "METHOD COMPARISON:", (30, 340),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
descriptions = [
 ("Laplacian: 2nd derivative, all directions", 30, 370),
 ("Sobel X: Horizontal edges (vertical changes)", 30, 400),
 ("Sobel Y: Vertical edges (horizontal changes)", 30, 430),
 ("Canny: Multi-stage, high quality", 30, 460),
]
for text, x, y in descriptions:
    cv2.putText(reference, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(200, 200, 200), 1)
# Canny parameters guide
cv2.putText(reference, "CANNY PARAMETER GUIDE:", (30, 510),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
cv2.putText(reference, "Low Threshold: 30-100 | High: 2× to 3× Low | Blur: 3,5,7,9", (30, 540),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(reference, "Lower thresholds = more edges (including noise)",(30, 565),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
cv2.putText(reference, "Higher thresholds = fewer edges (strong only)",(30, 585),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
cv2.imshow("EDGE DETECTION REFERENCE GUIDE", reference)
cv2.waitKey(0)
cv2.imwrite("edge_detection_reference.png", reference)
print("Saved: edge_detection_reference.png")
print("\n" + "=" * 70)
print("DAY 12 COMPLETE!")
print("=" * 70)
cv2.destroyAllWindows()