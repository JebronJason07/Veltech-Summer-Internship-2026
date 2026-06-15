
# day10_smoothing_blurring.py
# Complete guide to image smoothing and blurring
from __future__ import print_function
import numpy as np
import argparse
import cv2
print("=" * 70)
print("DAY 10: SMOOTHING AND BLURRING")
print("=" * 70)
# -----------------------------------------------------------------
# SECTION 1: LOAD IMAGE AND CREATE NOISY VERSIONS
# -----------------------------------------------------------------
print("\n[Section 1] Loading image and creating noisy versions...")
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="Path to the image")
args = vars(ap.parse_args())
# Load image
image = cv2.imread(args["image"])
if image is None:
    print("ERROR: Could not load image!")
    exit()
height, width = image.shape[:2]
print(f"Loaded: {width} x {height} pixels")
# Convert to grayscale for simpler demonstration
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Create noisy versions for testing
def add_salt_pepper_noise(img, prob=0.02):
    """Add salt-and-pepper noise to image"""
    output = img.copy()
    num_salt = np.ceil(prob * img.size * 0.5)
    num_pepper = np.ceil(prob * img.size * 0.5)
 
 # Add salt (white)
    coords = [np.random.randint(0, i, int(num_salt)) for i in img.shape]
    output[coords[0], coords[1]] = 255
 
 # Add pepper (black)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in img.shape]
    output[coords[0], coords[1]] = 0
    return output
def add_gaussian_noise(img, mean=0, sigma=25):
    """Add Gaussian noise to image"""
    noise = np.random.normal(mean, sigma, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype("uint8")
# Create noisy versions
noisy_saltpepper = add_salt_pepper_noise(gray, 0.03)
noisy_gaussian = add_gaussian_noise(gray, 0, 20)
cv2.imshow("Original (Grayscale)", gray)
cv2.imshow("Salt-and-Pepper Noise", noisy_saltpepper)
cv2.imshow("Gaussian Noise", noisy_gaussian)
cv2.waitKey(0)
print("Created noisy images for testing blurring methods.")
# -----------------------------------------------------------------
# SECTION 2: AVERAGING BLUR (MEAN FILTER)
# -----------------------------------------------------------------
print("\n[Section 2] Averaging Blur (Mean Filter)")
print("-" * 50)
print("How it works: Each pixel becomes the average of its neighbors.")
print("Formula: output = (1/k²) × sum of all pixels in k×k kernel")
print("Kernel size must be odd!")
# Apply averaging blur with different kernel sizes
kernel_sizes = [(3, 3), (5, 5), (7, 7), (11, 11)]
avg_results = []
print("\nApplying averaging blur with different kernel sizes:")
for ksize in kernel_sizes:
    blurred = cv2.blur(gray, ksize)
    avg_results.append(blurred)
    print(f" Kernel {ksize[0]}×{ksize[1]} applied")
 
 # Show result
    cv2.imshow(f"Averaging Blur - {ksize[0]}×{ksize[1]}", blurred)
    cv2.waitKey(0)
# Compare all averaging results
comparison_avg = np.hstack([
 cv2.resize(gray, (200, 150)),
 cv2.resize(avg_results[0], (200, 150)),
 cv2.resize(avg_results[1], (200, 150)),
 cv2.resize(avg_results[2], (200, 150))
])
cv2.imshow("Averaging Blur Comparison: Original | 3x3 | 5x5 | 7x7",comparison_avg)
cv2.waitKey(0)
# Test on noisy images
print("\n--- Testing Averaging Blur on Noisy Images ---")
avg_denoise_sp = cv2.blur(noisy_saltpepper, (5, 5))
avg_denoise_gauss = cv2.blur(noisy_gaussian, (5, 5))
cv2.imshow("Salt-Pepper + Averaging Blur", avg_denoise_sp)
cv2.imshow("Gaussian Noise + Averaging Blur", avg_denoise_gauss)
cv2.waitKey(0)
print("Observation: Averaging blur reduces noise but also blurs edges.")
print(" - Works okay for Gaussian noise")
print(" - Not great for salt-and-pepper noise (noise still visible)")
# -----------------------------------------------------------------
# SECTION 3: GAUSSIAN BLUR
# -----------------------------------------------------------------
print("\n[Section 3] Gaussian Blur")
print("-" * 50)
print("How it works: Weighted average where center pixels have more influence.")
print("Formula: Uses Gaussian distribution (bell curve) for weights.")
print("Result: More natural-looking blur than averaging.")
# Apply Gaussian blur
gaussian_results = []
kernel_sizes = [(3, 3), (5, 5), (7, 7), (11, 11)]
print("\nApplying Gaussian blur with different kernel sizes:")
for ksize in kernel_sizes:
    blurred = cv2.GaussianBlur(gray, ksize, 0) # Sigma=0 means auto￾compute
    gaussian_results.append(blurred)
    print(f" Kernel {ksize[0]}×{ksize[1]} applied")
 
    cv2.imshow(f"Gaussian Blur - {ksize[0]}×{ksize[1]}", blurred)
    cv2.waitKey(0)
# Compare Gaussian vs Averaging
print("\n--- Gaussian vs Averaging Comparison ---")
avg_7x7 = cv2.blur(gray, (7, 7))
gauss_7x7 = cv2.GaussianBlur(gray, (7, 7), 0)
comparison = np.hstack([
 cv2.resize(gray, (200, 150)),
 cv2.resize(avg_7x7, (200, 150)),
 cv2.resize(gauss_7x7, (200, 150))
])
cv2.imshow("Original | Averaging 7x7 | Gaussian 7x7", comparison)
cv2.waitKey(0)
print("Observation: Gaussian blur preserves edges better than averaging.")
print(" - More natural 'out of focus' look")
print(" - Better for lens blur simulation")
# Gaussian with different sigma values
print("\n--- Effect of Sigma (Standard Deviation) ---")
print("Sigma controls the 'spread' of the Gaussian curve.")
sigmas = [0.5, 1, 2, 5, 10]
for sigma in sigmas:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma) # Kernel auto from sigma
    cv2.putText(blurred, f"Sigma = {sigma}", (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(f"Gaussian Blur - Sigma={sigma}", blurred)
    cv2.waitKey(0)
print("Higher sigma = stronger blur")
# Test on Gaussian noise
print("\n--- Gaussian Blur on Gaussian Noise (Best Match!) ---")
gauss_denoise = cv2.GaussianBlur(noisy_gaussian, (5, 5), 0)
cv2.imshow("Gaussian Noise + Gaussian Blur", gauss_denoise)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 4: MEDIAN BLUR
# -----------------------------------------------------------------
print("\n[Section 4] Median Blur")
print("-" * 50)
print("How it works: Each pixel becomes the MEDIAN of its neighbors.")
print("Why median? Median ignores extreme values (perfect for salt-and￾pepper!)")
print("Example: [10, 20, 255, 30, 40] → median = 30 (ignores 255)")
# Apply median blur
median_results = []
kernel_sizes = [3, 5, 7, 11] # Note: just one number for median
print("\nApplying median blur with different kernel sizes:")
for ksize in kernel_sizes:
    blurred = cv2.medianBlur(gray, ksize)
    median_results.append(blurred)
    print(f" Kernel {ksize}×{ksize} applied")
 
    cv2.imshow(f"Median Blur - {ksize}×{ksize}", blurred)
    cv2.waitKey(0)
# Test on salt-and-pepper noise (where median excels!)
print("\n--- Median Blur on Salt-and-Pepper Noise (Perfect!) ---")
median_denoise = cv2.medianBlur(noisy_saltpepper, 5)
cv2.imshow("Salt-Pepper + Median Blur (Best for this noise!)",median_denoise)
cv2.waitKey(0)
# Compare all methods on salt-pepper noise
print("\n--- Comparison: All Methods on Salt-Pepper Noise ---")
avg_on_sp = cv2.blur(noisy_saltpepper, (5, 5))
gauss_on_sp = cv2.GaussianBlur(noisy_saltpepper, (5, 5), 0)
median_on_sp = cv2.medianBlur(noisy_saltpepper, 5)
comparison_sp = np.hstack([
 cv2.resize(noisy_saltpepper, (180, 120)),
 cv2.resize(avg_on_sp, (180, 120)),
 cv2.resize(gauss_on_sp, (180, 120)),
 cv2.resize(median_on_sp, (180, 120))
])
cv2.imshow("Salt-Pepper: Original | Averaging | Gaussian | MEDIAN (best)", comparison_sp)
cv2.waitKey(0)
print("\nCONCLUSION: Median blur is BEST for salt-and-pepper noise!")
print(" - Averaging: noise still visible")
print(" - Gaussian: noise still visible") 
print(" - Median: noise almost completely removed!")
# -----------------------------------------------------------------
# SECTION 5: BILATERAL FILTER (EDGE-PRESERVING BLUR)
# -----------------------------------------------------------------
print("\n[Section 5] Bilateral Filter (Edge-Preserving Blur)")
print("-" * 50)
print("How it works: Two Gaussian filters - one for space, one for intensity.")
print(" - Spatial Gaussian: nearby pixels matter")
print(" - Intensity Gaussian: similar-colored pixels matter")
print("Result: Blurs flat areas but keeps edges sharp!")
# Apply bilateral filter with different parameters
print("\nParameters:")
print(" d = diameter of pixel neighborhood")
print(" sigmaColor = standard deviation in color space (higher = more colors considered)")
print(" sigmaSpace = standard deviation in coordinate space (higher = farther pixels matter)")
# Create a test image with edges and flat areas
test_edges = np.zeros((300, 500, 3), dtype="uint8")
# Add a rectangle
cv2.rectangle(test_edges, (50, 50), (250, 250), (200, 100, 50), -1)
# Add a circle
cv2.circle(test_edges, (400, 150), 80, (50, 150, 200), -1)
# Add text
cv2.putText(test_edges, "Edge Test", (60, 280), cv2.FONT_HERSHEY_SIMPLEX,
1.5, (255, 255, 255), 3)
# Add some noise
noise = np.random.randint(0, 30, test_edges.shape, dtype="uint8")
test_edges_noisy = cv2.add(test_edges, noise)
cv2.imshow("Test Image with Edges + Noise", test_edges_noisy)
cv2.waitKey(0)
# Compare blurring methods
print("\n--- Comparison: Gaussian vs Bilateral on Edges ---")
gauss_test = cv2.GaussianBlur(test_edges_noisy, (15, 15), 0)
bilateral_test = cv2.bilateralFilter(test_edges_noisy, 15, 75, 75)
comparison_edges = np.hstack([
 cv2.resize(test_edges_noisy, (200, 150)),
 cv2.resize(gauss_test, (200, 150)),
 cv2.resize(bilateral_test, (200, 150))
])
cv2.imshow("Edges: Original Noisy | Gaussian (edges blurred) | Bilateral (edges sharp!)",comparison_edges)
cv2.waitKey(0)
print("Observation: Bilateral filter removes noise but keeps edges!")
print(" - Gaussian: edges become soft and blurred")
print(" - Bilateral: edges remain sharp while flat areas are smoothed")
# Apply bilateral to real image
print("\n--- Applying Bilateral Filter to Your Image ---")
bilateral_params = [
 (5, 50, 50, "Mild"),
 (9, 75, 75, "Medium"),
 (15, 100, 100, "Strong"),
 (21, 150, 150, "Very Strong")
]
for d, sc, ss, name in bilateral_params:
    bilateral = cv2.bilateralFilter(image, d, sc, ss)
    cv2.putText(bilateral, f"Bilateral: {name}", (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(f"Bilateral Filter - {name}", bilateral)
    cv2.waitKey(0)
# Compare all four methods side by side
print("\n--- All Four Methods Compared ---")
# Use a moderate kernel size for fair comparison
kernel = 7
avg_blur = cv2.blur(image, (kernel, kernel))
gauss_blur = cv2.GaussianBlur(image, (kernel, kernel), 0)
median_blur = cv2.medianBlur(image, kernel)
bilateral_blur = cv2.bilateralFilter(image, kernel, 75, 75)
# Resize for consistent display
h, w = image.shape[:2]
display_h = 200
avg_blur = cv2.resize(avg_blur, (display_h, display_h))
gauss_blur = cv2.resize(gauss_blur, (display_h, display_h))
median_blur = cv2.resize(median_blur, (display_h, display_h))
bilateral_blur = cv2.resize(bilateral_blur, (display_h, display_h))
original_resized = cv2.resize(image, (display_h, display_h))
# Create labels
for img, label in [(avg_blur, "Averaging"), (gauss_blur, "Gaussian"),(median_blur, "Median"), (bilateral_blur,"Bilateral")]:
    cv2.putText(img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255), 1)
    top_row = np.hstack([original_resized, avg_blur, gauss_blur])
    bottom_row = np.hstack([median_blur, bilateral_blur,np.zeros((display_h, display_h, 3),dtype="uint8")])
all_methods = np.vstack([top_row, bottom_row])
cv2.imshow("ALL METHODS COMPARED (k=7)", all_methods)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 6: PRACTICAL APPLICATIONS
# -----------------------------------------------------------------
print("\n[Section 6] Practical Applications")
print("-" * 50)
# Application 1: Preprocessing for Edge Detection
print("\nApplication 1: Blur before edge detection")
print(" - Blurring removes noise that would create false edges")
# Create edges on original vs blurred
edges_original = cv2.Canny(gray, 50, 150)
edges_blurred = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
comparison_edges = np.hstack([
 cv2.resize(gray, (250, 150)),
 cv2.resize(edges_original, (250, 150)),
 cv2.resize(edges_blurred, (250, 150))
])
cv2.imshow("Edge Detection: Original | No Blur (noisy edges) | With Blur (clean edges)",comparison_edges)
cv2.waitKey(0)
print(" - Pre-blurring removes small details and noise")
print(" - Results in cleaner, more meaningful edges")
# Application 2: Skin Smoothing (Bilateral Filter)
print("\nApplication 2: Skin smoothing for portraits")
print(" - Bilateral filter smooths skin while keeping eyes/mouth sharp")
# Simulate portrait smoothing (use bilateral)
portrait_smooth = cv2.bilateralFilter(image, 15, 80, 80)
portrait_smooth = cv2.bilateralFilter(portrait_smooth, 15, 80, 80) # Apply twice
skin_comparison = np.hstack([
 cv2.resize(image, (250, 200)),
 cv2.resize(portrait_smooth, (250, 200))
])
cv2.imshow("Portrait Smoothing: Original | Bilateral (skin smoothed)",skin_comparison)
cv2.waitKey(0)
# Application 3: Creating Motion Blur Effect
print("\nApplication 3: Creating artistic motion blur")
# Create motion blur kernel (horizontal)
kernel_motion = np.zeros((15, 15), dtype="float32")
kernel_motion[7, :] = 1.0 / 15 # Horizontal line
motion_blur = cv2.filter2D(image, -1, kernel_motion)
cv2.imshow("Motion Blur Effect (horizontal)", motion_blur)
cv2.waitKey(0)
# Vertical motion blur
kernel_motion_v = np.zeros((15, 15), dtype="float32")
kernel_motion_v[:, 7] = 1.0 / 15 # Vertical line
motion_blur_v = cv2.filter2D(image, -1, kernel_motion_v)
cv2.imshow("Motion Blur Effect (vertical)", motion_blur_v)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 7: BLUR DETECTION (FOCUS CHECK)
# -----------------------------------------------------------------
print("\n[Section 7] Blur Detection - Is my image blurry?")
print("-" * 50)
def is_blurry(image, threshold=100):
    """Detect if an image is blurry using variance of Laplacian"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance < threshold, variance
# Create a blurry version of the image
blurry_image = cv2.GaussianBlur(image, (25, 25), 0)
# Check focus
is_blur, variance = is_blurry(image)
is_blur_blurry, variance_blurry = is_blurry(blurry_image)
print(f"Original image variance: {variance:.2f}")
print(f" → {'BLURRY' if is_blur else 'SHARP'} (threshold=100)")
print(f"\nBlurry image variance: {variance_blurry:.2f}")
print(f" → {'BLURRY' if is_blur_blurry else 'SHARP'}")
# Visualize
focus_comparison = np.hstack([
 cv2.resize(image, (250, 200)),
 cv2.resize(blurry_image, (250, 200))
])
cv2.putText(focus_comparison, f"Sharp (var={variance:.0f})", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.putText(focus_comparison, f"Blurry (var={variance_blurry:.0f})",(260, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.imshow("Blur Detection: Sharp vs Blurry", focus_comparison)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 8: CREATE BLURRING REFERENCE GUIDE
# -----------------------------------------------------------------
print("\n[Section 8] Creating Blurring Reference Guide")
reference = np.zeros((600, 800, 3), dtype="uint8")
ref_height, ref_width = reference.shape[:2]
# Title
cv2.putText(reference, "BLURRING METHODS REFERENCE GUIDE", (ref_width//2- 230, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
# Create small versions for display
small_original = cv2.resize(gray, (120, 90))
small_avg = cv2.resize(avg_blur, (120, 90))
small_gauss = cv2.resize(gauss_blur, (120, 90))
small_median = cv2.resize(median_blur, (120, 90))
small_bilateral = cv2.resize(bilateral_blur, (120, 90))
# Row 1: Method name and image
reference[60:150, 50:170] = cv2.cvtColor(small_original,
cv2.COLOR_GRAY2BGR)
reference[60:150, 190:310] = cv2.cvtColor(small_avg, cv2.COLOR_GRAY2BGR)
reference[60:150, 330:450] = cv2.cvtColor(small_gauss,
cv2.COLOR_GRAY2BGR)
reference[60:150, 470:590] = cv2.cvtColor(small_median,
cv2.COLOR_GRAY2BGR)
reference[60:150, 610:730] = small_bilateral
# Labels
cv2.putText(reference, "Original", (85, 170), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
cv2.putText(reference, "Averaging", (220, 170), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
cv2.putText(reference, "Gaussian", (365, 170), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
cv2.putText(reference, "Median", (515, 170), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
cv2.putText(reference, "Bilateral", (645, 170), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
# Row 2: Descriptions
y_start = 200
cv2.putText(reference, "DESCRIPTION:", (50, y_start + 20),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
descriptions = [
 ("Mean of all pixels", 50, y_start + 50),
 ("Weighted average", 190, y_start + 50),
 ("Median value", 330, y_start + 50),
 ("Edge-preserving", 470, y_start + 50),
 ("Spatial + intensity", 610, y_start + 50),
]
for text, x, y in descriptions:
    cv2.putText(reference, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(200, 200, 200), 1)
# Row 3: Best use cases
cv2.putText(reference, "BEST FOR:", (50, y_start + 90),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
use_cases = [
 ("General noise", 50, y_start + 120),
 ("Natural blur", 190, y_start + 120),
 ("Salt-pepper", 330, y_start + 120),
 ("Edge preservation", 470, y_start + 120),
 ("Skin smoothing", 610, y_start + 120),
]
for text, x, y in use_cases:
    cv2.putText(reference, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(200, 200, 200), 1)
# Row 4: Kernel formula examples
y_formula = 380
cv2.putText(reference, "KERNEL EXAMPLE (3x3):", (50, y_formula),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
# Draw kernel visualizations
kernels = [
 [[1,1,1],[1,1,1],[1,1,1]],
 [[1,2,1],[2,4,2],[1,2,1]],
 [[1,1,1],[1,1,1],[1,1,1]], # Median (concept)
 [[1,1,1],[1,1,1],[1,1,1]], # Bilateral (concept)
]
kernel_names = ["All 1/9", "Weighted", "Median", "2 Gaussians"]
for i, (kernel, name) in enumerate(zip(kernels, kernel_names)):
    x_off = 50 + i * 180

    for row in range(3):
        for col in range(3):

            if i == 1:  # Gaussian weights
                val = kernel[row][col]

                cv2.rectangle(
                    reference,
                    (x_off + col * 18, y_formula + 30 + row * 18),
                    (x_off + (col + 1) * 18, y_formula + 48 + row * 18),
                    (100, 100, 100),
                    -1
                )

                cv2.putText(
                    reference,
                    str(val),
                    (x_off + col * 18 + 4, y_formula + 47 + row * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

            else:
                color = (100, 100, 255) if i == 0 else ((100, 255, 100) if i == 2 else (255, 100, 100))

                cv2.rectangle(
                    reference,
                    (x_off + col * 18, y_formula + 30 + row * 18),
                    (x_off + (col + 1) * 18, y_formula + 48 + row * 18),
                    color,
                    -1
                )

    cv2.putText(
        reference,
        name,
        (x_off + 20, y_formula + 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )

# Footer: Key takeaways
cv2.putText(reference, "KEY TAKEAWAYS:", (50, 500),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

cv2.putText(reference, "1. Kernel size = larger = stronger blur", (50, 525),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

cv2.putText(reference, "2. Median is best for salt-and-pepper noise", (50, 550),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

cv2.putText(reference, "3. Bilateral preserves edges while removing noise", (50, 575),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

cv2.imshow("BLURRING REFERENCE GUIDE", reference)
cv2.waitKey(0)
cv2.imwrite("blurring_reference.png", reference)

print("Saved: blurring_reference.png")
print("\n" + "=" * 70)
print("DAY 10 COMPLETE!")
print("=" * 70)

cv2.destroyAllWindows()