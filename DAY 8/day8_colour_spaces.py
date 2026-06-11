# day8_color_spaces.py
# Complete guide to color spaces and channels
from __future__ import print_function
import numpy as np
import argparse
import cv2
print("=" * 70)
print("DAY 8: COLOR SPACES & CHANNELS")
print("=" * 70)
# -----------------------------------------------------------------
# SECTION 1: LOAD IMAGE
# -----------------------------------------------------------------
print("\n[Section 1] Loading image...")
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="Path to the image")
args = vars(ap.parse_args())
# Load image in BGR (OpenCV default)
image_bgr = cv2.imread(args["image"])
if image_bgr is None:
    print("ERROR: Could not load image!")
    exit()
height, width = image_bgr.shape[:2]
print(f"Loaded image: {width} x {height} pixels")
print(f"Default color space: BGR")
cv2.imshow("Original (BGR)", image_bgr)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 2: COLOR SPACE CONVERSIONS
# -----------------------------------------------------------------
print("\n[Section 2] Converting between color spaces...")
print("-" * 50)
# 2a: BGR to RGB
print("\n--- BGR to RGB ---")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
cv2.imshow("BGR (OpenCV default)", image_bgr)
cv2.imshow("RGB (Human perception)", image_rgb)
cv2.waitKey(0)
print("Note: BGR and RGB look different! Red and blue channels are swapped.")
# 2b: BGR to Grayscale
print("\n--- BGR to Grayscale ---")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", image_gray)
cv2.waitKey(0)
print(f"Grayscale shape: {image_gray.shape} (single channel)")
# 2c: BGR to HSV
print("\n--- BGR to HSV ---")
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV (Hue-Saturation-Value)", image_hsv)
cv2.waitKey(0)
print(f"HSV shape: {image_hsv.shape} (still 3 channels, but different meaning)")
# 2d: BGR to LAB
print("\n--- BGR to LAB ---")
image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
cv2.imshow("LAB (L*a*b*)", image_lab)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 3: UNDERSTANDING GRAYSCALE
# -----------------------------------------------------------------
print("\n[Section 3] Understanding Grayscale conversion...")
print("-" * 50)
# Different ways to convert to grayscale
print("Method 1: cv2.cvtColor (standard, uses weighted formula)")
gray_standard = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
print("Method 2: Average of BGR channels")
gray_avg = np.mean(image_bgr, axis=2).astype("uint8")
print("Method 3: Take only the Green channel (eye most sensitive to green)")
gray_green = image_bgr[:, :, 1] # Green channel
# Show all methods
cv2.imshow("Grayscale - Standard (weighted)", gray_standard)
cv2.imshow("Grayscale - Average of BGR", gray_avg)
cv2.imshow("Grayscale - Green Channel Only", gray_green)
cv2.waitKey(0)
# Compare differences
print("\nComparison:")
print("- Standard formula gives best perceptual results")
print("- Average method can lose contrast")
print("- Single channel method discards color info")
# Create side-by-side comparison
top_row = np.hstack([
 cv2.resize(gray_standard, (200, 150)),
 cv2.resize(gray_avg, (200, 150))
])
bottom_row = np.hstack([
 cv2.resize(gray_green, (200, 150)),
 np.zeros((150, 200), dtype="uint8")
])
comparison = np.vstack([top_row, bottom_row])
cv2.imshow("Grayscale Methods: Standard | Average | Green Only",
comparison)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 4: SPLITTING CHANNELS
# -----------------------------------------------------------------
print("\n[Section 4] Splitting images into channels...")
print("-" * 50)
# Method 1: Using indexing (fastest)
print("Method 1: NumPy indexing")
blue_channel = image_bgr[:, :, 0] # Blue
green_channel = image_bgr[:, :, 1] # Green
red_channel = image_bgr[:, :, 2] # Red
# Method 2: Using cv2.split()
print("Method 2: cv2.split() function")
b, g, r = cv2.split(image_bgr)
# Verify both methods produce same result
print(f"Indexing vs split - Blue matches: {np.array_equal(blue_channel,b)}")
print(f"Indexing vs split - Green matches: {np.array_equal(green_channel,g)}")
print(f"Indexing vs split - Red matches: {np.array_equal(red_channel,r)}")
# Display individual channels (as grayscale)
cv2.imshow("Blue Channel (0 = no blue, 255 = max blue)", blue_channel)
cv2.imshow("Green Channel", green_channel)
cv2.imshow("Red Channel", red_channel)
cv2.waitKey(0)
print("\nInterpretation:")
print("- Bright areas in Blue channel = lots of blue in original")
print("- Dark areas = little of that color")
# -----------------------------------------------------------------
# SECTION 5: VISUALIZING CHANNELS IN COLOR
# -----------------------------------------------------------------
print("\n[Section 5] Visualizing channels in color...")
print("-" * 50)
# Create blank (zero) arrays
zeros = np.zeros(image_bgr.shape[:2], dtype="uint8")
# Create color visualizations (showing only one channel at a time)
blue_vis = cv2.merge([blue_channel, zeros, zeros]) # Only Blue
green_vis = cv2.merge([zeros, green_channel, zeros]) # Only Green
red_vis = cv2.merge([zeros, zeros, red_channel]) # Only Red
cv2.imshow("Blue Channel Visualized (Blue only)", blue_vis)
cv2.imshow("Green Channel Visualized (Green only)", green_vis)
cv2.imshow("Red Channel Visualized (Red only)", red_vis)
cv2.waitKey(0)
# Create channel intensity maps (heatmap style)
print("\n--- Channel Intensity Analysis ---")
def analyze_channel(channel, name):
    min_val = np.min(channel)
    max_val = np.max(channel)
    mean_val = np.mean(channel)
    print(f"{name}: min={min_val}, max={max_val}, mean={mean_val:.1f}")
analyze_channel(blue_channel, "Blue Channel")
analyze_channel(green_channel, "Green Channel")
analyze_channel(red_channel, "Red Channel")
# -----------------------------------------------------------------
# SECTION 6: HSV COLOR SPACE DEEP DIVE
# -----------------------------------------------------------------
print("\n[Section 6] HSV Color Space Deep Dive")
print("-" * 50)
# Split HSV channels
h, s, v = cv2.split(image_hsv)
cv2.imshow("Hue Channel (color information)", h)
cv2.imshow("Saturation Channel (color purity)", s)
cv2.imshow("Value Channel (brightness)", v)
cv2.waitKey(0)
print("\nHSV Channel Explanations:")
print("-" * 40)
print("HUE (0-179): Represents the color type")
print(" - 0 = Red, 30 = Orange, 60 = Yellow")
print(" - 90 = Green, 120 = Cyan, 150 = Blue")
print(" - Uniform areas = same color regardless of lighting")
print()
print("SATURATION (0-255): Color purity")
print(" - Low (0-50) = grayscale, washed out")
print(" - High (200-255) = vivid, pure colors")
print()
print("VALUE (0-255): Brightness")
print(" - Low = dark, High = bright")
print(" - Similar to grayscale")
# Demonstrate HSV invariance to lighting (create test)
print("\n--- HSV Invariance Demonstration ---")
# Create bright and dark versions of same color
bright_red = np.zeros((100, 100, 3), dtype="uint8")
bright_red[:, :] = (0, 0, 255) # BGR red
dark_red = np.zeros((100, 100, 3), dtype="uint8")
dark_red[:, :] = (0, 0, 100) # Darker red
# Convert to HSV
bright_hsv = cv2.cvtColor(bright_red, cv2.COLOR_BGR2HSV)
dark_hsv = cv2.cvtColor(dark_red, cv2.COLOR_BGR2HSV)
print(f"Bright red BGR: (0,0,255) → HSV Hue: {bright_hsv[0,0,0]}")
print(f"Dark red BGR: (0,0,100) → HSV Hue: {dark_hsv[0,0,0]}")
print("Notice: Hue values are similar! This is why HSV is great for color detection.")
# -----------------------------------------------------------------
# SECTION 7: PRACTICAL COLOR DETECTION
# -----------------------------------------------------------------
print("\n[Section 7] Practical: Color Detection with HSV")
print("-" * 50)
# Create a test image with different colors
test_colors = np.zeros((200, 600, 3), dtype="uint8")
# Draw colored rectangles
colors_bgr = [
 ((0, 0, 255), "RED"),
 ((0, 255, 0), "GREEN"),
 ((255, 0, 0), "BLUE"),
 ((0, 255, 255), "YELLOW"),
 ((255, 0, 255), "MAGENTA"),
 ((255, 255, 0), "CYAN")
]
for i, (color, name) in enumerate(colors_bgr):
    x_start = i * 100
    x_end = (i + 1) * 100
    test_colors[50:150, x_start:x_end] = color
    cv2.putText(test_colors, name, (x_start + 10, 180),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.imshow("Test Colors (BGR)", test_colors)
cv2.waitKey(0)
# Convert to HSV
test_hsv = cv2.cvtColor(test_colors, cv2.COLOR_BGR2HSV)
# Define color ranges in HSV
# Red has two ranges (wraps around at 0)
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([179, 255, 255])
green_lower = np.array([40, 50, 50])
green_upper = np.array([80, 255, 255])
blue_lower = np.array([100, 50, 50])
blue_upper = np.array([130, 255, 255])
# Detect blue regions
blue_mask = cv2.inRange(test_hsv, blue_lower, blue_upper)
blue_detected = cv2.bitwise_and(test_colors, test_colors, mask=blue_mask)
cv2.imshow("Blue Color Detection (mask)", blue_mask)
cv2.imshow("Blue Pixels Only", blue_detected)
cv2.waitKey(0)
print("Color detection with HSV is powerful for tracking colored objects!")
# -----------------------------------------------------------------
# SECTION 8: MERGING CHANNELS
# -----------------------------------------------------------------
print("\n[Section 8] Merging channels back together...")
print("-" * 50)
# Method 1: Using cv2.merge()
print("Method 1: cv2.merge()")
merged_back = cv2.merge([blue_channel, green_channel, red_channel])
cv2.imshow("Merged back (should match original)", merged_back)
cv2.waitKey(0)
# Method 2: Manual merging
print("Method 2: Manual merging with NumPy")
manual_merge = np.zeros(image_bgr.shape, dtype="uint8")
manual_merge[:, :, 0] = blue_channel
manual_merge[:, :, 1] = green_channel
manual_merge[:, :, 2] = red_channel
# Verify all methods match
print(f"cv2.merge equals original: {np.array_equal(merged_back,image_bgr)}")
print(f"Manual merge equals original: {np.array_equal(manual_merge,image_bgr)}")
# Create channel-swapped versions (creative effects)
print("\n--- Creative Channel Swapping ---")
# Swap red and blue
bgr_to_rgb_swap = cv2.merge([red_channel, green_channel, blue_channel])
cv2.imshow("Channel Swap: Red and Blue swapped", bgr_to_rgb_swap)
cv2.waitKey(0)
# Use only green and red (zero out blue)
no_blue = cv2.merge([zeros, green_channel, red_channel])
cv2.imshow("Blue removed (only Green and Red)", no_blue)
cv2.waitKey(0)
# Use only blue channel in all positions (monochrome blue effect)
blue_only = cv2.merge([blue_channel, blue_channel, blue_channel])
cv2.imshow("Blue channel in all positions", blue_only)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 9: BGR VS RGB - THE IMPORTANT DIFFERENCE
# -----------------------------------------------------------------
print("\n[Section 9] BGR vs RGB - Critical Differences")
print("-" * 50)
# Create a test pattern to see the difference
test = np.zeros((200, 300, 3), dtype="uint8")
# Left: Pure Red in BGR (OpenCV)
test[50:150, 50:100] = (0, 0, 255) # BGR red
cv2.putText(test, "BGR: (0,0,255)", (50, 180), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
# Middle: Pure Green in BGR
test[50:150, 125:175] = (0, 255, 0)
cv2.putText(test, "BGR: (0,255,0)", (125, 180), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
# Right: Pure Blue in BGR
test[50:150, 200:250] = (255, 0, 0)
cv2.putText(test, "BGR: (255,0,0)", (200, 180), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
# Convert to RGB for display in matplotlib (if we had it)
test_rgb = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
cv2.imshow("BGR Color Test (OpenCV)", test)
cv2.waitKey(0)
print("\nSUMMARY: When using OpenCV, remember:")
print(" - Image[:,:,0] = BLUE channel")
print(" - Image[:,:,1] = GREEN channel")
print(" - Image[:,:,2] = RED channel")
print(" - To display with non-OpenCV tools, convert BGR→RGB first!")
# -----------------------------------------------------------------
# SECTION 10: CREATE A COLOR SPACE REFERENCE GUIDE
# -----------------------------------------------------------------
print("\n[Section 10] Creating color space reference guide...")
# Create a reference canvas
ref_width = 800
ref_height = 500
reference = np.zeros((ref_height, ref_width, 3), dtype="uint8")
# Title
cv2.putText(reference, "COLOR SPACE REFERENCE GUIDE", (ref_width//2 -200, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# Section 1: Original
orig_small = cv2.resize(image_bgr, (150, 100))
reference[70:170, 20:170] = orig_small
cv2.putText(reference, "Original (BGR)", (50, 190),
cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
# Section 2: RGB
rgb_small = cv2.resize(image_rgb, (150, 100))
reference[70:170, 190:340] = rgb_small
cv2.putText(reference, "RGB", (230, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 1)
# Section 3: Grayscale
gray_small = cv2.resize(cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR),(150, 100))
reference[70:170, 360:510] = gray_small
cv2.putText(reference, "Grayscale", (400, 190), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
# Section 4: HSV
hsv_small = cv2.resize(image_hsv, (150, 100))
reference[70:170, 530:680] = hsv_small
cv2.putText(reference, "HSV", (580, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 1)
# Channel visualizations
channels_y = 240
cv2.putText(reference, "CHANNELS:", (20, channels_y + 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
# Blue channel
blue_small = cv2.resize(cv2.cvtColor(blue_channel, cv2.COLOR_GRAY2BGR),(100, 80))
reference[channels_y + 30:channels_y + 110, 20:120] = blue_small
cv2.putText(reference, "BLUE", (40, channels_y + 125),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
# Green channel
green_small = cv2.resize(cv2.cvtColor(green_channel, cv2.COLOR_GRAY2BGR),(100, 80))
reference[channels_y + 30:channels_y + 110, 140:240] = green_small
cv2.putText(reference, "GREEN", (165, channels_y + 125),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
# Red channel
red_small = cv2.resize(cv2.cvtColor(red_channel, cv2.COLOR_GRAY2BGR),(100, 80))
reference[channels_y + 30:channels_y + 110, 260:360] = red_small
cv2.putText(reference, "RED", (295, channels_y + 125),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
# HSV channels
h_small = cv2.resize(cv2.cvtColor(h, cv2.COLOR_GRAY2BGR), (100, 80))
reference[channels_y + 30:channels_y + 110, 450:550] = h_small
cv2.putText(reference, "HUE", (480, channels_y + 125),
cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
s_small = cv2.resize(cv2.cvtColor(s, cv2.COLOR_GRAY2BGR), (100, 80))
reference[channels_y + 30:channels_y + 110, 570:670] = s_small
cv2.putText(reference, "SAT", (605, channels_y + 125),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
v_small = cv2.resize(cv2.cvtColor(v, cv2.COLOR_GRAY2BGR), (100, 80))
reference[channels_y + 30:channels_y + 110, 690:790] = v_small
cv2.putText(reference, "VAL", (725, channels_y + 125),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
# Notes section
notes_y = 390
cv2.putText(reference, "NOTES:", (20, notes_y + 15),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
cv2.putText(reference, "- OpenCV uses BGR (not RGB!)", (20, notes_y +40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(reference, "- HSV is best for color detection", (20, notes_y + 65),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(reference, "- Use cv2.cvtColor() to convert between spaces",(20, notes_y + 90),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.imshow("COLOR SPACE REFERENCE GUIDE", reference)
cv2.waitKey(0)
cv2.imwrite("color_space_reference.png", reference)
print("Saved: color_space_reference.png")
print("\n" + "=" * 70)
print("DAY 8 COMPLETE!")
print("=" * 70)
cv2.destroyAllWindows()