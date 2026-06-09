
# day7_mini_challenge.py
# Week 1 Review and Mini Challenge
from __future__ import print_function
import numpy as np
import argparse
import cv2
import imutils # Our utility from Day 5
import os
from datetime import datetime
print("=" * 70)
print("DAY 7: WEEK 1 REVIEW & MINI CHALLENGE")
print("=" * 70)
# -----------------------------------------------------------------
# STEP 1: LOAD THE IMAGE
# -----------------------------------------------------------------
print("\n[STEP 1] Loading the image...")
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="Path to the main image")
ap.add_argument("-j", "--blend", required=False,help="Path to blend image (optional)")
args = vars(ap.parse_args())
# Load main image
image = cv2.imread(args["image"])
if image is None:
    print("ERROR: Could not load main image!")
    exit()
height, width = image.shape[:2]
print(f"Loaded: {width} x {height} pixels")
# Load blend image if provided
blend_image = None
if args["blend"]:
    blend_image = cv2.imread(args["blend"])
    if blend_image is not None:
        blend_image = cv2.resize(blend_image, (width, height))
        print("Loaded blend image")
# Create output directory
output_dir = "day7_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
cv2.imshow("1. Original Image", image)
cv2.waitKey(0)
# -----------------------------------------------------------------
# STEP 2: DRAW ANNOTATIONS
# -----------------------------------------------------------------
print("\n[STEP 2] Adding annotations to the image...")
annotated = image.copy()
# Draw a bounding box around the center region
box_size = min(width, height) // 3
center_x, center_y = width // 2, height // 2
x1 = center_x - box_size // 2
y1 = center_y - box_size // 2
x2 = center_x + box_size // 2
y2 = center_y + box_size // 2
cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
# Add text label
cv2.putText(annotated, "REGION OF INTEREST", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
# Draw a circle highlighting the center
cv2.circle(annotated, (center_x, center_y), 30, (0, 0, 255), 3)
cv2.putText(annotated, "CENTER", (center_x - 35, center_y - 35),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
# Add image info text
cv2.putText(annotated, f"Size: {width}x{height}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(annotated, f"Date: {datetime.now().strftime('%Y-%m-%d')}",(10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
# Add decorative corner boxes
corner_size = 30
cv2.rectangle(annotated, (0, 0), (corner_size, corner_size), (255, 0, 0),3)
cv2.rectangle(annotated, (width - corner_size, 0), (width, corner_size),(255, 0, 0), 3)
cv2.rectangle(annotated, (0, height - corner_size), (corner_size,height), (255, 0, 0), 3)
cv2.rectangle(annotated, (width - corner_size, height - corner_size),(width, height), (255, 0, 0), 3)
cv2.imshow("2. Annotated Image", annotated)
cv2.waitKey(0)
cv2.imwrite(f"{output_dir}/01_annotated.png", annotated)
print("Saved: 01_annotated.png")
# -----------------------------------------------------------------
# STEP 3: APPLY TRANSFORMATIONS
# -----------------------------------------------------------------
print("\n[STEP 3] Applying transformations...")
# 3a: Resize (maintaining aspect ratio)
resized = imutils.resize(image, width=300)
cv2.imshow("3a. Resized (width=300)", resized)
cv2.waitKey(0)
cv2.imwrite(f"{output_dir}/02_resized.png", resized)
print("Saved: 02_resized.png")
# 3b: Rotate by 15 degrees
rotated = imutils.rotate(image, 15)
cv2.imshow("3b. Rotated 15 Degrees", rotated)
cv2.waitKey(0)
cv2.imwrite(f"{output_dir}/03_rotated.png", rotated)
print("Saved: 03_rotated.png")
# 3c: Horizontal flip
flipped = imutils.flip(image, 'horizontal')
cv2.imshow("3c. Horizontally Flipped", flipped)
cv2.waitKey(0)
cv2.imwrite(f"{output_dir}/04_flipped.png", flipped)
print("Saved: 04_flipped.png")
# 3d: Crop the center region
cropped = imutils.crop(image, y1, y2, x1, x2)
cv2.imshow("3d. Cropped Center Region", cropped)
cv2.waitKey(0)
cv2.imwrite(f"{output_dir}/05_cropped.png", cropped)
print("Saved: 05_cropped.png")
# -----------------------------------------------------------------
# STEP 4: CREATE BLENDED VERSION
# -----------------------------------------------------------------
print("\n[STEP 4] Creating blended version...")
if blend_image is not None:
 # Try different blend ratios
    blends = [
    (0.3, 0.7, "30% Main, 70% Blend"),
    (0.5, 0.5, "50% Main, 50% Blend"),
    (0.7, 0.3, "70% Main, 30% Blend")
    ]
 
    for i, (alpha, beta, label) in enumerate(blends):
        blended = cv2.addWeighted(image, alpha, blend_image, beta, 0)
        cv2.putText(blended, label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(f"4. Blended: {label}", blended)
        cv2.waitKey(0)
        cv2.imwrite(f"{output_dir}/06_blended_{i+1}.png", blended)
        print(f"Saved: 06_blended_{i+1}.png")
else:
 # If no blend image, create a color overlay blend
    print("No blend image provided. Creating color overlay...")
 
 # Create a gradient color overlay
    overlay = np.zeros(image.shape, dtype="uint8")
    for y in range(height):
        color_value = int(y * 255 / height)
        cv2.line(overlay, (0, y), (width, y), (color_value, 255 - color_value, 128), 1)
 
    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    cv2.imshow("4. Color Overlay Blend", blended)
    cv2.waitKey(0)
    cv2.imwrite(f"{output_dir}/06_blended_overlay.png", blended)
    print("Saved: 06_blended_overlay.png")
# -----------------------------------------------------------------
# STEP 5: APPLY MASKING
# -----------------------------------------------------------------
print("\n[STEP 5] Creating and applying masks...")
# Create an elliptical mask
mask = np.zeros((height, width), dtype="uint8")
cv2.ellipse(mask, (center_x, center_y), (box_size, box_size//2), 0, 0,360, 255, -1)
cv2.imshow("5a. Elliptical Mask", mask)
cv2.waitKey(0)
cv2.imwrite(f"{output_dir}/07_mask.png", mask)
print("Saved: 07_mask.png")
# Apply mask to original image
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("5b. Masked Image (elliptical region)", masked)
cv2.waitKey(0)
cv2.imwrite(f"{output_dir}/08_masked.png", masked)
print("Saved: 08_masked.png")
# Create inverted mask and apply
mask_inv = cv2.bitwise_not(mask)
masked_inv = cv2.bitwise_and(image, image, mask=mask_inv)
cv2.imshow("5c. Inverted Mask (everything outside ellipse)", masked_inv)
cv2.waitKey(0)
cv2.imwrite(f"{output_dir}/09_masked_inverted.png", masked_inv)
print("Saved: 09_masked_inverted.png")
# Create a gradient mask (for smooth transition)
gradient_mask = np.zeros((height, width), dtype="uint8")
for x in range(width):
    value = int(255 * x / width)
    gradient_mask[:, x] = value
gradient_masked = cv2.bitwise_and(image, image, mask=gradient_mask)
cv2.imshow("5d. Gradient Mask (smooth transition)", gradient_masked)
cv2.waitKey(0)
cv2.imwrite(f"{output_dir}/10_gradient_masked.png", gradient_masked)
print("Saved: 10_gradient_masked.png")
# -----------------------------------------------------------------
# STEP 6: CREATE A MASTER COLLAGE
# -----------------------------------------------------------------
print("\n[STEP 6] Creating master collage of all versions...")
# Define the images to include in collage
collage_images = [
 ("Original", image),
 ("Annotated", annotated),
 ("Resized", cv2.resize(resized, (200, 150))),
 ("Rotated", cv2.resize(rotated, (200, 150))),
 ("Flipped", cv2.resize(flipped, (200, 150))),
 ("Cropped", cv2.resize(cropped, (200, 150))),
 ("Masked", cv2.resize(masked, (200, 150))),
 ("Blended", cv2.resize(blended if blend_image is not None else
blended, (200, 150)))
]
# Create a 2x4 grid
grid_rows = []
current_row = []
target_height = 150
target_width = 200
for i, (label, img) in enumerate(collage_images):
 # Resize to consistent size
    if img.shape[0] != target_height or img.shape[1] != target_width:
        img = cv2.resize(img, (target_width, target_height))
 
 # Add label
        cv2.putText(img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255, 255), 1)
 
        current_row.append(img)
 
    if len(current_row) == 4:
        grid_rows.append(np.hstack(current_row))
        current_row = []
# Add any remaining images
if current_row:
    while len(current_row) < 4:
        blank = np.zeros((target_height, target_width, 3), dtype="uint8")
        current_row.append(blank)
    grid_rows.append(np.hstack(current_row))
# Stack rows vertically
collage = np.vstack(grid_rows)
# Add title banner
title_height = 50
title_banner = np.zeros((title_height, collage.shape[1], 3),dtype="uint8")
cv2.putText(title_banner, "DAY 7 MINI CHALLENGE - WEEK 1 SUMMARY",(collage.shape[1]//2 - 250, 35),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
final_collage = np.vstack([title_banner, collage])
cv2.imshow("6. MASTER COLLAGE - All Results", final_collage)
cv2.waitKey(0)
cv2.imwrite(f"{output_dir}/11_master_collage.png", final_collage)
print("Saved: 11_master_collage.png")
# -----------------------------------------------------------------
# STEP 7: CREATE A PIPELINE FUNCTION
# -----------------------------------------------------------------
print("\n[STEP 7] Creating a reusable pipeline function...")
def image_processing_pipeline(image,
                              resize_width=None,
                              rotate_angle=None,
                              flip_direction=None,
                              crop_region=None,
                              brightness_add=0,
                              apply_mask=False):
    """
    Complete image processing pipeline
    """
    
    result = image.copy()

 
 # Step 1: Resize
    if resize_width:
        result = imutils.resize(result, width=resize_width)
 
 # Step 2: Rotate
    if rotate_angle:
        result = imutils.rotate(result, rotate_angle)
 
 # Step 3: Flip
    if flip_direction:
        result = imutils.flip(result, flip_direction)
 
 # Step 4: Crop
    if crop_region:
        y1, y2, x1, x2 = crop_region
        result = imutils.crop(result, y1, y2, x1, x2)
 
 # Step 5: Adjust brightness
    if brightness_add != 0:
        if brightness_add > 0:
            result = cv2.add(result, np.ones(result.shape, dtype="uint8")* brightness_add)
        else:
            result = cv2.subtract(result, np.ones(result.shape,dtype="uint8") * abs(brightness_add))
 
 # Step 6: Apply circular mask
    if apply_mask:
        h, w = result.shape[:2]
        mask = np.zeros((h, w), dtype="uint8")
        center = (w//2, h//2)
        radius = min(h, w) // 3
        cv2.circle(mask, center, radius, 255, -1)
        result = cv2.bitwise_and(result, result, mask=mask)
 
        return result
# Test the pipeline
print("\nTesting pipeline with different parameters...")
test_configs = [
 ("Resized only", {"resize_width": 250}),
 ("Rotated 30 deg", {"rotate_angle": 30}),
 ("Flipped horizontal", {"flip_direction": "horizontal"}),
 ("Brightness +80", {"brightness_add": 80}),
 ("All transforms", {"resize_width": 200, "rotate_angle": 15,
 "flip_direction": "horizontal", "brightness_add":
30}),
 ("With mask", {"resize_width": 250, "apply_mask": True})
]
for name, params in test_configs:
    processed = image_processing_pipeline(image, **params)
    cv2.putText(processed, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
    cv2.imshow(f"Pipeline: {name}", processed)
    cv2.waitKey(0)
    safe_name = name.replace(" ", "_").lower()
    cv2.imwrite(f"{output_dir}/12_pipeline_{safe_name}.png", processed)
    print(f"Saved: 12_pipeline_{safe_name}.png")
# -----------------------------------------------------------------
# STEP 8: GENERATE REPORT
# -----------------------------------------------------------------
print("\n[STEP 8] Generating project report...")
report = f"""
=========================================================================
=======
DAY 7 MINI CHALLENGE - PROJECT REPORT
=========================================================================
=======
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Input Image: {args['image']}
Image Dimensions: {width} x {height} pixels
=========================================================================
=======
TASKS COMPLETED
=========================================================================
=======
✓ STEP 1: Loaded image successfully
✓ STEP 2: Added annotations (bounding box, text, circle, corner 
decorations)
✓ STEP 3: Applied transformations:
 - Resized to width=300
 - Rotated 15 degrees
 - Horizontal flip
 - Cropped center region
✓ STEP 4: Created blended version
✓ STEP 5: Created and applied masks:
 - Elliptical mask
 - Inverted mask
 - Gradient mask
✓ STEP 6: Created master collage
✓ STEP 7: Built reusable pipeline function
✓ STEP 8: Generated this report
=========================================================================
=======
FILES GENERATED
=========================================================================
=======
Output Directory: {output_dir}/
01_annotated.png - Image with annotations
02_resized.png - Resized image
03_rotated.png - Rotated image
04_flipped.png - Horizontally flipped image
05_cropped.png - Cropped center region
06_blended_*.png - Blended versions
07_mask.png - Elliptical mask
08_masked.png - Masked image
09_masked_inverted.png - Inverted mask result
10_gradient_masked.png - Gradient mask result
11_master_collage.png - Complete summary collage
12_pipeline_*.png - Pipeline test results
=========================================================================
=======
SKILLS DEMONSTRATED
=========================================================================
=======
• Image loading, display, and saving
• Pixel manipulation and coordinate systems
• Drawing shapes and text
• Image transformations (resize, rotate, flip, crop)
• Image arithmetic and blending
• Bitwise operations and masking
• Creating reusable functions
• Documentation and reporting
=========================================================================
=======
END OF REPORT
=========================================================================
=======
"""
# Save report
with open(f"{output_dir}/REPORT.txt", "w") as f:
 f.write(report)
print(report)
# -----------------------------------------------------------------
# CONCLUSION
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("DAY 7 MINI CHALLENGE - COMPLETE!")
print("=" * 70)
print(f"\nAll outputs saved to: {output_dir}/")
print("\nCONGRATULATIONS! You have completed Week 1!")
print("\nSkills acquired:")
print(" ✓ Environment setup")
print(" ✓ Image I/O and properties")
print(" ✓ Pixel manipulation")
print(" ✓ Drawing on images")
print(" ✓ Transformations (translate, rotate, resize, flip, crop)")
print(" ✓ Image arithmetic and blending")
print(" ✓ Bitwise operations and masking")
print("\nReady for Week 2!")
cv2.destroyAllWindows()