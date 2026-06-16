# exercise_1_threshold_optimizer.py
import cv2
import numpy as np
def threshold_optimizer(image_path):
    """Find optimal threshold by analyzing histogram"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
 # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
 
 # Find peaks in histogram
    peaks = []
    for i in range(1, 255):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks.append((i, hist[i]))
 
 # Sort peaks by height
    peaks.sort(key=lambda x: x[1], reverse=True)
 
    if len(peaks) >= 2:
 # Optimal threshold is between two highest peaks
        valley_start = min(peaks[0][0], peaks[1][0])
        valley_end = max(peaks[0][0], peaks[1][0])
 
 # Find minimum between peaks
        min_val = float('inf')
        optimal_T = (valley_start + valley_end) // 2
 
        for i in range(valley_start, valley_end + 1):
            if hist[i] < min_val:
                min_val = hist[i]
                optimal_T = i
 
        print(f"Detected two peaks at {peaks[0][0]} and {peaks[1][0]}")
        print(f"Optimal threshold T = {optimal_T}")
    else:
        optimal_T = 127
        print(f"Only {len(peaks)} peak(s) found. Using T=127")
 
 # Apply threshold
    _, thresholded = cv2.threshold(gray, optimal_T, 255,cv2.THRESH_BINARY)
 
 # Show results
    comparison = np.hstack([
        cv2.resize(gray, (300, 250)),
        cv2.resize(thresholded, (300, 250))
        ])
 
    cv2.putText(comparison, f"Optimal T = {optimal_T}", (320, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
    cv2.imshow("Threshold Optimizer", comparison)
    cv2.waitKey(0)
 
    return optimal_T
# Run optimizer
threshold_optimizer("test_image.png")
cv2.destroyAllWindows()