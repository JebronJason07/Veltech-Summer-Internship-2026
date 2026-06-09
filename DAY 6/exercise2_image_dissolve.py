# exercise_2_image_dissolve.py
import cv2
import numpy as np
image1 = cv2.imread("test_image.png")
image2 = cv2.imread("test_pattern.png")
# Resize to match
image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
print("Creating dissolve transition...")
frames = []
# Create 50 frames of transition
for i in range(50):
 alpha = i / 50 # 0 to 1
 beta = 1 - alpha
 blended = cv2.addWeighted(image1, alpha, image2, beta, 0)
 
 # Add progress bar
 cv2.rectangle(blended, (10, blended.shape[0] - 30),
 (10 + int((blended.shape[1] - 20) * alpha),
blended.shape[0] - 10),
 (0, 255, 0), -1)
 cv2.putText(blended, f"Dissolve: {int(alpha*100)}%", (10,
blended.shape[0] - 35),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
 
 frames.append(blended)
 cv2.imshow("Image Dissolve", blended)
 cv2.waitKey(50)
# Save as video (optional)
out = cv2.VideoWriter('dissolve.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
20,
 (image1.shape[1], image1.shape[0]))
for frame in frames:
 out.write(frame)
out.release()
print("Dissolve complete! Saved as dissolve.mp4")
cv2.destroyAllWindows()