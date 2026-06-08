# exercise_3_rotation_animation.py
import cv2
import imutils
image = cv2.imread("test_image.png")
# Create a sequence of rotated images
for angle in range(0, 360, 15):
    rotated = imutils.rotate(image, angle)
 
 # Add angle text
    cv2.putText(rotated, f"Angle: {angle} deg", (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),2)
 
    cv2.imshow("Rotating Animation (Press any key to stop)",rotated)
 
 # Wait 100ms between frames
    if cv2.waitKey(100) != -1:
        break
    cv2.destroyAllWindows()