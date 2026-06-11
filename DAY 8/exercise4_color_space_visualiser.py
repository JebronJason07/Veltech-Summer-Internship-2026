# exercise_4_color_space_visualizer.py
import cv2
import numpy as np
def create_hsv_wheel():
    """Create a visualization of the HSV color wheel"""
    wheel = np.zeros((360, 360, 3), dtype="uint8")
 
    for h in range(180): # Hue 0-179
        for s in range(255): # Saturation 0-255
 # Map to wheel coordinates
            angle = h * 2 # 0-358 degrees
            radius = s / 255 * 150
 
            x = int(180 + radius * np.cos(np.radians(angle)))
            y = int(180 + radius * np.sin(np.radians(angle)))
 
            if 0 <= x < 360 and 0 <= y < 360:
                wheel[y, x] = (h, s, 255) # Full value
 
    wheel_bgr = cv2.cvtColor(wheel, cv2.COLOR_HSV2BGR)
    return wheel_bgr
# Create HSV wheel
hsv_wheel = create_hsv_wheel()
cv2.imshow("HSV Color Wheel", hsv_wheel)
cv2.waitKey(0)
cv2.imwrite("hsv_color_wheel.png", hsv_wheel)
# Create value gradient
value_gradient = np.zeros((256, 512, 3), dtype="uint8")
for v in range(256):
    value_gradient[v, :] = (0, 0, v) # Varying value
value_gradient_hsv = cv2.cvtColor(value_gradient, cv2.COLOR_BGR2HSV)
value_gradient_bgr = cv2.cvtColor(value_gradient_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("Value Gradient (Brightness)", value_gradient_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()