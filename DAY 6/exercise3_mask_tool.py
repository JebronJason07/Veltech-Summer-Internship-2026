# exercise_3_mask_tool.py
import cv2
import numpy as np
image = cv2.imread("test_image.png")
mask = np.zeros(image.shape[:2], dtype="uint8")
drawing = False
ix, iy = -1, -1
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, mask
 
    if event == cv2.EVENT_LBUTTONDOWN:
       drawing = True
       ix, iy = x, y
 
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(mask, (ix, iy), (x, y), 255, -1)
 
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(mask, (ix, iy), (x, y), 255, -1)
cv2.namedWindow("Select Region")
cv2.setMouseCallback("Select Region", draw_rectangle)
print("Draw a rectangle on the image to create a mask")
print("Press 's' to save mask, 'r' to reset, ESC to exit")
while True:
 # Show mask overlay on image
    masked_vis = image.copy()
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked_vis = cv2.addWeighted(masked_vis, 0.7, mask_colored, 0.3, 0)
 
    cv2.imshow("Select Region", masked_vis)
 
    key = cv2.waitKey(10) & 0xFF
    if key == ord('s'):
 # Apply mask and show result
        result = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow("Masked Result", result)
        cv2.imwrite("custom_mask.png", mask)
        cv2.imwrite("masked_result.png", result)
        print("Saved mask and result!")
    elif key == ord('r'):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        print("Mask reset")
    elif key == 27: # ESC
        break
cv2.destroyAllWindows()