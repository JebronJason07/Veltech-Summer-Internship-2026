# exercise_1_thumbnail_gallery.py
import numpy as np
import argparse
import cv2
import imutils
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
# Create a 4x2 grid of thumbnails
thumb_size = 150
gallery = np.zeros((thumb_size * 2, thumb_size * 4, 3),
dtype="uint8")
transformations = [
 ("Original", lambda img: imutils.resize(img,
width=thumb_size)),
 ("H Flip", lambda img: imutils.resize(imutils.flip(img,
'horizontal'), width=thumb_size)),
 ("V Flip", lambda img: imutils.resize(imutils.flip(img,
'vertical'), width=thumb_size)),
 ("Both Flip", lambda img:
imutils.resize(imutils.flip(img, 'both'), width=thumb_size)),
 ("Rotate 30", lambda img:
imutils.resize(imutils.rotate(img, 30), width=thumb_size)),
 ("Rotate 60", lambda img:
imutils.resize(imutils.rotate(img, 60), width=thumb_size)),
 ("Crop Center", lambda img: imutils.resize(
 imutils.crop(img, img.shape[0]//4, 3*img.shape[0]//4,
 img.shape[1]//4, 3*img.shape[1]//4),
 width=thumb_size)),
 ("Zoom In", lambda img: imutils.resize(
 imutils.crop(img, img.shape[0]//3, 2*img.shape[0]//3,
 img.shape[1]//3, 2*img.shape[1]//3),
 width=thumb_size))
]
for i, (name, transform) in enumerate(transformations):
 row = i // 4
 col = i % 4
 thumb = cv2.resize(transform(image), (thumb_size, thumb_size))
 y_start = row * thumb_size
 y_end = (row + 1) * thumb_size
 x_start = col * thumb_size
 x_end = (col + 1) * thumb_size
 gallery[y_start:y_end, x_start:x_end] = thumb
 cv2.putText(gallery, name, (x_start + 5, y_start + 20),
 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255,255), 1)
 cv2.imshow("Thumbnail Gallery", gallery)
 cv2.waitKey(0)
 cv2.imwrite("thumbnail_gallery.png", gallery)
 cv2.destroyAllWindows()
 cv2.imshow("Thumbnail Gallery", gallery)
cv2.waitKey(0)
cv2.imwrite("thumbnail_gallery.png", gallery)
cv2.destroyAllWindows()