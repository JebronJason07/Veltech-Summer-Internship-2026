# imutils.py
# Custom utility functions for image processing

import numpy as np
import cv2


def translate(image, x, y):
    """
    Shift an image by x pixels to the right and y pixels down.
    """

    # Create translation matrix
    M = np.float32([[1, 0, x], [0, 1, y]])

    # Apply translation
    shifted = cv2.warpAffine(
        image,
        M,
        (image.shape[1], image.shape[0])
    )

    return shifted


def rotate(image, angle, center=None, scale=1.0):
    """
    Rotate an image by a given angle around a center point.
    """

    # Get image dimensions
    height, width = image.shape[:2]

    # If center is not provided, use image center
    if center is None:
        center = (width // 2, height // 2)

    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Apply rotation
    rotated = cv2.warpAffine(image, M, (width, height))

    return rotated


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize an image while preserving aspect ratio.
    """

    # Get original dimensions
    h, w = image.shape[:2]

    # If neither width nor height provided, return original
    if width is None and height is None:
        return image

    # Calculate new dimensions
    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    # Perform resize
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def flip(image, direction):
    """
    Flip an image horizontally, vertically, or both.
    """

    if direction == "horizontal":
        return cv2.flip(image, 1)

    elif direction == "vertical":
        return cv2.flip(image, 0)

    elif direction == "both":
        return cv2.flip(image, -1)

    else:
        raise ValueError(
            "Direction must be 'horizontal', 'vertical', or 'both'"
        )


def crop(image, start_y, end_y, start_x, end_x):
    """
    Crop a rectangular region from an image.
    """

    return image[start_y:end_y, start_x:end_x]