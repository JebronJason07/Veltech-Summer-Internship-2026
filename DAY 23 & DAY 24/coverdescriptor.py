# Line 1: Import the NumPy library for numerical and array operations
import numpy as np

# Line 2: Import the OpenCV library for image processing and feature extraction
import cv2


# Line 4: Define a class named CoverDescriptor
class CoverDescriptor:
    # Lines 6-7: Initialize the descriptor and specify whether SIFT should be used
    def __init__(self, useSIFT=False):
        self.useSIFT = useSIFT

    # Lines 8-17: Define a method to extract keypoints and feature descriptors from an image
    def describe(self, image):
        # Line 9: Create a BRISK feature descriptor object
        descriptor = cv2.BRISK_create()

        # Lines 11-12: If SIFT is enabled, use the SIFT descriptor instead of BRISK
        if self.useSIFT:
            descriptor = cv2.SIFT_create()

        # Line 14: Detect keypoints and compute feature descriptors
        (kps, descs) = descriptor.detectAndCompute(image, None)

        # Line 15: Convert keypoint objects into NumPy arrays containing (x, y) coordinates
        kps = np.float32([kp.pt for kp in kps])

        # Line 17: Return the keypoints and descriptors
        return (kps, descs)