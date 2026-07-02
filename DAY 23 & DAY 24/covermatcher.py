# Line 1: Import the NumPy library for numerical and array operations
import numpy as np

# Line 2: Import the OpenCV library for image processing and feature matching
import cv2


# Line 4: Define a class named CoverMatcher
class CoverMatcher:
    # Lines 5-14: Initialize the cover matcher and configure matching parameters
    def __init__(self, descriptor, coverPaths, ratio=0.7,
                 minMatches=40, useHamming=True):

        # Line 7: Store the feature descriptor object
        self.descriptor = descriptor

        # Line 8: Store the list of cover image paths
        self.coverPaths = coverPaths

        # Line 9: Store Lowe's ratio test threshold
        self.ratio = ratio

        # Line 10: Store the minimum number of matches required
        self.minMatches = minMatches

        # Line 11: Set the default feature matching method
        self.distanceMethod = "BruteForce"

        # Lines 13-14: Use Hamming distance if binary descriptors are being used
        if useHamming:
            self.distanceMethod += "-Hamming"

    # Lines 16-31: Define a method to search for matching cover images
    def search(self, queryKps, queryDescs):
        # Create a dictionary to store matching results
        results = {}

        # Loop through each cover image in the database
        for coverPath in self.coverPaths:
            # Load the current cover image
            cover = cv2.imread(coverPath)

            # Convert the cover image to grayscale
            gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)

            # Extract keypoints and descriptors from the cover image
            (kps, descs) = self.descriptor.describe(gray)

            # Compare the query image features with the cover image features
            score = self.match(queryKps, queryDescs, kps, descs)

            # Store the matching score
            results[coverPath] = score

        # Sort results by score
        if len(results) > 0:
            results = sorted(
                [(v, k) for (k, v) in results.items() if v > 0],
                reverse=True
            )

        return results

    # Lines 33-49: Define a method to match features between two images
    def match(self, kpsA, featuresA, kpsB, featuresB):
        # Create a descriptor matcher
        matcher = cv2.DescriptorMatcher_create(self.distanceMethod)

        # Find the two best matches for each descriptor
        rawMatches = matcher.knnMatch(featuresB, featuresA, 2)

        # Create a list to store good matches
        matches = []

        # Apply Lowe's ratio test
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # If enough matches exist, compute a homography
        if len(matches) > self.minMatches:
            ptsA = np.float32([kpsA[i] for (i, _) in matches])
            ptsB = np.float32([kpsB[j] for (_, j) in matches])

            (_, status) = cv2.findHomography(
                ptsA,
                ptsB,
                cv2.RANSAC,
                4.0
            )

            # Return the percentage of inlier matches
            return float(status.sum()) / status.size

        # Return -1.0 if there are not enough matches
        return -1.0