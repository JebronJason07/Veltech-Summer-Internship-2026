# Line 1: Import print_function for Python 2 and Python 3 compatibility
from __future__ import print_function

# Line 2: Import the RGBHistogram class
from pyimagesearch.rgbhistogram import RGBHistogram

# Line 3: Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Line 4: Import Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

# Line 5: Import train_test_split
from sklearn.model_selection import train_test_split

# Line 6: Import classification_report
from sklearn.metrics import classification_report

# Line 7: Import NumPy
import numpy as np

# Line 8: Import argparse
import argparse

# Line 9: Import glob
import glob

# Line 10: Import OpenCV
import cv2

# Create ArgumentParser
ap = argparse.ArgumentParser()

# Add image dataset path
ap.add_argument(
    "-i",
    "--images",
    required=True,
    help="path to the image dataset"
)

# Add mask dataset path
ap.add_argument(
    "-m",
    "--masks",
    required=True,
    help="path to the image masks"
)

# Parse arguments
args = vars(ap.parse_args())

# Load image and mask paths
imagePaths = sorted(glob.glob(args["images"] + "/*.png"))
maskPaths = sorted(glob.glob(args["masks"] + "/*.png"))

# Initialize data and labels
data = []
target = []

# Create RGB histogram descriptor
desc = RGBHistogram([8, 8, 8])

# Loop over the dataset
for (imagePath, maskPath) in zip(imagePaths, maskPaths):

    # Load image and mask
    image = cv2.imread(imagePath)
    mask = cv2.imread(maskPath)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Extract features
    features = desc.describe(image, mask)

    # Store features
    data.append(features)

    # Store label
    target.append(imagePath.split("_")[-2])

# Obtain unique class names
targetNames = np.unique(target)

# Encode labels
le = LabelEncoder()
target = le.fit_transform(target)

# Split dataset
(trainData, testData, trainTarget, testTarget) = train_test_split(
    data,
    target,
    test_size=0.3,
    random_state=42
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=25,
    random_state=84
)

model.fit(trainData, trainTarget)

# Print classification report
print(
    classification_report(
        testTarget,
        model.predict(testData),
        target_names=targetNames
    )
)

# Test on 10 random images
for i in np.random.choice(np.arange(0, len(imagePaths)), 10):

    imagePath = imagePaths[i]
    maskPath = maskPaths[i]

    image = cv2.imread(imagePath)
    mask = cv2.imread(maskPath)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    features = desc.describe(image, mask)

    flower = le.inverse_transform(model.predict([features]))[0]

    print(imagePath)
    print("I think this flower is a {}".format(flower.upper()))

    cv2.imshow("Image", image)
    cv2.waitKey(0)