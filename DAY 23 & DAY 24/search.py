
# Line 1: Import print_function from future versions of Python for Python 2 and 3 compatibility
from __future__ import print_function

# Line 2: Import the CoverDescriptor class for extracting image features
from pyimagesearch.coverdescriptor import CoverDescriptor

# Line 3: Import the CoverMatcher class for matching book covers
from pyimagesearch.covermatcher import CoverMatcher
import os 

# Line 4: Import argparse for handling command-line arguments
import argparse

# Line 5: Import glob for finding files using wildcard patterns
import glob

# Line 6: Import csv for reading book database information
import csv

# Line 7: Import OpenCV for image processing
import cv2

# Line 9: Create an ArgumentParser object
ap = argparse.ArgumentParser()

# Lines 10-11: Add a required argument specifying the book database CSV file
ap.add_argument(
    "-d",
    "--db",
    required=True,
    help="path to the book database"
)

# Lines 12-13: Add a required argument specifying the directory containing book cover images
ap.add_argument(
    "-c",
    "--covers",
    required=True,
    help="path to the directory that contains our book covers"
)

# Lines 14-15: Add a required argument specifying the query book cover image
ap.add_argument(
    "-q",
    "--query",
    required=True,
    help="path to the query book cover"
)

# Lines 16-17: Add an optional argument to enable or disable SIFT descriptors
ap.add_argument(
    "-s",
    "--sift",
    type=int,
    default=0,
    help="whether or not SIFT should be used"
)

# Line 18: Parse command-line arguments and store them as a dictionary
args = vars(ap.parse_args())

# Line 20: Create an empty dictionary to store book information
db = {}

# Lines 22-23: Read the CSV database and store each record in the dictionary
for l in csv.reader(open(args["db"])):
    db[l[0]] = l[1:]

# Line 25: Check whether the user has enabled SIFT
useSIFT = args["sift"] > 0

# Line 26: Use Hamming distance matching when SIFT is not enabled
useHamming = args["sift"] == 0

# Line 27: Set Lowe's ratio test threshold
ratio = 0.7

# Line 28: Set the default minimum number of matches required
minMatches = 40

# Lines 30-31: If SIFT is being used, increase the minimum match threshold
if useSIFT:
    minMatches = 50

# Line 32: Create a CoverDescriptor object
cd = CoverDescriptor(useSIFT=useSIFT)

# Lines 33-34: Create a CoverMatcher object
cv = CoverMatcher(
    cd,
    glob.glob(args["covers"] + "/*.png"),
    ratio=ratio,
    minMatches=minMatches,
    useHamming=useHamming
)

# Line 36: Load the query book cover image
queryImage = cv2.imread(args["query"])

# Line 37: Convert the query image to grayscale
gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

# Line 38: Extract keypoints and descriptors from the query image
(queryKps, queryDescs) = cd.describe(gray)

# Line 40: Search the cover database for matching book covers
results = cv.search(queryKps, queryDescs)

# Line 42: Display the query image
cv2.imshow("Query", queryImage)

# Lines 44-46: Check if no matching book covers were found
if len(results) == 0:
    print("I could not find a match for that cover!")
    cv2.waitKey(0)

# Lines 48-56: Execute if one or more matches were found
else:
    for (i, (score, coverPath)) in enumerate(results):
        # Retrieve the author and title from the database
        filename = os.path.basename(coverPath)
        (author, title) = db[filename]
        # Display the rank, matching percentage, author name, and book title
        print(
            "{}. {:.2f}% : {} - {}".format(
                i + 1,
                score * 100,
                author,
                title
            )
        )

        # Load the matched cover image
        result = cv2.imread(coverPath)

        # Display the matched cover image
        cv2.imshow("Result", result)

        # Wait for a key press before showing the next result
        cv2.waitKey(0)