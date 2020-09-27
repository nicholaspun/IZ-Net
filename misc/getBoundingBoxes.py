from math import floor
from PIL import Image, ImageDraw, ImageFont
import cv2
import json
import numpy as np
import os
import io

PROTOTXT_PATH = os.path.join('model_data', 'deploy.prototxt')
CAFFEMODEL_PATH = os.path.join('model_data', 'weights.caffemodel')

def getClosestAnchor(width, height):
    ratio = np.true_divide(height, width)
    return 1 if ((ratio - 1.15) ** 2) < ((ratio - 1) ** 2) else 0 

print('Reading model ...')
model = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
print('Model read ...')

output = {}
maxBoundingBoxes = 0
width = []
height = []

for member in os.listdir('train'):
    for imagePath in os.listdir(os.path.join('train', member)):
        fullPath = os.path.join('train', member, imagePath)
        print("Computing bounding boxes for {}".format(fullPath))

        image = cv2.imread(fullPath)
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (299, 299)), 1.0, (299, 299), (104.0, 177.0, 123.0))
        model.setInput(blob)
        detections = model.forward()

        encoding = np.zeros((13, 13, 2, 5))

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if (confidence > 0.5):
                (startX, startY, endX, endY) = detections[0, 0, i, 3:7]

                midpointX = (startX + endX) / 2
                midpointY = (startY + endY) / 2

                boundingBoxRow = floor(midpointY * 13)
                boundingBoxCol = floor(midpointX * 13)

                percentBoundingBoxX = (midpointX * 13) - floor(midpointX * 13)
                percentBoundingBoxY = (midpointY * 13) - floor(midpointY * 13)

                scaledWidth = (endX - startX) * (299 / 13)
                scaledHeight = (endY - startY) * (299 / 13)

                closestAnchor = getClosestAnchor(width, height)

                if encoding[boundingBoxRow][boundingBoxCol][closestAnchor][0] == 1:
                    closestAnchor = (closestAnchor + 1) % 2

                encoding[boundingBoxRow][boundingBoxCol][closestAnchor][0] = 1
                encoding[boundingBoxRow][boundingBoxCol][closestAnchor][1] = percentBoundingBoxX
                encoding[boundingBoxRow][boundingBoxCol][closestAnchor][2] = percentBoundingBoxY
                encoding[boundingBoxRow][boundingBoxCol][closestAnchor][3] = scaledWidth
                encoding[boundingBoxRow][boundingBoxCol][closestAnchor][4] = scaledHeight

        if not np.all(encoding == 0):
            output[fullPath] = encoding

print(len(output))

# See https://stackoverflow.com/questions/30698004/how-can-i-serialize-a-numpy-array-while-preserving-matrix-dimensions
memfile = io.BytesIO()
np.save(memfile, output)
memfile.seek(0)
serialized = json.dumps(memfile.read().decode('latin-1'))
with open("output.json", "w+") as outfile:
    outfile.write(serialized)

with open("output.json") as infile:
    memfile = io.BytesIO()
    memfile.write(json.load(infile).encode('latin-1'))
    memfile.seek(0)
    a = np.load(memfile, allow_pickle=True)
    print(len(a[()]))
