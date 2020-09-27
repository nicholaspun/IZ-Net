from PIL import Image, ImageDraw
import cv2
import json
import numpy as np
import os
import io

def getBoundingBoxCoordinates(image, boundingBoxRow, boundingBoxCol, boundingBoxInfo):
    imageWidth, imageHeight = image.size
    widthScale = imageWidth / 13
    heightScale = imageHeight / 13

    (bbInfoX, bbInfoY) = boundingBoxInfo[:2]
    (bbInfoW, bbInfoH) = boundingBoxInfo[2:]

    midX = bbInfoX + boundingBoxCol
    midY = bbInfoY + boundingBoxRow

    startX = (midX - (bbInfoW / 2.)) * widthScale
    startY = (midY - (bbInfoH / 2.)) * heightScale
    endX = (midX + (bbInfoW / 2.)) * widthScale
    endY = (midY + (bbInfoH / 2.)) * heightScale

    return [startY, startX, endY, endX]

def drawImg(image, boundingBoxRow, boundingBoxCol, boundingBoxInfo):
    Draw = ImageDraw.Draw(image)
    (startY, startX, endY, endX) = getBoundingBoxCoordinates(image, boundingBoxRow, boundingBoxCol, boundingBoxInfo)
    for i in range(10):
        Draw.rectangle([(startX + i, startY + i), (endX - i, endY - i)], outline='red')

with open("output.json") as infile:
    memfile = io.BytesIO()
    memfile.write(json.load(infile).encode('latin-1'))
    memfile.seek(0)
    data = np.load(memfile, allow_pickle=True)

    for filepath, encoding in data[()].items():
        print("Checking {}".format(filepath))
        theImage = Image.open(filepath).convert('RGB')

        for row in range(13):
            for col in range(13):
                for boxNum in range(2):
                    if encoding[row][col][boxNum][0] == 1:
                        drawImg(theImage, row, col, encoding[row][col][boxNum][1:])
        
        cv2.imwrite(os.path.join('check', filepath.replace('train/', '')), np.array(theImage)[..., ::-1])