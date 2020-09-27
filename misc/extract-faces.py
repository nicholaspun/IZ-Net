import cv2
import numpy as np
import os

PROTOTXT_PATH = os.path.join('model_data', 'deploy.prototxt')
CAFFEMODEL_PATH = os.path.join('model_data', 'weights.caffemodel')
TRAIN_PATH = 'train'
FACES_PATH = 'faces'

# Read the model
print('Reading model ...')
model = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
print('Model read ...')

# Loop through all images and strip out faces
for member in os.listdir(os.path.join(TRAIN_PATH)):
    if member == 'multi':
        continue

    print('Extracting faces for {} ...'.format(member))
    for image_name in os.listdir(os.path.join(TRAIN_PATH, member)):
        image = cv2.imread(os.path.join(TRAIN_PATH, member, image_name))

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        model.setInput(blob)
        detections = model.forward()

        for i in range(detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
        
            confidence = detections[0, 0, i, 2]

            if (confidence > 0.5):
                frame = image[startY:endY, startX:endX]
                try:
                    outfileName = "{}-{}{}".format(os.path.splitext(image_name)[0], i, os.path.splitext(image_name)[1])
                    print("Saving ... {}".format(os.path.join(FACES_PATH, member, outfileName)))
                    cv2.imwrite(os.path.join(FACES_PATH, member, outfileName), frame)
                except Exception as e:
                    print("Exception occured: {}".format(e))
                    continue