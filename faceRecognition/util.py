import cv2
import numpy as np
import os
import random

from consts import ALL_MEMBERS, FACES_PATH

def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img[..., ::-1]  # Reverse channels
    img = np.around(img/255.0, decimals=12)  # Normalize
    return np.array([img])

def get_triplets(num, network):
    anchor = np.empty((0, 224, 224, 3))
    positive = np.empty((0, 224, 224, 3))
    negative = np.empty((0, 224, 224, 3))

    while num > 0:
        member1, member2 = random.sample(ALL_MEMBERS, k=2)

        anchor_path, pos_path = random.sample(os.listdir(os.path.join(FACES_PATH, member1)), k=2)
        neg_path = random.sample(os.listdir(os.path.join(FACES_PATH, member2)), k=1)[0]

        candidate_anchor = load_img(os.path.join(FACES_PATH, member1, anchor_path))
        candidate_positive = load_img(os.path.join(FACES_PATH, member1, pos_path))
        candidate_negative = load_img(os.path.join(FACES_PATH, member2, neg_path))

        embedding_anchor = network(candidate_anchor)
        embedding_positive = network(candidate_positive)
        embedding_negative = network(candidate_negative)

        if (np.linalg.norm(embedding_anchor - embedding_positive) + 0.2 >= np.linalg.norm(embedding_anchor - embedding_negative)):
            anchor = np.append(anchor, candidate_anchor, axis=0)
            positive = np.append(positive, candidate_positive, axis=0)
            negative = np.append(negative, candidate_negative, axis=0)
            num -= 1

    return [np.array(anchor), np.array(positive), np.array(negative)]

def get_pairs(num_samples):
    left = np.empty((0, 224, 224, 3))
    right = np.empty((0, 224, 224, 3))
    out = np.zeros((0, 1))

    for _ in range(num_samples // 2):
        member = random.sample(os.listdir(FACES_PATH), k=1)[0]
        sample_left_path, sample_right_path = random.sample(
            os.listdir(os.path.join(FACES_PATH, member)), k=2)
        left_img = load_img(os.path.join(
            FACES_PATH, member, sample_left_path))
        right_img = load_img(os.path.join(
            FACES_PATH, member, sample_right_path))

        left = np.append(left, left_img, axis=0)
        right = np.append(right, right_img, axis=0)
        out = np.append(out, [[1]], axis=0)

    for _ in range(num_samples // 2):
        member_left, member_right = random.sample(
            os.listdir(FACES_PATH), k=2)
        left_img = load_img(
            os.path.join(FACES_PATH, member_left, random.sample(os.listdir(os.path.join(FACES_PATH, member_left)), k=1)[0]))
        right_img = load_img(
            os.path.join(FACES_PATH, member_right, random.sample(os.listdir(os.path.join(FACES_PATH, member_right)), k=1)[0]))

        left = np.append(left, left_img, axis=0)
        right = np.append(right, right_img, axis=0)
        out = np.append(out, [[0]], axis=0)

    return [left, right, out]



