import cv2
import numpy as np
import os

from TripletPickerHelper import TripletPickerHelper
from PairPickerHelper import PairPickerHelper

TRIPLET_PICKER_HELPER = TripletPickerHelper()
PAIR_PICKER_HELPER = PairPickerHelper()

def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img[..., ::-1]  # Reverse channels
    img = np.around(img/255.0, decimals=12)  # Normalize
    return np.array([img])

def compute_pairwise_distance_matrix(embeddings):
    gram = np.matmul(embeddings, np.transpose(embeddings))
    norms = np.diag(gram)
    dist_mat = np.expand_dims(norms, 0) - 2.0 * gram + np.expand_dims(norms, 1)
    return dist_mat

# def compute_pairwise_distance_matrix_between_2_sets_of_embeddings(embeddings_1, embeddings_2):
#     norms_1 = np.diag(np.matmul(embeddings_1, np.transpose(embeddings_1)))
#     norms_2 = np.diag(np.matmul(embeddings_2, np.transpose(embeddings_2)))
#     dot_product_mat = np.matmul(embeddings_1, np.transpose(embeddings_2))
#     return np.expand_dims(norms_1, 1) - 2 * dot_product_mat + np.expand_dims(norms_2, 0)

def compute_distances_between_embeddings(e1, e2):
    norms1 = np.diag(np.matmul(e1, np.transpose(e1)))
    norms2 = np.diag(np.matmul(e2, np.transpose(e2)))
    dotprd = np.diag(np.matmul(e1, np.transpose(e2)))
    return norms1 + norms2 - 2 * dotprd

def get_triplets(num, network, alpha=0.2):
    print("Grabbing triplets ...")
    anchor = np.empty((0, 224, 224, 3))
    positive = np.empty((0, 224, 224, 3))
    negative = np.empty((0, 224, 224, 3))

    while num > 0:
        print("{} triplets to go ... ".format(num))
        anchorPaths, positivePaths, negativePaths = TRIPLET_PICKER_HELPER.chooseTriplets(128)

        embeddings_anc = [np.squeeze(network(load_img(path))) for path in anchorPaths]
        embeddings_pos = [np.squeeze(network(load_img(path))) for path in positivePaths]
        embeddings_neg = [np.squeeze(network(load_img(path))) for path in negativePaths]

        dist_pos = compute_distances_between_embeddings(embeddings_anc, embeddings_pos)
        dist_neg = compute_distances_between_embeddings(embeddings_anc, embeddings_neg)

        triplet_dists = dist_pos - dist_neg + alpha

        good_triplets = np.nonzero(triplet_dists > 0)[0]

        for index in good_triplets:
            anchor = np.append(anchor, load_img(anchorPaths[index]), axis=0)
            positive = np.append(positive, load_img(positivePaths[index]), axis=0)
            negative = np.append(negative, load_img(negativePaths[index]), axis=0)
            num -= 1

    return [np.array(anchor), np.array(positive), np.array(negative)]

def get_pairs(num_samples):
    left = np.empty((0, 224, 224, 3))
    right = np.empty((0, 224, 224, 3))
    out = np.zeros((0, 1))

    for (leftPath, rightPath) in PAIR_PICKER_HELPER.choosePositivePairs(num_samples // 2):
        left = np.append(left, load_img(leftPath), axis=0)
        right = np.append(right, load_img(rightPath), axis=0)
        out = np.append(out, [[1]], axis=0)

    for (leftPath, rightPath) in PAIR_PICKER_HELPER.chooseNegativePairs(num_samples // 2):
        left = np.append(left, load_img(leftPath), axis=0)
        right = np.append(right, load_img(rightPath), axis=0)
        out = np.append(out, [[0]], axis=0)

    return [left, right, out]
