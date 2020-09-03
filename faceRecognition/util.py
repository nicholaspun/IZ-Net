import cv2
import functools
import numpy as np
import os
import random
import time

from consts import ALL_MEMBERS, FACES_PATH
from Distribution import Distribution
from SegmentTree import SegmentTree

def countPairs(imgs):
    return len(imgs) * (len(imgs) - 1) * 0.5

class TripletPickerHelper:
    def __init__(self):
        imageDistribution = [countPairs(os.listdir(os.path.join(FACES_PATH, member))) for member in ALL_MEMBERS]
        memberProbabilities = Distribution.normalizeDistribution(imageDistribution)
        memberIntervals = Distribution.distributionToIntervals(memberProbabilities)

        self.memberSegmentTree = SegmentTree.createSegmentTreeFromIntervals(memberIntervals, ALL_MEMBERS)
        self.memberPaths = { 
            member: [os.path.join(FACES_PATH, member, imgName) for imgName in os.listdir(os.path.join(FACES_PATH, member))] 
            for member in ALL_MEMBERS }
        self.allPaths = set([imgPath for member, imgPaths in self.memberPaths.items() for imgPath in imgPaths])
        self.memberPathsComplements = { member: self.allPaths - set(self.memberPaths[member]) for member in ALL_MEMBERS }

    def chooseTriplets(self, amount):
        anchors = []
        positives = []
        negatives = []

        for _ in range(amount):
            memberAnchor = self.memberSegmentTree.find(random.random())
            memberAnchorSamples = random.sample(self.memberPaths[memberAnchor], k=2)
            anchors.append(memberAnchorSamples[0])
            positives.append(memberAnchorSamples[1])
            negatives.append(random.sample(self.memberPathsComplements[memberAnchor], k=1)[0])

        return anchors, positives, negatives

add = lambda x, y: x + y

class PairPickerHelper:
    def __init__(self):
        positivePairsDistribution = [countPairs(os.listdir(os.path.join(FACES_PATH, member))) for member in ALL_MEMBERS]
        positivePairProbabilities = Distribution.normalizeDistribution(positivePairsDistribution)
        positivePairIntervals = Distribution.distributionToIntervals(positivePairProbabilities)

        totalImages = functools.reduce(add, [len(os.listdir(os.path.join(FACES_PATH, member))) for member in ALL_MEMBERS])
        negativePairsDistribution = []
        for member in ALL_MEMBERS:
            numMemberImages = len(os.listdir(os.path.join(FACES_PATH, member)))
            negativePairsDistribution.append((totalImages -numMemberImages) * numMemberImages * 2)
        negativePairsProbabilities = Distribution.normalizeDistribution(negativePairsDistribution)
        negativePairsIntervals = Distribution.distributionToIntervals(negativePairsProbabilities)

        self.positivePairsSegmentTree = SegmentTree.createSegmentTreeFromIntervals(positivePairIntervals, ALL_MEMBERS)
        self.negativePairsSegmentTree = SegmentTree.createSegmentTreeFromIntervals(negativePairsIntervals, ALL_MEMBERS)
        self.memberPaths = {
            member: [os.path.join(FACES_PATH, member, imgName) for imgName in os.listdir(os.path.join(FACES_PATH, member))]
            for member in ALL_MEMBERS }
        self.allPaths = set([imgPath for member, imgPaths in self.memberPaths.items() for imgPath in imgPaths])
        self.memberPathsComplements = { member: self.allPaths - set(self.memberPaths[member]) for member in ALL_MEMBERS }

    def choosePositivePairs(self, amount):
        pairs = []
        for _ in range(amount):
            member = self.positivePairsSegmentTree.find(random.random())
            memberSamples = random.sample(self.memberPaths[member], k = 2)
            pairs.append((memberSamples[0], memberSamples[1]))
        
        return pairs

    def chooseNegativePairs(self, amount):
        pairs = []
        for _ in range(amount):
            firstMember = self.negativePairsSegmentTree.find(random.random())
            firstMemberSample = random.sample(self.memberPaths[firstMember], k = 1)[0]
            secondMemberSample = random.sample(self.memberPathsComplements[firstMember], k = 1)[0]
            pairs.append((firstMemberSample, secondMemberSample))

        return pairs
        
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



