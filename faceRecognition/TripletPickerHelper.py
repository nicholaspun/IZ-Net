import os
import random

from Distribution import Distribution
from SegmentTree import SegmentTree
from consts import FACES_PATH, ALL_MEMBERS

def countPairs(imgs): return len(imgs) * (len(imgs) - 1) * 0.5

class TripletPickerHelper:
    def __init__(self):
        imageDistribution = [countPairs(os.listdir(os.path.join(FACES_PATH, member))) for member in ALL_MEMBERS]
        memberProbabilities = Distribution.normalizeDistribution(imageDistribution)
        memberIntervals = Distribution.distributionToIntervals(memberProbabilities)

        self.memberSegmentTree = SegmentTree.createSegmentTreeFromIntervals(memberIntervals, ALL_MEMBERS)
        self.memberPaths = {
            member: [os.path.join(FACES_PATH, member, imgName) for imgName in os.listdir(
                os.path.join(FACES_PATH, member))]
            for member in ALL_MEMBERS}
        self.allPaths = set(
            [imgPath for member, imgPaths in self.memberPaths.items() for imgPath in imgPaths])
        self.memberPathsComplements = {
            member: self.allPaths - set(self.memberPaths[member]) for member in ALL_MEMBERS}

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
