import functools
import os
import random

from Distribution import Distribution
from SegmentTree import SegmentTree
from consts import FACES_PATH, ALL_MEMBERS

def add(x, y): return x + y

def countPairs(imgs):  return len(imgs) * (len(imgs) - 1) * 0.5

class PairPickerHelper:
    def __init__(self):
        positivePairsDistribution = [countPairs(os.listdir(
            os.path.join(FACES_PATH, member))) for member in ALL_MEMBERS]
        positivePairProbabilities = Distribution.normalizeDistribution(
            positivePairsDistribution)
        positivePairIntervals = Distribution.distributionToIntervals(
            positivePairProbabilities)

        totalImages = functools.reduce(
            add, [len(os.listdir(os.path.join(FACES_PATH, member))) for member in ALL_MEMBERS])
        negativePairsDistribution = []
        for member in ALL_MEMBERS:
            numMemberImages = len(os.listdir(os.path.join(FACES_PATH, member)))
            negativePairsDistribution.append(
                (totalImages - numMemberImages) * numMemberImages * 2)
        negativePairsProbabilities = Distribution.normalizeDistribution(
            negativePairsDistribution)
        negativePairsIntervals = Distribution.distributionToIntervals(
            negativePairsProbabilities)

        self.positivePairsSegmentTree = SegmentTree.createSegmentTreeFromIntervals(
            positivePairIntervals, ALL_MEMBERS)
        self.negativePairsSegmentTree = SegmentTree.createSegmentTreeFromIntervals(
            negativePairsIntervals, ALL_MEMBERS)
        self.memberPaths = {
            member: [os.path.join(FACES_PATH, member, imgName) for imgName in os.listdir(
                os.path.join(FACES_PATH, member))]
            for member in ALL_MEMBERS}
        self.allPaths = set(
            [imgPath for member, imgPaths in self.memberPaths.items() for imgPath in imgPaths])
        self.memberPathsComplements = {
            member: self.allPaths - set(self.memberPaths[member]) for member in ALL_MEMBERS}

    def choosePositivePairs(self, amount):
        pairs = []
        for _ in range(amount):
            member = self.positivePairsSegmentTree.find(random.random())
            memberSamples = random.sample(self.memberPaths[member], k=2)
            pairs.append((memberSamples[0], memberSamples[1]))

        return pairs

    def chooseNegativePairs(self, amount):
        pairs = []
        for _ in range(amount):
            firstMember = self.negativePairsSegmentTree.find(random.random())
            firstMemberSample = random.sample(
                self.memberPaths[firstMember], k=1)[0]
            secondMemberSample = random.sample(
                self.memberPathsComplements[firstMember], k=1)[0]
            pairs.append((firstMemberSample, secondMemberSample))

        return pairs
