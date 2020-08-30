class Distribution:
    @staticmethod
    def normalizeDistribution(distribution):
        return Distribution.__normalizeDistribution(distribution, 0)[0]

    @staticmethod
    def __normalizeDistribution(distribution, runningSum):
        if len(distribution) == 0:
            return [], runningSum
        rest, theSum = Distribution.__normalizeDistribution(distribution[1:], runningSum + distribution[0])
        return [distribution[0]/theSum] + rest, theSum

    @staticmethod
    def distributionToIntervals(distribution):
        return Distribution.__distributionToIntervals([0] + distribution)

    @staticmethod
    def __distributionToIntervals(distribution):
        if len(distribution) == 0 or len(distribution) == 1:
            return distribution
        if len(distribution) == 2:
            return [distribution[0], distribution[0] + distribution[1]]

        return [distribution[0]] + Distribution.__distributionToIntervals([distribution[0] + distribution[1]] + distribution[2:])

# Testing!
# testDistribution = Distribution.normalizeDistribution([2,3,1,4])
# print(testDistribution)
# print(Distribution.distributionToIntervals(testDistribution))


