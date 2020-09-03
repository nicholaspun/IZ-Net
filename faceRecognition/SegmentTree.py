class SegmentTree:
  '''
  A specialized Segment Tree that assumes we are given intervals of the form: 
  (-\infty, a_1), [a_1, a_2), ..., [a_n, \infty)
  '''
  def __init__(self, key, leftChild, rightChild):
    self.key = key
    self.leftChild = leftChild
    self.rightChild = rightChild

  '''
  Results in possibly poor naming, but given the assumptions from the ctor, 
  intervals can be given as an array of numbers.
  '''
  @staticmethod
  def createSegmentTreeFromIntervals(intervals, leafNames):
    if len(intervals) == 3:
      return SegmentTree(intervals[1], leafNames[0], leafNames[1])

    midpoint = len(intervals) // 2
    if (len(intervals) % 2 == 0):
      key = (intervals[midpoint] + intervals[midpoint - 1]) / 2
      return SegmentTree(
        key,
        SegmentTree.createSegmentTreeFromIntervals(intervals[:midpoint] + [key], leafNames[:midpoint]),
        SegmentTree.createSegmentTreeFromIntervals([key] + intervals[midpoint:], leafNames[midpoint - 1:])
      )
    else:
      return SegmentTree(
        intervals[midpoint],
        SegmentTree.createSegmentTreeFromIntervals(intervals[:midpoint + 1], leafNames[:midpoint]),
        SegmentTree.createSegmentTreeFromIntervals(intervals[midpoint:], leafNames[midpoint:])
      )

  @staticmethod
  def printTree(tree):
    print(SegmentTree.__printTree(tree, 0))

  # ignore the style of this function :)
  @staticmethod
  def __printTree(tree, level):
    if isinstance(tree, str):
      return tree

    res = "SegmentTree({}, {}{}{}, {}{}{})".format(
      tree.key,
      "" if isinstance(tree.leftChild, str) else "\n",
      "" if isinstance(tree.leftChild, str) else "  " * (level + 1),
      SegmentTree.__printTree(tree.leftChild, level + 1),
      "" if isinstance(tree.rightChild, str) else "\n",
      "" if isinstance(tree.rightChild, str) else "  " * (level + 1),
      SegmentTree.__printTree(tree.rightChild, level + 1),
      "  " * level
    )

    return res

  def find(self, key):
    if isinstance(self.leftChild, str) and isinstance(self.rightChild, str):
      return self.leftChild if key < self.key else self.rightChild
    elif isinstance(self.leftChild, str) and key < self.key:
      return self.leftChild
    elif isinstance(self.rightChild, str) and key >= self.key:
      return self.rightChild
    else:  
      return self.leftChild.find(key) if key < self.key else self.rightChild.find(key)
  

# Testing!
# SegmentTree.printTree(SegmentTree.createSegmentTreeFromIntervals([1,2,3,4,5,6,7,8,9, 10,11,12], ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']))
# SegmentTree.printTree(SegmentTree.createSegmentTreeFromIntervals([1,2,3], ['a', 'b']))
# SegmentTree.printTree(SegmentTree.createSegmentTreeFromIntervals([1,2,3,4], ['a', 'b', 'c']))
# SegmentTree.printTree(SegmentTree.createSegmentTreeFromIntervals([1,2,3,4,5], ['a', 'b', 'c', 'd']))

# testTree = SegmentTree.createSegmentTreeFromIntervals([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'])
# print(testTree.find(5) == 'e')
# print(testTree.find(0) == 'a')
# print(testTree.find(13) == 'k')
# print(testTree.find(6.5) == 'f')
