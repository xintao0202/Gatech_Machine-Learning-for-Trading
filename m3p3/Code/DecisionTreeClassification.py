import numpy as np
from scipy import stats
import random
import math


class DTclass(object):
    def __init__(self, leaf_size=5, verbose=False):
        self.leaf_size = leaf_size  # move along, these aren't the drones you're looking for

    def author(self):
        return 'xtao41'  # replace tb34 with your Georgia Tech username

    def build_RT(self, data):
        dataX = data[:, 0:-1]
        dataY = data[:, -1]
        splitted = 0
        if data.shape[0] <= self.leaf_size: return np.array([['leaf', stats.mode(dataY,axis=0)[0][0], 'NA', 'NA']])
        if all(x == dataY[0] for x in dataY):
            return np.array([['leaf', dataY[0], 'NA', 'NA']])
        else:
            rand_feature_index = random.randrange(0, dataX.shape[1], 1)
            # repeat ing the procedure until a definitive split is obtained. If no such split is found after 10 tries, the node is declared to be terminal
            for n in range(1, 10):
                rand_row1_index = random.randrange(0, dataX.shape[0], 1)
                rand_row2_index = random.randrange(0, dataX.shape[0], 1)
                SplitVal = (dataX[rand_row1_index, rand_feature_index] + dataX[rand_row2_index, rand_feature_index]) / 2
                left = data[data[:, rand_feature_index] <= SplitVal]
                right = data[data[:, rand_feature_index] > SplitVal]
                if (left.shape[0] * right.shape[0] != 0):
                    LeftTree = self.build_RT(left)
                    RightTree = self.build_RT(right)
                    Root = np.array([[rand_feature_index, SplitVal, 1, LeftTree.shape[0] + 1]])
                    splitted = 1
                    return np.concatenate((Root, LeftTree, RightTree), axis=0)
                else:
                    continue
            if (splitted == 0):
                return np.array([['leaf', stats.mode(dataY,axis=0)[0][0], 'NA', 'NA']])

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        data = np.column_stack([dataX, dataY])
        self.RT = self.build_RT(data)

    def traverse(self, row, tree):
        #print tree[0][0]
        if tree[0][0] == 'leaf':
            return tree[0][1]
            # print row[int(float(tree[0, 0]))],tree[0, 1],'compare'
        if (float(row[int(float(tree[0][0]))]) <= float(tree[0][1])):
            # print'left'
            LeftTree = tree[int(float(tree[0][2])):int(float(tree[0][3]))]
            return self.traverse(row, LeftTree)
        elif (float(row[int(float(tree[0][0]))]) > float(tree[0][1])):
            # print 'right'
            RightTree = tree[int(float(tree[0][3])):tree.shape[0]]
            return self.traverse(row, RightTree)

    def query(self, Xtest):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        predictVal = []
        for row in Xtest:
            predictVal.append(self.traverse(row, self.RT))
            predict = np.asarray(predictVal).astype(np.float)
        return predict


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"