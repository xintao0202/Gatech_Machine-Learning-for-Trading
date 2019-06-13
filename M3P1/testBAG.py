

import numpy as np
import math
import BagLearner as bl
import RTLearner as rt
import matplotlib.pyplot as plt
#%matplotlib inline

if __name__=="__main__":
    inf = open('Data/Istanbul.csv')
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = math.floor(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    Xtrain = data[:train_rows,0:-1]
    Ytrain = data[:train_rows,-1]
    Xtest = data[train_rows:,0:-1]
    Ytest = data[train_rows:,-1]

#     learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)
#     learner.addEvidence(Xtrain, Ytrain) # train it
#     Y = learner.query(Xtrain)

    Correlation_train = []
    Correlation_test = []
    RMSE_train = []
    RMSE_test = []

    for n in range(1, 100):
    # create a learner and train it
        learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":10}, bags = n, boost = False, verbose = False)
        learner.addEvidence(Xtrain, Ytrain) # train it


    # evaluate in sample
        Yin = learner.query(Xtrain) # get the predictions
        rmse_train = math.sqrt(((Ytrain - Yin) ** 2).sum()/Ytrain.shape[0])
        RMSE_train.append(rmse_train)

        #print "In sample results"
        #print "RMSE: ", rmse
        #c = np.corrcoef(Y, y=Ytrain)
        #print "in corr: ", c[0,1]
        #Correlation_train.append(c[0,1])

    # evaluate out of sample
        Yout = learner.query(Xtest) # get the predictions
        rmse_test = math.sqrt(((Ytest - Yout) ** 2).sum()/Ytest.shape[0])
        RMSE_test.append(rmse_test)

        # print "Out of sample results"
        # print "RMSE: ", rmse
        #c = np.corrcoef(Y, y=Ytest)
        #print "out corr: ", c[0,1]
        #Correlation_test.append(c[0,1])
        # print Correlation_train
        # print Correlation_test

    # plot for Correlation
#     plt.plot(Correlation_train, label='Ctrain', color='g')
#     plt.plot(Correlation_test, label='Ctest', color='m')

#     plt.title('Correlation for RTLearner with different leaf size')
#     plt.xlabel('leaf_size')
#     plt.ylabel('Correlation')
#     plt.legend(loc='upper right')
#     plt.show()

    # plot for RMSE
    plt.plot(RMSE_train, label='train', color='g')
    plt.plot(RMSE_test, label='test', color='m')

    plt.title('RMSE for BagLearner with different number of bags')
    plt.xlabel('number of bags')
    plt.ylabel('RMSE')
    plt.legend(loc='upper right')
    plt.show()

