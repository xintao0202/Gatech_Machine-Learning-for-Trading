import numpy as np
import math
import RTLearner as rt
import BagLearner as bl
import matplotlib.pyplot as plt
#%matplotlib inline

if __name__=="__main__":
    inf = open('Data/Istanbul.csv')
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    #print data
    #print data[:,-1]

    # compute how much of the data is training and testing
    #train_rows = int(round(0.6* data.shape[0]))
    #print train_rows
    #test_rows = data.shape[0] - train_rows

#     # separate out training and testing data
    #Xtrain = data[:train_rows,0:-1]

    #print Xtrain.shape[0]
    #Ytrain = data[:train_rows,-1]
    #print Ytrain
    #print Xtrain.shape
    #print Ytrain.shape

    # http://stackoverflow.com/questions/4158388/numpy-concatenating-multidimensional-and-unidimensional-arrays
    #newdata = np.column_stack((Xtrain,Ytrain))
    #print newdata.shape


    #Xtest = data[train_rows:,0:-1]
    #Ytest = data[train_rows:,-1]

#     learner = rt.RTLearner(leaf_size=1)
#     learner.addEvidence(Xtrain, Ytrain) # train it
#     #Y = learner.query(Xtest)
#     Y = learner.query(Xtrain)
    #print tree[0][0]
    #print tree.shape[1]
    #print len(tree)
    #print tree2

    #print learner.author()


    #Correlation_train = []
    #Correlation_test = []
    RMSE_train0 = []
    RMSE_test0 = []

    #Correlation_train = []
    #Correlation_test = []
    RMSE_train1 = []
    RMSE_test1 = []

    # create a learner and train it
    for k in range (1, 50):

        # shuffle data
        np.random.shuffle(data)

        train_rows = int(round(0.6* data.shape[0]))
        test_rows = data.shape[0] - train_rows
        Xtrain = data[:train_rows,0:-1]
        Ytrain = data[:train_rows,-1]
        Xtest = data[train_rows:,0:-1]
        Ytest = data[train_rows:,-1]

        # learner no bagging
        learner0 = rt.RTLearner(leaf_size=k)
        learner0.addEvidence(Xtrain, Ytrain) # train it

        learner1 = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":k}, bags = 20, boost = False, verbose = False)
        learner1.addEvidence(Xtrain, Ytrain) # train it

    # evaluate in sample for learner0
        Ytrain_0 = learner0.query(Xtrain) # get the predictions
        rmse_train0 = math.sqrt(((Ytrain - Ytrain_0) ** 2).sum()/Ytrain.shape[0])
        RMSE_train0.append(rmse_train0)

        # evaluate in sample for learner1
        Ytrain_1 = learner1.query(Xtrain) # get the predictions
        rmse_train1 = math.sqrt(((Ytrain - Ytrain_1) ** 2).sum()/Ytrain.shape[0])
        RMSE_train1.append(rmse_train1)

        #print "In sample results"
        #print "RMSE: ", rmse
        #c = np.corrcoef(Y, y=Ytrain)
        #print "in corr: ", c[0,1]
        #Correlation_train.append(c[0,1])

    # evaluate out of sample for learner 0
        Ytest_0 = learner0.query(Xtest) # get the predictions
        rmse_test0 = math.sqrt(((Ytest - Ytest_0) ** 2).sum()/Ytest.shape[0])
        RMSE_test0.append(rmse_test0)

        # evaluate out of sample for learner 1
        Ytest_1 = learner1.query(Xtest) # get the predictions
        rmse_test1 = math.sqrt(((Ytest - Ytest_1) ** 2).sum()/Ytest.shape[0])
        RMSE_test1.append(rmse_test1)

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
    plt.plot(RMSE_train0, label='train without bagging', linestyle='--', color='g')
    plt.plot(RMSE_train1, label='train with bagging', color='g')
    plt.plot(RMSE_test0, label='test without bagging', linewidth=2, linestyle='--', color='m')
    plt.plot(RMSE_test1, label='test with bagging', linewidth=2, color='m')
    plt.axvline(3, color='red', linestyle='--')

    plt.title('RMSE for RTLearner vs BagLearner with different leaf size')
    plt.xlabel('leaf_size')
    plt.ylabel('RMSE')
    plt.legend(loc='lower right')
    plt.show()

