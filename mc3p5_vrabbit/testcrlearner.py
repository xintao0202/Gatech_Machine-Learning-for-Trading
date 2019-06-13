"""
Test a CR Learner in a navigation problem.  (c) 2017 Tucker Balch
"""

import numpy as np
import random as rand
import time
import math
import CRLearner as cr

# Globals/constants
MAXR = 5.0 # Max row pos
MAXC = 5.0 # Max col pos
RSTD = 1.0 # row noise
CSTD = 1.0 # col noise

# print out the map
def printmap(data):
    for col in range(0, data.shape[1]): print "-",
    print
    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row,col] == 0: # Empty space
                print " ",
            if data[row,col] == 1: # Obstacle
                print "O",
            if data[row,col] == 2: # El roboto
                print "*",
            if data[row,col] == 3: # Goal
                print "X",
            if data[row,col] == 4: # Trail
                print ".",
            if data[row,col] == 5: # Quick sand
                print "~",
            if data[row,col] == 6: # Stepped in quicksand
                print "@",
        print
    for col in range(0, data.shape[1]): print "-",
    print

# find where the robot is in the map
def getrobotpos(data):
    R = -999
    C = -999
    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row,col] == 2:
                C = col
                R = row
    if (R+C)<0:
        print "warning: start location not defined"
    return R, C

# find where the goal is in the map
def getgoalpos(data):
    R = -999
    C = -999
    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row,col] == 3:
                C = col
                R = row
    if (R+C)<0:
        print "warning: goal location not defined"
    return (R, C)

# move the robot and report reward
def movebot(data,oldpos,a):
    testr, testc = oldpos

    randomrate = 0.20 # how often do we move randomly
    quicksandreward = -100 # penalty for stepping on quicksand

    # decide if we're going to ignore the action and 
    # choose a random one instead
    if rand.uniform(0.0, 1.0) <= randomrate: # going rogue
        a = rand.randint(0,3) # choose the random direction

    # update the test location
    if a == 0: #north
        testr = testr - 1
    elif a == 1: #east
        testc = testc + 1
    elif a == 2: #south
        testr = testr + 1
    elif a == 3: #west
        testc = testc - 1

    reward = -1 # default reward is negative one
    # see if it is legal. if not, revert
    if testr < 0: # off the map
        testr, testc = oldpos
    elif testr >= data.shape[0]: # off the map
        testr, testc = oldpos
    elif testc < 0: # off the map
        testr, testc = oldpos
    elif testc >= data.shape[1]: # off the map
        testr, testc = oldpos
    elif data[testr, testc] == 1: # it is an obstacle
        testr, testc = oldpos
    elif data[testr, testc] == 5: # it is quicksand
        reward = quicksandreward
        data[testr, testc] = 6 # mark the event
    elif data[testr, testc] == 6: # it is still quicksand
        reward = quicksandreward
        data[testr, testc] = 6 # mark the event
    elif data[testr, testc] == 3:  # it is the goal
        reward = 1 # for reaching the goal

    return (testr, testc), reward #return the new, legal location

# convert the location to a single integer
# this function name is used for legacy purposes, it actually
# returns a continuous state
def discretize(pos):
    row = rand.gauss(pos[0]/MAXR, RSTD)
    col = rand.gauss(pos[1]/MAXC, CSTD)
    return np.asarray((row,col))
    # return pos[0]*10 + pos[1] # the old discrete version

def test(map, iterations, learner, verbose):
# each iteration involves one trip to the goal
    startpos = getrobotpos(map) #find where the robot starts
    goalpos = getgoalpos(map) #find where the goal is
    scores = np.zeros((iterations,1))
    for iteration in range(1,iterations+1): 
        total_reward = 0
        data = map.copy()
        robopos = startpos
        state = discretize(robopos) #convert the location to a state
        action = learner.querysetstate(state) #set the state and get first action
        count = 0
        while (robopos != goalpos) & (count<100000):

            #move to new location according to action and then get a new action
            newpos, stepreward = movebot(data,robopos,action)
            if newpos == goalpos:
                r = 1 # reward for reaching the goal
            else:
                r = stepreward # negative reward for not being at the goal
            state = discretize(newpos)
            action = learner.query(state,r)
    
            if data[robopos] != 6:
                data[robopos] = 4 # mark where we've been for map printing
            if data[newpos] != 6:
                data[newpos] = 2 # move to new location
            robopos = newpos # update the location
            #if verbose: time.sleep(1)
            total_reward += stepreward
            count = count + 1
        if count == 100000:
            print "timeout"
        if verbose: printmap(data)
        if verbose: print iteration, total_reward
        scores[iteration-1,0] = total_reward
    return np.median(scores)

# run the code to test a learner
def test_code():

    global MAXR 
    global MAXC
    global RSTD
    global CSTD

    verbose = False # print lots of debug stuff if True

    # read in the map
    filename = 'testworlds/vr_02_011.csv'
    inf = open(filename)
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
    originalmap = data.copy() #make a copy so we can revert to the original map later
    MAXR = float(data.shape[0] - 1) # Max row position
    MAXC = float(data.shape[1] - 1) # Max col position
    RSTD = (MAXR + 1)/1000.0 # stdev to use for location noise
    CSTD = (MAXC + 1)/1000.0 # stdev to use for location noise

    if verbose: 
        printmap(data)

    rand.seed(5)

    ######## run test ########
    learner = cr.CRLearner(num_dimensions=2,\
        num_actions = 4, \
        verbose = verbose) #initialize the learner
    iterations = 500
    total_reward = test(data, iterations, learner, verbose)
    print "results for", filename
    print "shape", data.shape
    print "iterations", iterations 
    print "median total_reward" , total_reward

if __name__=="__main__":
    test_code()
