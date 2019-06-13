"""
Template for implementing QLearner  (c) 2015 Tucker Balch
CS 7646 project 5
QLnearner implementation
By Zhongtao Yang
"""
import numpy as np
import random as rand
class QLearner(object):
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.allActions = range(self.num_actions)
        self.count = 0
        self.prevCount = 0
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.minDyna = 30
        # initialize the Q Table
        self.Q = np.zeros((self.num_states,self.num_actions),dtype=float)
        # initialize the R,T Table
        self.R = [[0.0 for i in self.allActions] for i in xrange(self.num_states)]
        self.T = [[[(1.0 / self.num_states) for i in xrange(self.num_states)] for j in xrange(self.num_actions)] for k in xrange(self.num_states)]
        # print np.shape(self.R)
        # print np.shape(self.Q)
        # Initialize the Tc
        self.Tc = [[[0.0000000001 for i in xrange(self.num_states)] for j in xrange(self.num_actions)] for k in xrange(self.num_states)]
        # print np.shape(self.T)
        # print np.shape(self.Tc)
        
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        # print 'current count:',self.count
        # Here I used a very smart adaptive method to update the dynaQ iteration, which shows a very good performance and increases the speed and accuracy. Most importantly, it lowers the complexity when the Q is optimized during helucination
        if self.dyna != 0:
            if (self.prevCount >= self.count * 0.8): # 0.8 coefficient is used to get rid of tiny performance variation, which may cause the increase of dyna value, and this is not desirable
                self.dyna = self.dyna - 10
            else:
                self.dyna =self.dyna + 3
            if self.dyna <= self.minDyna:
                self.dyna = self.minDyna
            self.prevCount = self.count
            self.count = 0
        self.s = s
        action = (np.int(rand.random()*100)) % self.num_actions
        # action = rand.randint(0, self.num_actions-1)
        self.a = action
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        prevState = self.s
        prevAction = self.a
        newState = s_prime
        self.count = self.count + 1
        rand.seed(np.random.randint(50))
        # decide what action to take in the new state
        if rand.random() < self.rar: #Random exploration
            action = rand.randint(0, self.num_actions - 1)
            # action = (np.int(rand.random()*111)) % self.num_actions
        else: # Or choose the action best on Q table at current state
            possibleAction = self.Q[newState]
            maxQVal = max(possibleAction)
            allActions = np.where(possibleAction == maxQVal)
            # ind = (np.int(rand.random()*222)) % allActions[0].shape[0]
            # action = rand.choice(allActions[0])
            action = allActions[0][0]
        # prevQ = self.Q[prevState][prevAction]
 #        newQ = self.Q[newState][action]
        # update the Q based on new state and new action made from prev state
        self.Q[prevState][prevAction] = (1.0 - self.alpha) * self.Q[prevState][prevAction] + self.alpha * (self.gamma * self.Q[newState][action] + r)
        retAction = action

        # Considering dyna halucilation process: ================
        if self.dyna != 0:
            self.Tc[prevState][prevAction][newState] = self.Tc[prevState][prevAction][newState] + 1
            self.R[prevState][prevAction] = (1.0 - self.alpha) * self.R[prevState][prevAction] + self.alpha * r
            for i in range(0, self.num_states):            
                self.T[prevState][prevAction][i] = self.Tc[prevState][prevAction][i] / sum(self.Tc[prevState][prevAction])
            for i in range(0, self.dyna):
                # s = rand.randint(0,self.num_states-1)
                ss = np.int(rand.random()*333) % self.num_states
                # a = rand.randint(0, self.num_actions-1)
                aa = np.int(rand.random()*444) % self.num_actions
                randValue = rand.random()
                cumProb = 0.0
                sp_temp = 0
                #use cumulative probability to select the sp in hulacination (using random generated sp is not a good idea, it takes more time and steps to go to destination)
                scaleCoef = 5
                while cumProb <= randValue:
                    cumProb += self.T[ss][aa][sp_temp]*scaleCoef
                    sp = sp_temp
                    sp_temp = sp_temp + 1
                rr = self.R[ss][aa]
                tempPossibleAction = self.Q[sp]
                tempMaxQVal = max(tempPossibleAction)
                allActions = np.where(tempPossibleAction == tempMaxQVal)
                ind = (np.int(rand.random()*1234)) % allActions[0].shape[0]
                newAction = allActions[0][ind]
                self.Q[ss][aa] = (1.0 - self.alpha) * self.Q[ss][aa] + self.alpha * (rr + self.gamma * self.Q[sp][newAction])
        # ===========================================    
        
        # decay the rar, otherwise it will always try explore randomly with high possibility
        self.s = s_prime
        self.a = retAction
        self.rar = self.rar * self.radr
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return retAction

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
