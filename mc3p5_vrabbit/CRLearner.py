"""
Template for implementing CRLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class CRLearner(object):

    def __init__(self, \
        num_dimensions = 2, \
        num_actions = 4, \
        verbose = False):

        self.num_dimensions = num_dimensions
        self.num_actions = num_actions
		self.hidden=num_dimensions+num_actions
		self.Q = np.zeros((self.num_dimensions,self.num_actions),dtype=float)
		self.W1=np.randome.randn(self.num_dimensions,self.hidden)
		self.W2=np.random.randn(self.hidden,self.action)
		self.verbose = verbose
	
	
    def author(self):
        return 'xtao41' # replace tb34 with your Georgia Tech username.
	
	def sigmoid(self.z):
	#activate scalar
		return 1/(1+np.exp(-z))
	
	def forward(self,X):
		#inputs of network
		self.z2=np.dot(X, self.W1)
		self.a2=self.sigmoid(self.z2)
		self.z3=np.dot(self.a2,self.W2)
		yHat=self.sigmoid(self.z3)
		return yHat
	
	def costFunctionReduction (self, X, Y):
		self.yHat=self.forward(X)
		delta3=np.multiply(-(Y-self.yHat), self.sigmoid(self.z3))
		delta2=np.dot(delta3.self.W2.T)*self.sigmoid(self,z2)
		W1_delta=np.dot(X.T,delta2)
		W2_delta=np.dot(self.a2.T,delta3)
		return W1_delta,W2_delta
	
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = rand.randint(0, self.num_actions-1)
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
		
		#action = rand.randint(0, self.num_actions-1)
		possibleAction = self.Q[newState]
        maxQVal = max(possibleAction)
        allActions = np.where(possibleAction == maxQVal)
		action = allActions[0][0]
		self.Q[prevState][prevAction] = (1.0 - self.alpha) * self.Q[prevState][prevAction] + self.alpha * (self.gamma * self.Q[newState][action] + r)
        retAction = action
		self.s = s_prime
        self.a = retAction
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

if __name__=="__main__":
    print "Yann Lecun was here"
