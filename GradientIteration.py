#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Gradient-Iteration Methods for Inverse Reinforcement Learning
This code will implement inverse reinforcement learning by considering each path as a joint
distribution of all state-action pairs of the path. The probability of a state-action pair is an
exponential function on Q value.

The gradient of Q value function is approximated by replacing the max-operator with p-norm
approximation and generalized soft-max approximation. Thus we can use gradient iteration to estimate
the gradient of a state-action pair.
'''
import numpy as np
import numpy.random as rn
from itertools import product
import numpy as np
import time
import CUtility
class gradientIRL(object):
    def __init__(self, n_actions, n_states, transitionProbability,
                 featureMatrix,discount,learning_rate,trajectories,epochs,method,approx_level,
                 confidence_level):
        self.n_actions =n_actions
        self.n_states = n_states
        self.discount=discount
        self.approx_level=approx_level
        self.b_level=confidence_level
        self.transitionProbability=transitionProbability
        self.featureMatrix=featureMatrix
        self.method=method #"pnorm" or "gsoft"
        self.trajectories=trajectories
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.featureMatrix=featureMatrix

    def gradientIterationIRL(self):
        print "gradient iteration irl with ", self.method
        n_states, d_states = self.featureMatrix.shape

        # Initialise weights.
        alpha = rn.random(size=(d_states,))

        # Gradient descent on alpha.
        log_likelihood=0;
        #for i in range(self.epochs):
        i=0
        prev_log=-float('inf');
        count_decrease=0;
        while(True):
            i=i+1
            if(i>self.epochs):
                print "iteration maximum", log_likelihood,np.linalg.norm(grad)
                break;

            start_time=time.time()
            # print("i: {}".format(i))
            reward = self.featureMatrix.dot(alpha)

           # Values and Gradients
            if(self.method=="pnorm"):
                Values, QValues=self.pnorm_value_iteration(reward);
                V_Gradients, Q_Gradients=self.pnorm_gradient_iteration(reward, Values, QValues);

            if(self.method=="gsoft"):
                Values, QValues=self.gsoft_value_iteration(reward);
                V_Gradients, Q_Gradients=self.gsoft_gradient_iteration(reward, Values, QValues);

            if(np.isnan(Values).any() or np.isnan(QValues).any()):
                print "Value Diverge: adjust the approximation level or the discount factor"
                return [],[]

            if(np.isnan(V_Gradients).any() or np.isnan(Q_Gradients).any()):
                print "Gradient Diverge: adjust the approximation level or the discount factor"
                return [],[]

            # compute gradient
            grad=np.zeros((d_states,));
            # sum the gradients of all state-action pairs
            for trajectory in self.trajectories:
                for s,a,r in trajectory:
                    d_sa=self.b_level*Q_Gradients[s,a]
                    for a1 in range(self.n_actions):
                        d_sa=d_sa-self.b_level*Q_Gradients[s,a1]/np.sum(np.exp(self.b_level*(QValues[s,:]-QValues[s,a1])))
                    grad=grad+d_sa;
            alpha=alpha+self.learning_rate*(grad-0.001*np.linalg.norm(alpha))

            # compte likelihood
            log_likelihood=0;
            for trajectory in self.trajectories:
                for s,a,r in trajectory:
                    log_likelihood=log_likelihood-np.log(np.sum(np.exp(self.b_level*(QValues[s,:]-QValues[s,a]))))
        return self.featureMatrix.dot(alpha).reshape((self.n_states,)), log_likelihood



    # In this application, a reward is a linear function, thus its gradient with respect to the
    # parameter is its feature value. However, this can be extended to other functions, like dnn,
    # with a different reward gradient.
    def rewardGradient(self,state_index):
        return self.featureMatrix[state_index,:]
    # approximate the max function with different orders of p-norm
    def pnorm_max(self,approx_level,value_set):
        return np.sum(np.abs(value_set)**approx_level)**(1./approx_level)

    def pnorm_value_iteration(self,reward):

        # Value iteration to find the optimal value, with the approximated max function
        '''
        print "python functions"
        start_time=time.time()
        V= np.zeros((self.n_states, 1))
        diff = np.ones((self.n_states,))
        while (diff > 1e-4).any():  # Iterate until convergence.
            new_V = np.zeros((self.n_states,1))
            for i in range(self.n_states):
                value_update=np.zeros((self.n_actions,1));
                for j in range(self.n_actions):
                    for k in range(self.n_states):
                        value_update[j]=value_update[j]+self.transitionProbability[i,j,k]*(reward[k]+self.discount*V[k])
                new_V[i]=self.pnorm_max(self.approx_level,value_update)
            # check diverge or converge
            # if not converge
            #new_V = (new_V - new_V.mean())/new_V.std()
            diff = abs(V - new_V)
            #print "value converge", np.min(diff)
            V = new_V
        Q = np.zeros((self.n_states, self.n_actions))
        for i in range(self.n_states):
            for j in range(self.n_actions):
                for k in range(self.n_states):
                    Q[i,j] = Q[i,j]+self.transitionProbability[i,j,k]*(reward[k]+self.discount*V[k])
        return V,Q
        print "time", time.time()-start_time

        '''
        start_time=time.time()
        # C Functions
        #print "c functions"
        transition_probability=self.transitionProbability.copy(order='C')
        reward=reward.copy(order='C')
        value,Qvalue=CUtility.i_pnorm_V_iteration(self.n_states, self.n_actions, transition_probability, reward,
                        self.discount, self.approx_level)
        #print "time", time.time()-start_time
        return value, Qvalue
        #'''
    def pnorm_gradient_iteration(self, reward, Values, QValues):

        # Value Gradient Iteration
        '''
        print "python functions"
        reward_para_shape=self.featureMatrix.shape[1]
        V_Gradients= np.zeros((self.n_states, reward_para_shape))
        diff = np.ones((self.n_states,reward_para_shape))
        # Pre compute several variables to speed up
        d_pnorm_vector=np.zeros((self.n_states,))
        d_summation_matrix=np.zeros((self.n_states,self.n_actions))
        for i in range(self.n_states):
            # compute the p norm derivative, d pnorm=1/k*()^(1/k-1)
            d_pnorm_vector[i]=1.0/self.approx_level*(np.sum(np.abs(QValues[i,:])**self.approx_level))**(1.0/self.approx_level-1.0)
            for j in range(self.n_actions):
                d_summation_matrix[i,j]=(self.approx_level*QValues[i,j]**(self.approx_level-1))
        n_iter=0;
        while (diff > 1e-4).any():  # Iterate until convergence.
            new_V_Gradients = np.zeros((self.n_states,reward_para_shape))
            # The last equation, dQ/dr
            for i in range(self.n_states):
                # compute the p norm derivative, d pnorm=1/k*()^(1/k-1)
                d_pnorm=d_pnorm_vector[i]
                d_sum_q=np.zeros((reward_para_shape))
                for j in range(self.n_actions):
                    # compute the summation of power derivative,
                    d_summation=d_summation_matrix[i,j]
                    d_Q=np.zeros((reward_para_shape))
                    for k in range(self.n_states):
                        # compute the derivative of Q
                        d_Q+=self.transitionProbability[i,j,k]*(self.rewardGradient(k)+self.discount*V_Gradients[k,:])
                    d_sum_q+=d_summation*d_Q
                new_V_Gradients[i,:]=d_pnorm*d_sum_q
            # check diverge or converge
            # if not converge
            #new_V_gradients = (new_V_gradients - new_V_gradients.mean())/new_V_gradients.std()
            diff = abs(V_Gradients - new_V_Gradients)
            V_Gradients = new_V_Gradients
            n_iter=n_iter+1;
        #print V_Gradients
       # Given the value gradients, find the Q value gradients.
        Q_Gradients= np.zeros((self.n_states, self.n_actions,reward_para_shape))
        for i in range(self.n_states):
            for j in range(self.n_actions):
                for k in range(self.n_states):
                    Q_Gradients[i,j,:]=Q_Gradients[i,j,:]+self.transitionProbability[i,j,k]*(self.rewardGradient(k)+self.discount*V_Gradients[k])
        return V_Gradients, Q_Gradients
        '''
        #print "c functions"
        transition_probability=self.transitionProbability.copy(order='C')
        reward=reward.copy(order='C')
        QValues=QValues.copy(order='C')
        Values=Values.copy(order='C')
        featureMatrix=self.featureMatrix.copy(order='C')
        n_features=self.featureMatrix.shape[1]
        V_Gradients, Q_Gradients=CUtility.i_pnorm_G_iteration(self.n_states,self.n_actions,n_features,featureMatrix,transition_probability,reward,self.discount,self.approx_level,Values,QValues)
        return V_Gradients, Q_Gradients
        #'''


    # approximate the max function with different orders of softmax function
    def gsoft_max(self,approx_level,value_set):
        return np.log(np.sum(np.exp(approx_level*value_set)))/approx_level
    def gsoft_value_iteration(self,reward):

        # Value iteration to find the optimal value, with the approximated max function
        '''
        print "python functions"
        V= np.zeros((self.n_states, 1))
        diff = np.ones((self.n_states,))
        while (diff > 1e-4).any():  # Iterate until convergence.
            new_V = np.zeros((self.n_states,1))
            for i in range(self.n_states):
                value_update=np.zeros((self.n_actions,1));
                for j in range(self.n_actions):
                    for k in range(self.n_states):
                        value_update[j]=value_update[j]+self.transitionProbability[i,j,k]*(reward[k]+self.discount*V[k]);
                new_V[i]=self.gsoft_max(self.approx_level,value_update)

            # check diverge or converge
            # if not converge
            #new_V = (new_V - new_V.mean())/new_V.std()
            diff = abs(V - new_V)
            #print "value converge", np.min(diff)
            V = new_V

        # Given the optimal values, find the Q values..
        Q = np.zeros((self.n_states, self.n_actions))
        for i in range(self.n_states):
            for j in range(self.n_actions):
                for k in range(self.n_states):
                    Q[i,j] = Q[i,j]+self.transitionProbability[i,j,k]*(reward[k]+V[k])
        return V,Q
        '''

        # Further processing, may be unnecessary
        # Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        # Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
        transition_probability=self.transitionProbability.copy(order='C')
        reward=reward.copy(order='C')
        value,Qvalue=CUtility.i_gsoft_V_iteration(self.n_states, self.n_actions, transition_probability, reward,
                        self.discount, self.approx_level)
        return value, Qvalue

    def gsoft_gradient_iteration(self, reward, Values, QValues):

        # Value Gradient Iteration
        '''
        print "python functions"
        reward_para_shape=self.featureMatrix.shape[1]
        V_Gradients= np.zeros((self.n_states, reward_para_shape))
        diff = np.ones((self.n_states,reward_para_shape))

        while (diff > 1e-4).any():  # Iterate until convergence.
            new_V_Gradients = np.zeros((self.n_states,reward_para_shape))

            # The last equation, dQ/dr
            for i in range(self.n_states):
                # compute the summation of all actions
                d_sum_q=np.zeros((reward_para_shape))
                for j in range(self.n_actions):
                    # compute the summation of power derivative,
                    d_softmax=np.exp(self.approx_level*QValues[i,j])/np.sum(np.exp(self.approx_level*QValues[i,:]))
                    d_Q=np.zeros((reward_para_shape))
                    for k in range(self.n_states):
                        # compute the derivative of Q
                        d_Q+=self.transitionProbability[i,j,k]*(self.rewardGradient(k)+self.discount*V_Gradients[k,:])
                    d_sum_q+=d_softmax*d_Q
                new_V_Gradients[i,:]=d_sum_q

            # check diverge or converge
            # if not converge
            #new_V_gradients = (new_V_gradients - new_V_gradients.mean())/new_V_gradients.std()
            diff = abs(V_Gradients - new_V_Gradients)
            #print "gradient converge", np.min(diff)
            V_Gradients = new_V_Gradients

        # Given the value gradients, find the Q value gradients.
        Q_Gradients= np.zeros((self.n_states, self.n_actions,reward_para_shape))
        for i in range(self.n_states):
            for j in range(self.n_actions):
                for k in range(self.n_states):
                    Q_Gradients[i,j,:]=Q_Gradients[i,j,:]+self.transitionProbability[i,j,k]*(self.rewardGradient(k)+self.discount*V_Gradients[k])
        '''
        #print "c functions"
        transition_probability=self.transitionProbability.copy(order='C')
        reward=reward.copy(order='C')
        QValues=QValues.copy(order='C')
        Values=Values.copy(order='C')
        featureMatrix=self.featureMatrix.copy(order='C')
        n_features=self.featureMatrix.shape[1]
        V_G, Q_G=CUtility.i_gsoft_G_iteration(self.n_states,self.n_actions,n_features,featureMatrix,transition_probability,reward,self.discount,self.approx_level,Values,QValues)
        return V_G, Q_G


