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
import os
import numpy as np
import numpy.random as rn
from itertools import product
import numpy as np
import time
import tensorflow as tf
import CUtility
class nonLinearOnlineGradientIRL(object):
    def __init__(self, n_actions, n_states, transitionProbability,
                 featureMatrix,discount,learning_rate,trajectories,epochs,method,approx_level,b_level,rewardStructure):
        self.n_actions =n_actions
        self.n_states = n_states
        self.discount=discount
        self.approx_level=approx_level
        self.transitionProbability=transitionProbability
        self.featureMatrix=featureMatrix
        self.method=method #"pnorm" or "gsoft"
        self.trajectory=trajectories
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.featureMatrix=featureMatrix-np.min(featureMatrix)
        self.rewardStructure=rewardStructure; #[self.featureMatrix.shape[1],10,5,1];
        self.b_level=b_level

    def gradientIterationIRL(self):
        tf.reset_default_graph()
        n_states, d_states = self.featureMatrix.shape
        b_level=self.b_level

        # Initialise paras.
        n_layers=len(self.rewardStructure);
        featureMatrix=tf.constant(self.featureMatrix,dtype=tf.float32);
        outputs=[featureMatrix]
        input_dicts=[]
        for i in range(1,n_layers):
            n_hidden=self.rewardStructure[i]
            n_visible=self.rewardStructure[i-1]
            W=tf.placeholder(tf.float32,shape=(n_visible,n_hidden))
            b=tf.placeholder(tf.float32,shape=(n_hidden,))
            if(i<n_layers):
               output=tf.nn.tanh(tf.matmul(outputs[-1], W) + b)
            else:
               output=tf.matmul(outputs[-1], W) + b
            input_dicts.append(W)
            input_dicts.append(b)
            outputs.append(output)

        s_reward=tf.reshape(outputs[-1], [-1])
        s_gradientMatrix=[]
        for i in range(n_states):
            gradients=tf.gradients(s_reward[i],input_dicts)
            gradient_list=[]
            for gradient in gradients:
                gradient_list.append(tf.reshape(gradient,[-1]))
            gradientList=tf.concat(gradient_list,axis=0)
            s_gradientMatrix.append(gradientList)
        s_gradientMatrix=tf.stack(s_gradientMatrix,axis=0)

        # initialvalue
        dict_values=[]
        theta=np.array([])
        for i in range(1,n_layers):
            n_hidden=self.rewardStructure[i]
            n_visible=self.rewardStructure[i-1]
            # randomly intialize W n_hidden*n_visible, and b, n_hidden
            W_value = np.asarray(
                          np.random.uniform(
                                              low=-1 * np.sqrt(6. / (n_hidden + n_visible)),
                                              high=1 * np.sqrt(6. / (n_hidden + n_visible)),
                                              size=(n_visible, n_hidden)
                                          ),
                          dtype=np.float32
                      )
            b_value =  np.zeros(n_hidden,dtype=np.float32)
            dict_values.append(W_value)
            dict_values.append(b_value)
            theta=np.concatenate([theta,W_value.flatten()])
            theta=np.concatenate([theta,b_value])

        shape_list=[para.shape for para in dict_values]
        log_sequence=[]
        cor_sequence=[]
        # Gradient descent on alpha.
        log_likelihood=0;
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        for s,a,r in self.trajectory:
            start_time=time.time()
            # Compute the rewards and the reward gradients over every states
            # via deep NN.
            reward, gradientMatrix =sess.run([s_reward,s_gradientMatrix],feed_dict={i: d for i, d in zip(input_dicts,
                                                                                    dict_values)})
            reward=reward.astype(np.float64)
            gradientMatrix=gradientMatrix.astype(np.float64)
            if(self.method=="pnorm"):
                Values, QValues=self.pnorm_value_iteration(reward);
                V_Gradients, Q_Gradients=self.pnorm_gradient_iteration(reward,gradientMatrix, Values, QValues);

            if(self.method=="gsoft"):
                Values, QValues=self.gsoft_value_iteration(reward);
                V_Gradients, Q_Gradients=self.gsoft_gradient_iteration(reward, gradientMatrix,Values, QValues);

            if(np.isnan(Values).any() or np.isnan(QValues).any()):
                print "Value Diverge: adjust the approximation level or the discount factor"
                return [],[]

            if(np.isnan(V_Gradients).any() or np.isnan(Q_Gradients).any()):
                print "Gradient Diverge: adjust the approximation level or the discount factor"
                return [],[]
             # compute gradient
            grad=np.zeros((theta.shape[0],));
            # sum the gradients of all state-action pairs
            d_sa=Q_Gradients[s,a]
            expected_sum=np.zeros((theta.shape[0],))
            for a1 in range(self.n_actions):
                expected_sum=expected_sum+Q_Gradients[s,a1]/np.sum(np.exp(b_level*(QValues[s,:]-QValues[s,a1])))
            d_sa=b_level*d_sa-b_level*expected_sum
            grad=grad+d_sa;

            # compte likelihood
            log_likelihood=log_likelihood-np.log(np.sum(np.exp(b_level*(QValues[s,:]-QValues[s,a]))))

            # gradient ascent
            theta=theta+self.learning_rate*grad
            # update parameters
            count=0
            dict_values=[]
            for i_para in range(len(shape_list)):
                shape=shape_list[i_para]
                if(len(shape)==1):
                   dict_values.append(theta[count:count+shape[0]].reshape((shape[0],)))
                   count+=shape[0]
                if(len(shape)==2):
                   dict_values.append(theta[count:count+shape[0]*shape[1]].reshape((shape[0],shape[1])))
                   count+=shape[0]*shape[1]
            log_sequence.append(log_likelihood)
        return [log_sequence]

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
                        value_update[j]=value_update[j]+self.transitionProbability[i,j,k]*(reward[i]+self.discount*V[k])
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
                Q[i,j]=reward[i]
                for k in range(self.n_states):
                    Q[i,j] = Q[i,j]+self.discount*self.transitionProbability[i,j,k]*V[k]
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
    def pnorm_gradient_iteration(self, reward,gradientMatrix, Values, QValues):

        # Value Gradient Iteration
        '''
        print "python functions"
        reward_para_shape=gradientMatrix.shape[1]
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
                        d_Q+=self.transitionProbability[i,j,k]*(gradientMatrix[i,:]+self.discount*V_Gradients[k,:])
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
                    Q_Gradients[i,j,:]=Q_Gradients[i,j,:]+self.transitionProbability[i,j,k]*(self.rewardGradient(i)+self.discount*V_Gradients[k])
        return V_Gradients, Q_Gradients
        '''
        #print "c functions"
        transition_probability=self.transitionProbability.copy(order='C')
        reward=reward.copy(order='C')
        QValues=QValues.copy(order='C')
        Values=Values.copy(order='C')
        gradientMatrix=gradientMatrix.copy(order='C')
        n_features=gradientMatrix.shape[1]
        V_Gradients,Q_Gradients=CUtility.i_pnorm_G_iteration(self.n_states,self.n_actions,n_features,gradientMatrix,transition_probability,reward,self.discount,self.approx_level,Values,QValues)
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
                        value_update[j]=value_update[j]+self.transitionProbability[i,j,k]*(reward[i]+self.discount*V[k]);
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
                Q[i,j]=reward[i]
                for k in range(self.n_states):
                    Q[i,j] = Q[i,j]+self.discount*self.transitionProbability[i,j,k]*V[k]
        # Further processing, may be unnecessary
        # Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        # Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
        '''
        transition_probability=self.transitionProbability.copy(order='C')
        reward=reward.copy(order='C')
        value,Qvalue=CUtility.i_gsoft_V_iteration(self.n_states, self.n_actions, transition_probability, reward,
                        self.discount, self.approx_level)
        return value, Qvalue

    def gsoft_gradient_iteration(self, reward,gradientMatrix, Values, QValues):

        # Value Gradient Iteration
        '''
        print "python functions"
        reward_para_shape=gradientMatrix.shape[1]
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
                        d_Q+=self.transitionProbability[i,j,k]*(gradientMatrix[i]+self.discount*V_Gradients[k,:])
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
                    Q_Gradients[i,j,:]=Q_Gradients[i,j,:]+self.transitionProbability[i,j,k]*(self.rewardGradient(i)+self.discount*V_Gradients[k])
        '''
        #print "c functions"
        transition_probability=self.transitionProbability.copy(order='C')
        reward=reward.copy(order='C')
        QValues=QValues.copy(order='C')
        Values=Values.copy(order='C')
        gradientMatrix=gradientMatrix.copy(order='C')
        n_features=gradientMatrix.shape[1]
        V_G,Q_G=CUtility.i_gsoft_G_iteration(self.n_states,self.n_actions,n_features,gradientMatrix,transition_probability,reward,self.discount,self.approx_level,Values,QValues)
        return V_G, Q_G


