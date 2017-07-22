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
import cython
from itertools import product
import numpy as np
cimport numpy as np
import time

cdef extern void pnorm_V_iteration(int n_states, int n_actions, double* transition_probabilities, double* reward,
                   double discount, double threshold, double approx_level, double*value,
                               double*new_value, double*value_update, double*Qvalue);
cdef extern void pnorm_G_iteration(int n_states, int n_actions, int n_features, double*featureMatrix, double*
                       transition_probabilities, double* reward, double discount, double threshold,
                                   double approx_level, double*Values, double*Qvalues,
                                   double*V_Gradients,double*Q_Gradients,double*d_pnorm_vector,double*d_summation_matrix,double*new_V_Gradients,double*d_sum_q)
cdef extern void gsoft_V_iteration(int n_states, int n_actions, double* transition_probabilities, double* reward,
                   double discount, double threshold, double approx_level, double*value,
                               double*new_value, double*value_update, double*Qvalue);
cdef extern void gsoft_G_iteration(int n_states, int n_actions, int n_features, double*featureMatrix, double*
                       transition_probabilities, double* reward, double discount, double threshold,
                                   double approx_level, double*Values, double*Qvalues,
                                   double*V_Gradients,double*Q_Gradients,double*d_pnorm_vector,double*d_summation_matrix,double*new_V_Gradients,double*d_sum_q)


# Value Iteration to compute value function
def i_pnorm_V_iteration(n_states, n_actions,np.ndarray[double, ndim=3,mode="c"]
                    transition_probabilities,np.ndarray[double, ndim=1,mode="c"] reward, discount, approx_level):
    cdef int num_states=n_states;
    cdef int num_actions=n_actions;
    cdef double discount_v=discount;
    cdef double threshold_v=0.0001;
    cdef double approx_level_v=approx_level;
    cdef np.ndarray[double, ndim=1,mode="c"] value=np.zeros(n_states, dtype=np.float64)
    cdef np.ndarray[double, ndim=1,mode="c"] new_value=np.zeros(n_states, dtype=np.float64)
    cdef np.ndarray[double, ndim=1,mode="c"] value_update=np.zeros(n_actions, dtype=np.float64)
    cdef np.ndarray[double, ndim=2,mode="c"] Qvalue=np.zeros((n_states,n_actions), dtype=np.float64)

    pnorm_V_iteration(num_states,  num_actions, &transition_probabilities[0,0,0], &reward[0],discount_v,
                    threshold_v, approx_level_v, &value[0],&new_value[0], &value_update[0],
                      &Qvalue[0,0]);

    return value, Qvalue

# Gradient Iteration to compute gradients
def i_pnorm_G_iteration(n_states,n_actions, n_features, np.ndarray[double, ndim=2,mode="c"]
                    featureMatrix, np.ndarray[double, ndim=3,mode="c"]
                    transition_probabilities,np.ndarray[double, ndim=1,mode="c"] reward, discount,
                        approx_level, np.ndarray[double, ndim=1,mode="c"] Values,
                        np.ndarray[double,ndim=2,mode="c"] Q_Values ):
    cdef int num_states=n_states;
    cdef int num_actions=n_actions;
    cdef int num_features=n_features;
    cdef double discount_v=discount;
    cdef double threshold_v=0.0001;
    cdef double approx_level_v=approx_level;
    cdef np.ndarray[double, ndim=2,mode="c"] V_Gradients=np.zeros((n_states,n_features), dtype=np.float64)
    cdef np.ndarray[double, ndim=2,mode="c"] new_V_Gradients=np.zeros((n_states,n_features), dtype=np.float64)
    cdef np.ndarray[double, ndim=3,mode="c"] Q_Gradients=np.zeros((n_states,n_actions,n_features), dtype=np.float64)
    cdef np.ndarray[double, ndim=1,mode="c"] d_pnorm_vector=np.zeros(n_states, dtype=np.float64)
    cdef np.ndarray[double, ndim=2,mode="c"] d_summation_matrix=np.zeros((n_states,n_actions), dtype=np.float64)
    cdef np.ndarray[double, ndim=1,mode="c"] d_sum_q=np.zeros(n_features, dtype=np.float64)
    pnorm_G_iteration(num_states,  num_actions, num_features, &featureMatrix[0,0],
                      &transition_probabilities[0,0,0], &reward[0], discount_v,
                    threshold_v, approx_level_v, &Values[0],&Q_Values[0,0],
                      &V_Gradients[0,0],&Q_Gradients[0,0,0],
                      &d_pnorm_vector[0],&d_summation_matrix[0,0],&new_V_Gradients[0,0],&d_sum_q[0]);

    return V_Gradients, Q_Gradients



def i_gsoft_V_iteration(n_states, n_actions,np.ndarray[double, ndim=3,mode="c"]
                    transition_probabilities,np.ndarray[double, ndim=1,mode="c"] reward, discount, approx_level):
    cdef int num_states=n_states;
    cdef int num_actions=n_actions;
    cdef double discount_v=discount;
    cdef double threshold_v=0.0001;
    cdef double approx_level_v=approx_level;
    cdef np.ndarray[double, ndim=1,mode="c"] value=np.zeros(n_states, dtype=np.float64)
    cdef np.ndarray[double, ndim=1,mode="c"] new_value=np.zeros(n_states, dtype=np.float64)
    cdef np.ndarray[double, ndim=1,mode="c"] value_update=np.zeros(n_actions, dtype=np.float64)
    cdef np.ndarray[double, ndim=2,mode="c"] Qvalue=np.zeros((n_states,n_actions), dtype=np.float64)

    gsoft_V_iteration(num_states,  num_actions, &transition_probabilities[0,0,0], &reward[0],discount_v,
                    threshold_v, approx_level_v, &value[0],&new_value[0], &value_update[0],
                      &Qvalue[0,0]);

    return value, Qvalue

# Gradient Iteration to compute gradients
def i_gsoft_G_iteration(n_states,n_actions, n_features, np.ndarray[double, ndim=2,mode="c"]
                    featureMatrix, np.ndarray[double, ndim=3,mode="c"]
                    transition_probabilities,np.ndarray[double, ndim=1,mode="c"] reward, discount,
                        approx_level, np.ndarray[double, ndim=1,mode="c"] Values,
                        np.ndarray[double,ndim=2,mode="c"] Q_Values ):
    cdef int num_states=n_states;
    cdef int num_actions=n_actions;
    cdef int num_features=n_features;
    cdef double discount_v=discount;
    cdef double threshold_v=0.0001;
    cdef double approx_level_v=approx_level;
    cdef np.ndarray[double, ndim=2,mode="c"] V_Gradients=np.zeros((n_states,n_features), dtype=np.float64)
    cdef np.ndarray[double, ndim=2,mode="c"] new_V_Gradients=np.zeros((n_states,n_features), dtype=np.float64)
    cdef np.ndarray[double, ndim=3,mode="c"] Q_Gradients=np.zeros((n_states,n_actions,n_features), dtype=np.float64)
    cdef np.ndarray[double, ndim=1,mode="c"] d_pnorm_vector=np.zeros(n_states, dtype=np.float64)
    cdef np.ndarray[double, ndim=2,mode="c"] d_summation_matrix=np.zeros((n_states,n_actions), dtype=np.float64)
    cdef np.ndarray[double, ndim=1,mode="c"] d_sum_q=np.zeros(n_features, dtype=np.float64)
    gsoft_G_iteration(num_states,  num_actions, num_features, &featureMatrix[0,0],
                      &transition_probabilities[0,0,0], &reward[0], discount_v,
                    threshold_v, approx_level_v, &Values[0],&Q_Values[0,0],
                      &V_Gradients[0,0],&Q_Gradients[0,0,0],
                      &d_pnorm_vector[0],&d_summation_matrix[0,0],&new_V_Gradients[0,0],&d_sum_q[0]);

    return V_Gradients, Q_Gradients





