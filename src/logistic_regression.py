#coding=utf8
from __future__ import division
from vector import *
from gradient import minimize_stochastic
from statistic import *
import random

def logistic(x):
    return 1.0 / (1 + math.exp(x))

def logistic_prime(x):
    return logistic(x) * (1 - logistic(x))


def logistic_log_likelihood_i(x_i, y_i, beta):
    if y_i == 1:
        return math.log(logistic(dot(x_i, beta)))
    else:
        return math.log(1 - logistic(dot(x_i, beta)))

def logistic_log_likelihood(x, y, beta):
    return sum(logistic_log_likelihood_i(x_i, y_i, beta) for x_i, y_i in zip(x, y))

def logistic_log_partial_ij(x_i, y_i, beta, j):
    return (y_1 - logistic(dot(x_i, beta))) * x_i[j]

def logistic_log_gradient_i(x_i, y_i, beta, j):
    return [logistic_log_partial_ij(x_i, y_1, beta, j) for j, _ in enumerate(beta)]
   
def logistic_log_gradient(x, y, beta):
    return reduce(vector_add, [logistic_log_gradient_i(x_i, y_i, beta) for x_i, y_i in zip(x, y)])
    
