# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:48:03 2018

This is a module to the robust linear regression with constraints. We will test
different configurations of the network (full, circle, random) with round-robin
and randomized coordinate selection

WARNING: Not refactored with new code
"""

import numpy as np
import algorithms
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from functools import partial

num_dims = 20
num_data = 100
num_nodes = 10
num_iter = 20

def gen_rob_reg_data(num_dims, num_data):
    x_act = np.exp(np.random.rand(1,num_dims)*5)
    x_act = x_act/np.sum(x_act)
    x_act[x_act<0.05]=0
    x_act = x_act/np.sum(x_act)
    
    meas = np.random.randn(num_data,num_dims)
    outlier_perc = 0.2
    sig_out = 4
    sig_in = 0.1
    is_outlier = np.random.rand(num_data,1)<= outlier_perc
    y_meas = np.dot(meas,x_act.T) + \
             np.where(is_outlier, sig_out*np.random.randn(num_data,1), 
                      sig_in*np.random.randn(num_data,1))
    all_data = np.concatenate((meas, y_meas), axis=1)
    return (all_data, x_act)

sys_to_test = ['full rand','full robin', 'circle rand','circle robin',
               ]#'no_comm']
num_test = len(sys_to_test)
num_time = 400
reduce_dim = 0.5

loss = dict((x_key, np.zeros([num_iter, num_time])) for x_key in sys_to_test)
rob_reg = algorithms.RobustRegression(num_dims)

P = algorithms.Network('static', num_nodes, gtype='full')
P_circ = algorithms.Network('static', num_nodes, gtype='circle', param=1)
P_o = algorithms.Network('static', num_nodes, gtype='circle', param=0)
alpha_func = partial(algorithms.alpha_sqrt, constant=1)

Sys = {}

def x_init(num_data=num_data, num_dims=num_dims):
    x_guess = np.random.rand(num_data, num_dims)
    return x_guess/np.sum(x_guess, axis=1, keepdims=True)

for iter_ind in tqdm(range(num_iter)):
    all_data, x_act = gen_rob_reg_data(num_dims, num_data)
    Sys[sys_to_test[0]] = algorithms.System(rob_reg, algorithms.entropy_prox, all_data,
                                            P, alpha_func, reduce_dim, x_init = x_init())
    Sys[sys_to_test[1]] = algorithms.System(rob_reg, algorithms.entropy_prox, all_data, P,
                                            alpha_func, reduce_dim, x_init=x_init(),
                                            coord_select='robin')
    Sys[sys_to_test[2]] = algorithms.System(rob_reg, algorithms.entropy_prox, all_data,
                                            P_circ, alpha_func, reduce_dim,
                                            x_init = x_init())
    Sys[sys_to_test[3]] = algorithms.System(rob_reg, algorithms.entropy_prox, all_data,
                                            P_circ, alpha_func, reduce_dim,
                                            x_init = x_init(), coord_select='robin')
#    Sys[sys_to_test[4]] = dcda.System(rob_reg, dcda.entropy_prox, all_data, 
#                               P_o, partial(dcda.alpha_sqrt, constant = 0.1), 
#                               reduce_dim, x_init = x_init())
        
    for tind in range(num_time):
        for key in sys_to_test:
            Sys[key].update()
            loss[key][iter_ind,tind] = np.linalg.norm(Sys[key].primal_av[1:2, :] -
                                                      x_act, ord=1)

file_name = 'rob_reg'
#with open('figures/{}.dat'.format(file_name),'wb') as fop:
#       pickle.dump(loss, fop)
with open('figures/{}.dat'.format(file_name),'rb') as fop:
       loss = pickle.load(fop)         
fig = plt.figure()
plt.rcParams.update({'font.size': 14})
cols_raw = ['k','b:','g-.','r--','c^']
cols = dict((key,col) for key, col in zip(sys_to_test, cols_raw))
for key in sys_to_test:
    l_mean = np.mean(loss[key],0)
    plt.plot(l_mean,cols[key], label=key, lw=3)
plt.legend()
plt.xlabel('Iterations')
plt.ylabel(r'$\||\hat{x}-x^*\||_1$')
plt.savefig('figures/{}.eps'.format(file_name),bbox_inches='tight')
