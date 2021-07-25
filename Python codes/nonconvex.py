# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:52:45 2018

WARNING: not refactored with new code
"""

import numpy as np
import algorithms
from functools import partial
import matplotlib.pyplot as plt

num_dim = 10
num_data = 300
num_nodes = 5

np.random.seed(1)
x_act = np.random.randn(1,num_dim)

data = np.random.randn(num_data,num_dim)
y = np.abs(np.dot(data,x_act.T))

all_data = np.concatenate([data,y],axis=1)

prop = algorithms.PhaseRetrieval(num_dim)
Sys = algorithms.System(prop,
                        algorithms.square_prox,
                        all_data,
                        algorithms.Network('static', num_nodes, gtype='full'),
                        partial(algorithms.alpha_sqrt, constant=0.1),
                        1.0)

loss=[]
for ind in range(10000):
    Sys.update()
    loss.append(prop.apply(all_data, Sys.primal_av[0:1, :]))
    
plt.plot(loss)