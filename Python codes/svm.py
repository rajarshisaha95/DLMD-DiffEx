# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:29:32 2018

This is a module to generate results for SVM in distributed communications. It
compares the impact of fully centralized optimization, no communication, and 
differing message sizes in random communications

"""
import pickle
from functools import partial
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from algorithms import Network, MessageCompressor, DDA, SystemBase
from math_ops import square_prox, alpha_sqrt, NormedHingeLoss, ExactMatch


def gen_data_2_dis(num_nodes=10,
                   num_dims=10,
                   num_data=100):
    
    x_act = np.random.randn(1, num_dims)
    x_act = x_act/np.linalg.norm(x_act)
    
    z_1 = np.random.randn(1, num_dims)
    z_1 = z_1/np.linalg.norm(z_1)
    lim = 0.7
    while np.dot(x_act, z_1.T) < lim:
        z_1 = np.random.randn(1, num_dims)
        z_1 = z_1/np.linalg.norm(z_1)
    
    z_2 = np.random.randn(1, num_dims)
    z_2 = z_2/np.linalg.norm(z_2)
    while np.dot(x_act, z_2.T) > -lim:
        z_2 = np.random.randn(1, num_dims)
        z_2 = z_2/np.linalg.norm(z_2)
    
    data1 = z_1 + 0.6 * np.random.randn(num_data // 2, num_dims)
    data2 = z_2 + 0.6 * np.random.randn(num_data // 2, num_dims)
    
    data = np.concatenate([data1, data2])
    y = np.sign(np.dot(data, x_act.T))
    all_data = np.concatenate([data, y], axis=1)
    return x_act, all_data


#    dat_mean = np.mean(data)
#    dU,dD,dV = np.linalg.svd(data)
#    data_reduc = np.dot(dU[:,:2],np.diag(dD[:2]))
#    x_act_reduc = np.dot(x_act,dV[:2,:].T).T
#    slop = -x_act_reduc[0]/x_act_reduc[1]
#    x_ax = np.linspace(np.min(data_reduc[:,0]),
#                       np.max(data_reduc[:,0]),
#                       100)
#    y_ax = slop*x_ax
#    fig = plt.figure(1)
#    plt.scatter(data_reduc[:,0],data_reduc[:,1],c=y+1)
#    plt.plot(x_ax,y_ax)
#    plt.ylim([np.min(data_reduc[:,1]),np.max(data_reduc[:,1])])


if __name__ == "__main__":

    num_nodes = 10
    num_dims = 30
    num_data = 100
    num_iter = 100

    time_lim = 150
    loss = np.zeros([5, num_iter, time_lim])  # Full, Full 0.5, Full 0.25, centr, no_share

    Sys: List[SystemBase] = [[] for _ in range(5)]
    for nind in tqdm(range(num_iter)):
        x_act, all_data = gen_data_2_dis(num_nodes, num_dims, num_data)
    #    np.random.shuffle(all_data)
        network = Network('static', num_nodes, gtype='full')
        svm_op = NormedHingeLoss(num_dims)

        Sys[0] = DDA(square_prox, alpha_sqrt, svm_op, all_data, network,
                     message_compressor=MessageCompressor(num_dims, num_dims))
        Sys[1] = DDA(square_prox, alpha_sqrt, svm_op, all_data, network,
                     message_compressor=MessageCompressor(num_dims, num_dims // 2))
        Sys[2] = DDA(square_prox, alpha_sqrt, svm_op, all_data, network,
                     message_compressor=MessageCompressor(num_dims, num_dims // 4))
        Sys[3] = DDA(square_prox, alpha_sqrt, svm_op, all_data, Network('static', 1, gtype='full'),
                     message_compressor=MessageCompressor(num_dims, num_dims))
        Sys[4] = DDA(square_prox, alpha_sqrt, svm_op, all_data, Network('static', num_nodes, gtype='circle', param=0),
                     message_compressor=MessageCompressor(num_dims, num_dims))

        for tind in range(time_lim):
            for sind in range(5):
                Sys[sind].update()
                loss[sind, nind, tind] = ExactMatch.apply(all_data, Sys[sind].primal_av[0, :])

    file_name = 'svm'
    # with open('../figures/{}.dat'.format(file_name), 'rb') as fop:
    #     loss = pickle.load(fop)
    fig = plt.figure(0)
    plt.rcParams.update({'font.size': 14})
    cols = ['k', 'b.', 'g:', 'r-.', 'y--']
    for sind in range(5):
        plt.plot(range(time_lim), np.mean(loss[sind, :, :], 0), cols[sind], lw=3)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Classification error')
    plt.legend(('full', '1/2 coord', '1/4 coord', 'central', 'no comm'), loc=1)

    plt.savefig('../figures/{}.eps'.format(file_name), bbox_inches='tight')
    plt.show()
    with open('../figures/{}.dat'.format(file_name), 'wb') as fop:
        pickle.dump(loss, fop)