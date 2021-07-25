# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:07:39 2018

This is a module to generate simulations for the linear regression case. We 
will be testing the impact of stochastic gradients, noise
"""
from functools import partial
import pickle

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from algorithms import Network, MessageCompressor, DDA
from math_ops import LinearRegression, square_prox, alpha_sqrt


num_dims = 30
num_data = 200
num_nodes = 10
num_iter = 10


def gen_lin_reg_data(num_dims, num_data, limit_data=5):
    x_act = np.random.randint(-limit_data, limit_data, size=[1, num_dims])
    meas = np.random.randn(num_data, num_dims)
    sig_in = 0.1
    y_act = np.dot(meas, x_act.T) + np.random.randn(num_data, 1) * sig_in
    all_data = np.concatenate([meas, y_act], axis=1)
    return all_data, x_act


if __name__ == '__main__':
    sys_to_test = ['full', 'stochastic', 'noisy 0.1', 'noisy 1']
    num_test = len(sys_to_test)
    num_time = 300
    reduce_dim = 0.5

    loss = dict((x_key, np.zeros([num_iter, num_time])) for x_key in sys_to_test)
    lin_reg = LinearRegression(num_dims)

    network = Network('static', num_nodes, gtype='full')
    alpha_func_s = partial(alpha_sqrt, constant=0.6)
    alpha_func = partial(alpha_sqrt, constant=0.7)
    Sys = {}

    compressor_type = 'random'
    comp_ind_mode = 'rand'

    for iter_ind in tqdm(range(num_iter)):
        compressor = MessageCompressor(num_dims, int(reduce_dim * num_dims), mode=compressor_type,
                                       indices_mode=comp_ind_mode)
        compressor_stoch = MessageCompressor(num_dims, int(reduce_dim * num_dims), mode=compressor_type,
                                             indices_mode=comp_ind_mode)
        compressor_noise_s = MessageCompressor(num_dims, int(reduce_dim * num_dims), mode=compressor_type,
                                               indices_mode=comp_ind_mode, add_noise=0.1)
        compressor_noise = MessageCompressor(num_dims, int(reduce_dim * num_dims), mode=compressor_type,
                                             indices_mode=comp_ind_mode, add_noise=10.0)

        all_data, x_act = gen_lin_reg_data(num_dims, num_data)
        Sys['full'] = DDA(square_prox, alpha_func, lin_reg, all_data, network, compressor)
        Sys['stochastic'] = DDA(square_prox, alpha_func_s, lin_reg, all_data, network, compressor, stochastic=0.2)
        Sys['noisy 0.1'] = DDA(square_prox, alpha_func, lin_reg, all_data, network, compressor_noise_s)
        Sys['noisy 1'] = DDA(square_prox, alpha_func, lin_reg, all_data, network, compressor_noise)

        for tind in range(num_time):
            for key in sys_to_test:
                Sys[key].update()
                loss[key][iter_ind, tind] = np.linalg.norm(Sys[key].primal_av[1:2, :] - x_act)

    file_name = 'lin_reg'
    with open(f'../figures/{file_name}.dat', 'wb') as fop:
        pickle.dump(loss, fop)
    # with open('figures/{}.dat'.format(file_name),'rb') as fop:
    #       loss=pickle.load(fop)

    fig = plt.figure()
    plt.rcParams.update({'font.size': 14})
    cols = {'full': 'k-', 'stochastic': 'b:', 'noisy 0.1': 'g-.', 'noisy 1': 'r--'}
    for key in sys_to_test:
        l_mean = np.mean(loss[key], 0)
        plt.semilogy(l_mean, cols[key], label=key, lw=3)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel(r'$\||\hat{x}-x^*\||_2$')
    plt.savefig(f'../figures/{file_name}.eps', bbox_inches='tight')
    plt.show()
