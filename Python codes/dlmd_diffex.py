import pdb
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from algorithms import Network, SystemBase, MessageCompressorBase
from math_ops import LinearRegression, square_prox, SquareProxInit
from lin_reg import gen_lin_reg_data
from mnist import NodeMnist


class QuantizedMessageCompressor(MessageCompressorBase):
    def __init__(
            self,
            limit: float,
            bins: int = 10,
            scaling_rate: Callable = lambda x: 1,
            add_noise: Optional[float] = None
    ):
        self.limit = limit
        self.bins = bins
        self.bin_vals = np.linspace(-limit, limit, bins)
        self.delta = self.bin_vals[1] - self.bin_vals[0]
        self.add_noise = add_noise
        self.t = 0
        self.scaling_rate = scaling_rate

    def update(self):
        self.t += 1

    def encode(self, message):
        """ Quantizes the message

        Raises:
            ValueError if out of bounds
        """
        # if np.max(message) > self.limit or np.min(message) < - self.limit:
        #     raise ValueError("Quantizer out of bounds")
        indices = np.searchsorted(self.bin_vals, message)
        lower_lim = self.bin_vals[indices - 1]
        upper_lim = self.bin_vals[indices]
        prob = (upper_lim - message) / self.delta
        return np.where(np.random.rand(*prob.shape) <= prob, lower_lim, upper_lim)

    def decode(self, message: np.ndarray):
        if self.add_noise is not None:
            return message + self.add_noise / self.scaling_rate(self.t) * np.random.randn(*message.shape)
        else:
            return message


class PerfectMessageCompressor(MessageCompressorBase):
    def encode(self, message: np.ndarray):
        return message

    def decode(self, message: np.ndarray):
        return message


class DLMDDiffEx(SystemBase):
    def __init__(
            self,
            network: Network,
            learning_rate: Callable,
            beta_rate: Callable,
            prox_operator: Callable,
            nodes=None,
            message_compressor: Optional[MessageCompressorBase] = None,
            primal_init: Optional[np.ndarray] = None,
            ):

        super().__init__(network, nodes=nodes, message_compressor=message_compressor, primal_init=primal_init)
        self.diff = []
        self.diff_estimate = []
        self.neighbour_list = self.network.neighbours_from_adjacency()
        self.initialize_diff()
        self.learning_rate = learning_rate
        self.beta_rate = beta_rate
        self.prox_operator = prox_operator

    # def __init__(
    #         self,
    #         local_op: Op,
    #         all_data: np.ndarray,
    #         network: Network,
    #         message_compressor: MessageCompressorBase,
    #         learning_rate: Callable,
    #         beta_rate: Callable,
    #         prox_operator: Callable,
    # ):
    #     super().__init__(local_op, all_data, network, message_compressor)

    def initialize_diff(self):
        for neigh in self.neighbour_list:
            self.diff.append(np.zeros([len(neigh), self.primal_dim]))
            self.diff_estimate.append(np.zeros([len(neigh), self.primal_dim]))

    def update_primal(self):
        def prox_op(x):
            return self.prox_operator(x, self.learning_rate(self.t))
        self.primal = np.apply_along_axis(prox_op, 1, self.dual)

    def update_dual(self):
        adjusted_weights = (1 - self.beta_rate(self.t)) * np.eye(self.num_nodes) + self.beta_rate(self.t) * self.network.doubly_stochastic
        y_sum = np.zeros(self.dual.shape)
        for i in range(self.num_nodes):
            delta_i = self.message_compressor.encode(
                self.dual[self.neighbour_list[i], :] - self.diff[i]
            )
            self.diff[i] += delta_i
            self.diff_estimate[i] += self.message_compressor.decode(delta_i)
            y_sum[i, :] = adjusted_weights[i, self.neighbour_list[i]] @ self.diff_estimate[i]

        self.dual = np.diag(np.diag(adjusted_weights)) @ self.dual + self.gradients() + y_sum


def linear_regression_test():
    num_dims = 20
    num_nodes = 5
    num_data = 200
    num_time = 1000

    network = Network(mode="static", size=num_nodes, gtype="full")
    all_data, model_weight = gen_lin_reg_data(num_dims, num_data, limit_data=2)
    lin_reg_op = LinearRegression(num_dims)

    scaling_rate = lambda t: 0.5 * t ** 0.25
    beta_rate = lambda t: 0.5 * t ** -0.25
    learning_rate = lambda t: 0.01 * t ** (-1.25 / 2)

    quantizer = QuantizedMessageCompressor(limit=5, bins=20, scaling_rate=scaling_rate, add_noise=0.2)

    dlmd_diffex = DLMDDiffEx(
        local_op=lin_reg_op, all_data=all_data, network=network, message_compressor=quantizer,
        learning_rate=learning_rate, beta_rate=beta_rate, prox_operator=square_prox
    )

    loss = []
    for tim in tqdm(range(num_time)):
        dlmd_diffex.update()
        loss.append(np.linalg.norm(dlmd_diffex.primal_av[1:2, :] - model_weight))

    fig = plt.Figure()
    plt.plot(list(range(num_time)), loss)
    plt.title("Model gap with time")
    plt.savefig("eg_fig.png")


def test_mnist():

    num_nodes = 5
    num_rounds = 100
    network = Network(mode="static", size=num_nodes, gtype="circle", param=1)
    nodes = [NodeMnist(local_index=ind, num_nodes=num_nodes, sharding="unbalanced") for ind in range(num_nodes)]

    scaling_rate = lambda t: 2 * t ** 0.25
    beta_rate = lambda t: 0.9  # * t ** -0.25
    learning_rate = lambda t: 0.03 * (t / 5) ** (-1.25 / 2)

    quantizer = QuantizedMessageCompressor(limit=20, bins=100, scaling_rate=scaling_rate, add_noise=0.3)
    # quantizer = PerfectMessageCompressor()

    primal_i = np.random.randn(nodes[0].primal_dim)

    dlmd_diffex = DLMDDiffEx(
        network=network, nodes=nodes, message_compressor=quantizer,
        learning_rate=learning_rate, beta_rate=beta_rate,
        prox_operator=SquareProxInit(primal_i), primal_init=primal_i
    )

    for ind in tqdm(range(num_rounds)):
        dlmd_diffex.update()
        if ind % 100 == 0:
            print(nodes[0].evaluate(dlmd_diffex.primal[0, :]))

    print(float(nodes[0].evaluate(dlmd_diffex.primal[0, :])['acc']))


def test_mnist_with_plots():

    num_nodes = 5
    num_rounds = 2000
    network = Network(mode="static", size=num_nodes, gtype="circle", param=1)
    nodes = [NodeMnist(local_index=ind, num_nodes=num_nodes, sharding="unbalanced") for ind in range(num_nodes)]

    # scaling_rate = lambda t: 2 * t ** 0
    scaling_rate = lambda t: t ** 0.7
    # scaling_rate = lambda t: 1
    beta_rate = lambda t: t ** (-0.15)  # * t ** -0.25
    # beta_rate = lambda t: 1  # * t ** -0.25
    # learning_rate = lambda t: 0.03 * (t / 5) ** (-1.25 / 2)
    learning_rate = lambda t: 0.6 * t ** (-0.5)

    quantizer = QuantizedMessageCompressor(limit=30, bins=100, scaling_rate=scaling_rate, add_noise=0.1)
    # quantizer = PerfectMessageCompressor()

    primal_i = np.random.randn(nodes[0].primal_dim)

    dlmd_diffex = DLMDDiffEx(
        network=network, nodes=nodes, message_compressor=quantizer,
        learning_rate=learning_rate, beta_rate=beta_rate,
        prox_operator=SquareProxInit(primal_i), primal_init=primal_i
    )

    acc_array = []
    kbest_array = []
    for ind in tqdm(range(num_rounds)):
        dlmd_diffex.update()
        # acc_array.append(float(nodes[0].evaluate(dlmd_diffex.primal[0, :])['acc']))
        # kbest_array.append(float(nodes[0].evaluate(dlmd_diffex.primal[0, :])['kbest']))
        if ind % 100 == 0:
            print(nodes[0].evaluate(dlmd_diffex.primal[0, :]))

    print(nodes[0].evaluate(dlmd_diffex.primal[0, :]))
            

    #fig = plt.Figure()
    #plt.plot(list(range(num_rounds)), acc_array)
    #plt.title("Accuracy")
    #plt.savefig("eg_fig_Acc.png")

if __name__ == "__main__":
    # linear_regression_test()
    test_mnist_with_plots()
