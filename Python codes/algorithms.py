# -*- coding: utf-8 -*-
"""
Distributed Subgradient Algorithms with Limited Communications

In this script, modules for distributed evaluation of convex problems are 
written. Instead of sharing all the gradients, limited communication restricts
the messages that can be sent. 

@author: Milind Rao and Stefano Rini
"""

from typing import Optional, Union, Iterable, Iterator, List
from functools import partial

import numpy as np
import scipy as sc
import scipy.linalg
from math_ops import Op

# from home/Documents/DLMD_DiffEx/DOLC/src/math_ops import Op


class Node:
    def gradient(self, xt):
        raise NotImplementedError

    @property
    def primal_dim(self) -> int:
        raise NotImplementedError



class NodeNumpy(Node):
    """ Class that implements behaviour of the nodes. 
            Calculating gradient, sharing it with the neighbours, having local
            data. Local variables X & Z are stored in the network object 
    """
    def __init__(self,
                 local_data,
                 local_op,
                 stochastic: Optional[float] = None,
                 **kwargs):
        """ Constructor
        Args:
            local_data - np array with node's copy of the data
            local_op - subclass of Op that has 2 static - apply, grad
            stochastic - if None, use full gradient, else float [0,1] that uses
                that percentage of the data values to compute the gradient
        """
        self.local_data = local_data
        self.local_op = local_op
        self.x_av = 0  # assessment of best value
        self.t = 0  # time
        self.stochastic = stochastic
    
    def partial_data_coord(self):
        """ For stochastic gradient descent, this returns continuous indices 
        with relating to the stochastic argument. 
        """
        num_stoc_data = np.int(np.ceil(self.stochastic * self.local_data.shape[0]))
        num_rounds = np.int(np.ceil(self.local_data.shape[0] / num_stoc_data))
        indices = np.arange((self.t % num_rounds) * num_stoc_data, (self.t % num_rounds + 1) * num_stoc_data)
        return indices

    def gradient(self, xt):
        """ Returns the gradient"""
        self.x_av = (self.t*self.x_av + xt)/(self.t+1)
        self.t += 1
        if not self.stochastic:
            return self.local_op.grad(self.local_data, xt)
        else:
            return self.local_op.grad(self.local_data[self.partial_data_coord(), :], xt)
    
    def current_eval(self):
        """ Evaluates the function to minimize at this point
        """
        return self.local_op.apply(self.local_data, self.x_av)

    @property
    def primal_dim(self) -> int:
        if isinstance(self.local_op, Op):
            return self.local_op.test_point_dim


class Network:
    """ Maintains weights in the network at each point in time. Could be static
    or changing in multiple ways. 
    """
    def __init__(self,
                 mode,
                 size,
                 gtype: str = 'full',
                 param: float = 0.9,
                 **kwargs):
        """ Args:
            mode - could be static/dynamic.
            size - number of nodes
            Optional kwargs-
            gtype - full/random/circle
            param - parameter . For random, it is the number of connections 
            lost. For circle, it is the number of neighbours. for full it has 
            no impact. 
        """
        
        self.mode = mode
        self.size = size
        self.gtype = gtype
        self.param = param
        self.graph_init = self.create_graph()  # tuple of doubly_stochastic, adjacency
        
    def doubly_stochastic_from_adjacency(self, adjacency):
        """ Function returns double stochastic P from adjacency matrix"""
        diag = np.diag(np.dot(adjacency, np.ones([self.size])))
        dmax = np.max(diag)
        doubly_stochastic = np.eye(self.size) + 1 / (dmax+1) * (adjacency - diag)
        return doubly_stochastic
        
    def create_graph(self):
        """ Creates full/random/circular graph """
        if self.gtype == 'full':
            adjacency = np.ones([self.size, self.size])-np.eye(self.size)
            doubly_stochastic = np.ones([self.size, self.size])/self.size
        elif self.gtype == 'random':
            a_ = np.tril(np.random.rand(self.size, self.size) < self.param, -1)
            adjacency = a_ + a_.T
            doubly_stochastic = self.doubly_stochastic_from_adjacency(adjacency)
        elif self.gtype == 'circle':
            c_ = [0]+[1]*self.param+[0]*(self.size - 2 * self.param-1) + [1]*self.param
            adjacency = sc.linalg.toeplitz(c_)
            doubly_stochastic = self.doubly_stochastic_from_adjacency(adjacency)
        return doubly_stochastic, adjacency
    
    def __call__(self):
        """ returns a weight matrix """
        if self.mode == 'static':
            return self.graph_init
        else:
            return self.create_graph()

    @property
    def doubly_stochastic(self):
        return self.__call__()[0]

    @property
    def adjacency(self):
        return self.__call__()[1]

    def neighbours_from_adjacency(self, adjacency=None):
        if adjacency is None:
            adjacency = self.graph_init[1]
        return [np.argwhere(adjacency[i, :] > 0).flatten() for i in range(self.size)]


class MessageCompressorBase:
    def encode(self, message: np.ndarray):
        raise NotImplementedError

    def decode(self, message: np.ndarray):
        raise NotImplementedError

    def update(self):
        pass


class MessageCompressor(MessageCompressorBase):
    """
    Class that implements random dimension reduction for compressing the size of message, typically: gradient or dual
    variables communicated
    """
    _MODES = ['subsample', 'fourier', 'random']

    def __init__(self,
                 dim_input: int,
                 dim_output: int,
                 mode: str = 'subsample',
                 add_noise: Optional[float] = None,
                 indices_mode: str = 'rand'):
        """
        Args:
            dim_input: dimension of the input vector
            dim_output: dim of output, should be less than or equal to the dimension of the input,
            mode: str, one of subsample/fourier
            add_noise - standard deviation of the additive noise
            indices_mode - either 'robin' or 'rand'
        """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.mode = mode
        self.add_noise = add_noise
        self.indices_mode = indices_mode

        self.t = 0  # internal clock
        self.shuffled_indices = np.arange(0, self.dim_input)
        self.indices = self.indices_from_time(self.t)

        self.orthonormal = self.get_orthonormal(mode)

    def get_orthonormal(self, mode: str) -> np.ndarray:
        assert mode in self._MODES, f"Not valid mode. {self._MODES}"
        if mode == "subsample":
            return np.eye(self.dim_input)
        elif mode == "fourier":
            return sc.linalg.dft(self.dim_input)/np.sqrt(self.dim_input)
        elif mode == 'random':
            return sc.linalg.polar(np.random.randn(self.dim_input, self.dim_input)+1e-5*np.eye(self.dim_input))[0]

    def indices_from_time(self, time):
        return self.shuffled_indices[
            np.arange(time * self.dim_output, (time + 1) * self.dim_output) % self.dim_input
        ]

    def update(self):
        self.t += 1
        if self.indices_mode == 'rand' and self.t % int(self.sampling_ratio) == 0:
            np.random.shuffle(self.shuffled_indices)
        self.indices = self.indices_from_time(self.t)

    def encode(self, message):
        """
        Args:
            message: 1 or 2-d array in which last dimension is to be compressed
        """
        return message @ self.orthonormal[:, self.indices]

    def decode(self, message):
        return message @ np.conj(self.orthonormal[:, self.indices].T)

    def encode_decode(self, message):
        """
        Result of a message that is encoded, sent across a possibly noisy network, and then decoded
        """
        network_message = self.encode(message)
        if self.add_noise is not None:
            network_message += self.add_noise * np.random.randn(*network_message.shape)
        return self.decode(network_message)

    def encode_decode_complement(self, message):
        return message - self.decode(self.encode(message))

    @property
    def sampling_ratio(self):
        return self.dim_input // self.dim_output


class SystemBase:
    """
    Base class for all decentralized subgradient algorithms with limited communications
    System is the collection of nodes, the network that collects them. It maintains dual, primal for the network
    """
    def __init__(self,
                 network: Network,
                 nodes: Optional[List[Node]] = None,
                 message_compressor: Optional[MessageCompressorBase] = None,
                 primal_init: Optional[np.ndarray] = None,  # If given, should be nodes x initial value
                 local_op: Optional[Op] = None,
                 all_data: Optional[np.ndarray] = None,  # TODO milind: expand to Union[Iterable, Iterator, np.ndarray]
                 stochastic: Optional[float] = None,
                 ):
        """
        Args:
            network - Network object with doubly stochastic matrix P_{ij}>0 if j receives message from i.
            nodes - list of nodes. Objects whose weights are flattened to primal_dim dim, has gradient function.
            local_op - Op at each node
            all_data - data evenly split among all nodes
            message_compressor - to do message reduction
            primal_init - Initial values of primal coordinates. (x_dim)
            stochastic - named argument. None if full gradient else fraction of data used to produce gradient
        """
        self.network = network
        self.num_nodes = network.size
        self.message_compressor = message_compressor
        self.nodes = nodes or self.create_nodes(all_data, local_op, stochastic)
        self.primal_dim = self.nodes[0].primal_dim or local_op.test_point_dim or primal_init.shape[1]
        self.primal = np.tile(primal_init, [self.num_nodes, 1]) if primal_init is not None else np.zeros([self.num_nodes, self.primal_dim])
        self.dual = np.zeros([self.num_nodes, self.primal_dim])
        self.primal_av = 0
        self.t = 0

    def batch_data(self, all_data: np.ndarray):
        # split the data among all the nodes
        num_data = int(all_data.shape[0] / self.num_nodes)
        indices_split = range(num_data, all_data.shape[0] - num_data + 1, num_data)
        yield from np.split(all_data, indices_split)

    def create_nodes(self, all_data: np.ndarray, local_op: Op, stochastic: Optional[float]):
        return [Node(data, local_op, stochastic) for data in self.batch_data(all_data)]

    def gradients(self):
        gradient_list = [node_i.gradient(self.primal[ind, :]) for ind, node_i in enumerate(self.nodes)]
        return np.stack(gradient_list)
        # return np.concatenate(gradient_list, axis=0)

    def update_dual(self):
        raise NotImplementedError

    def update_primal(self):
        raise NotImplementedError

    def update_primal_avg(self):
        self.primal_av = ((self.t - 1) * self.primal_av + self.primal) / self.t

    def update(self):
        self.t += 1
        if self.message_compressor is not None:
            self.message_compressor.update()
        self.update_dual()
        self.update_primal()
        self.update_primal_avg()


class DDA(SystemBase):
    def __init__(self,
                 prox_operator,
                 alpha,
                 local_op: Op,
                 all_data: np.ndarray,
                 network: Network,
                 message_compressor: MessageCompressor,
                 stochastic: Optional[float] = None,
                 primal_init: Optional[np.ndarray] = None,
                 ):

        super().__init__(local_op, all_data, network, message_compressor, stochastic, primal_init)
        self.prox_operator = prox_operator
        self.alpha = alpha

    def update_dual(self):
        """ Updates the dual variables"""
        self.dual = (self.network.doubly_stochastic @ self.message_compressor.encode_decode(self.dual) +
                     self.message_compressor.encode_decode_complement(self.dual) +
                     self.gradients())
            
    def update_primal(self):
        def prox_op(x):
            return self.prox_operator(x, self.alpha(self.t))
        self.primal = np.apply_along_axis(prox_op, 1, self.dual)


class DOGD(SystemBase):
    """
    Distributed Online Gradient Descent algorithm with limited communication
    """
    def __init__(self,
                 alpha_initial,
                 epoch_length_initial,
                 local_op: Op,
                 all_data: np.ndarray,
                 network: Network,
                 message_compressor: MessageCompressor,
                 stochastic: Optional[float] = None,
                 primal_init: Optional[np.ndarray] = None,
                 ):
        super().__init__(local_op, all_data, network, message_compressor, stochastic, primal_init)
        self.alpha_initial = alpha_initial
        self.epoch_length_initial = epoch_length_initial

    @property
    def alpha(self):
        epoch = np.ceil(np.log2(np.ceil(self.t / self.epoch_length_initial)))
        return self.alpha_initial / 2**epoch

    def update_dual(self):
        self.dual = (self.network.doubly_stochastic @ self.message_compressor.encode_decode(self.dual) +
                     self.message_compressor.encode_decode_complement(self.dual) -
                     self.alpha * self.gradients())

    def update_primal(self):
        self.primal = self.dual

    
class DNG(SystemBase):
    """
    Distributed Nesterov Gradient Descent with limited communication
    """
    def __init__(self,
                 learning_rate,
                 local_op: Op,
                 all_data: np.ndarray,
                 network: Network,
                 message_compressor: MessageCompressor,
                 stochastic: Optional[float] = None,
                 primal_init: Optional[np.ndarray] = None,
                 ):
        super().__init__(local_op, all_data, network, message_compressor, stochastic, primal_init)
        self.learning_rate = learning_rate

    def update_dual(self):
        q = self.t // self.message_compressor.sampling_ratio
        received = self.message_compressor.encode_decode(self.dual)
        primal_old = self.primal
        self.primal = (self.network.doubly_stochastic @ received +
                       self.message_compressor.encode_decode_complement(self.dual) -
                       (self.learning_rate / (q + 1) * self.gradients()))
        self.dual = self.primal + q / (q + 3) * (self.primal - primal_old)

    def update_primal(self):
        pass


if __name__ == '__main__':
    pass


    # def coord_to_send(self):
    #     """ Either random selection or round robin
    #     """
    #     if self.coord_select == 'rand':
    #         return self.random_coord()
    #     else:
    #         return self.round_robin()
    #
    #
    # def random_coord(self):
    #     """ returns a list of bools which state which coordinates to transmit
    #     """
    #     return np.random.rand(self.x_dim) <= self.reduce_dim
    #
    #
    # def round_robin(self):
    #     """ returns the coordinates from a round robin approach
    #     """
    #     num_coord = np.ceil(self.reduce_dim * self.x_dim)
    #     num_rounds = np.ceil(self.x_dim / num_coord)
    #     return [(self.t % num_rounds) * num_coord <= ind < (self.t % num_rounds + 1) * num_coord
    #             for ind in range(self.x_dim)]
    #
    #
    # def update_z(self):
    #     """ Updates the dual variables"""
    #     G_list = [node_i.gradient(self.primal[ind:ind + 1, :]) for ind, node_i in enumerate(self.nodes_net)]
    #     G = np.concatenate(G_list, axis=0)
    #
    #     P = self.network()[0]
    #     c2s = self.coord_to_send()
    #     self.dual[:, c2s] = np.dot(P, self.dual[:, c2s])  # Exchanging Z values
    #     if self.add_noise:  # Additive noise
    #         self.dual[:, c2s] += self.add_noise * np.random.randn(self.num_nodes, np.sum(c2s))
    #     self.dual += G  # Updating based on gradients
    #
    #
    # def update_x(self):
    #     self.t = self.t + 1
    #
    #     def prox_op(x):
    #         return self.prox_operator(x, self.alpha(self.t))
    #
    #     self.primal = np.apply_along_axis(prox_op, 1, self.dual)
    #     self.primal_av = ((self.t - 1) * self.primal_av + self.primal) / self.t

