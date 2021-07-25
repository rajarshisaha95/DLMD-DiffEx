from abc import ABC

import numpy as np
import scipy as sc
import scipy.special


class Op:
    """ Abstract operation that serves as local function and gradient op """

    def __init__(self, test_point_dim=None):
        self.test_point_dim = test_point_dim

    @staticmethod
    def apply(data, test_point):
        """ Applies the loss function
            Args:
                data - typically, each row is a different data point
                test_point - row vector indicating the point

            Returns:
                loss - scalar. Should be mean loss per data point
        """
        raise NotImplementedError

    @staticmethod
    def grad(data, test_point):
        """ Finds the gradient
        Returns:
            grad - row vector returning the gradient.
        """
        raise NotImplementedError


class HingeLoss(Op):
    @staticmethod
    def apply(data, test_point):
        return np.mean(np.maximum(1 - data[:, -1:] * np.dot(data[:, :-1], test_point.T), 0))

    @staticmethod
    def grad(data, test_point):
        is_active = data[:, -1:] * np.dot(data[:, 0:-1], test_point.T) < 1
        return -np.mean(is_active * data[:, -1:] * data[:, 0:-1], 0)


class NormedHingeLoss(Op):
    @staticmethod
    def apply(data, test_point):
        return 1 / 2 * np.dot(test_point, test_point.T) / data.shape[0] + HingeLoss.apply(data, test_point)

    @staticmethod
    def grad(data, test_point):
        return test_point / data.shape[0] + HingeLoss.grad(data, test_point)


class ExactMatch(Op, ABC):
    @staticmethod
    def apply(data, test_point):
        pred = np.sign(np.dot(data[:, :-1], test_point.T))
        return np.mean(pred * data[:, -1] < 0)


class LinearRegression(Op):
    @staticmethod
    def apply(data, test_point):
        """ linear regression function"""
        y_guess = np.dot(data[:, :-1], test_point.T)
        return 0.5 * np.linalg.norm(y_guess - data[:, -1:]) ** 2 / data.shape[0]

    @staticmethod
    def grad(data, test_point):
        """ linear regression function gradient"""
        y_err = np.dot(data[:, :-1], test_point.T) - data[:, -1:]
        return np.dot(y_err.T, data[:, :-1]) / data.shape[0]


class RobustRegression(Op):
    @staticmethod
    def apply(data, test_point):
        y_guess = np.dot(data[:, :-1], test_point.T)
        return np.linalg.norm(y_guess - data[:, -1:], ord=1) / data.shape[0]

    @staticmethod
    def grad(data, test_point):
        y_guess_sign = np.sign(np.dot(data[:, :-1], test_point.T) - data[:, -1:])
        return np.dot(y_guess_sign.T, data[:, :-1]) / data.shape[0]


class Huber(Op):
    delta = 1

    @staticmethod
    def apply(data, test_point):
        y_guess = np.dot(data[:, :-1], test_point.T)
        err = y_guess - data[:, -1:]
        return np.mean(sc.special.huber(Huber.delta, err))

    @staticmethod
    def grad(data, test_point):
        y_guess = np.dot(data[:, :-1], test_point.T)
        err = y_guess - data[:, -1:]
        grad_scal = np.where(np.abs(err) <= Huber.delta,
                             err,
                             Huber.delta * np.sign(err))
        return np.dot(grad_scal.T, data[:, :-1]) / data.shape[0]


class PhaseRetrieval(Op):
    @staticmethod
    def apply(data, test_point):
        y_guess = np.abs(np.dot(data[:, :-1], test_point.T))
        return 0.5 * np.linalg.norm(y_guess - data[:, -1:]) ** 2 / data.shape[0]

    @staticmethod
    def grad(data, test_point):
        y_guess = np.abs(np.dot(data[:, :-1], test_point.T))
        return np.dot(((y_guess - data[:, -1:]) * np.sign(y_guess)).T, data[:, :-1]) / data.shape[0]


def square_prox(test_point, alpha):
    """ Proximal projection with prox function - squared l2"""
    return -alpha * test_point


class SquareProxInit:
    def __init__(self, primal_init: np.ndarray):
        self._primal_init = primal_init

    def __call__(self, test_point, alpha):
        return self._primal_init - alpha * test_point


def entropy_prox(test_point, alpha):
    """ Proximal projection with prox function - entropic loss"""
    logits_raw = -alpha * test_point
    prob_raw = np.exp(logits_raw - np.max(logits_raw))
    return prob_raw / np.sum(prob_raw)


def alpha_sqrt(time, constant=1):
    """ How step size function alpha changes with time"""
    return constant / np.sqrt(time)
