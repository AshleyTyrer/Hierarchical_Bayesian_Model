import numpy as np
import torch
import pyro
from platform import python_version
from math import exp
assert pyro.__version__.startswith("1.8")  # I'm writing this tutorial with version


class SetParameters:

    def __init__(self, num_coeff, alpha_shape: str, n):
        """For initialising the SetParameters object
        Args:
            num_coeff: number of coefficients for modulating alpha shape/gradient, i.e. length of w
            alpha_shape: string stating whether alpha will be polynomial or sigmoid
            n: number of trials"""

        self.ncoeff = num_coeff
        self.alpha_shape = alpha_shape
        self.numtrials = n

    def polynomial_alpha(self, w):
        """For defining alpha in a polynomial shape
        Args:
            w: shape parameters for alpha"""

        alpha = torch.zeros((self.numtrials,))
        ii = torch.linspace(-1, 1, self.numtrials)

        for j in range(self.numtrials):
            iij = torch.empty((self.ncoeff,))
            for i in range(self.ncoeff):
                iij[i] = ii[j] ** (i + 1)
            alpha[j] = 1 + torch.dot(iij.double(), w.double())

        return alpha

    def sigmoid_alpha(self, w, m):
        """For defining alpha in a polynomial shape
        Args:
            w: gradient for sigmoid function
            m: midpoint for sigmoid function"""

        alpha = torch.zeros((self.numtrials,))
        ii = torch.linspace(0, 3, self.numtrials)

        for j in range(self.numtrials):
            alpha[j] = 1 / (1 + exp(torch.matmul(-w, (ii[j] - m))))

        return alpha

    def calculate_beta(self, beta_0, beta_grad, alpha, p):
        """For calculating betas from beta_0, beta_grad and alpha
        Args:
            beta_0: initial condition of beta
            beta_grad: gradient of beta
            alpha: trial-by-trial 'learning' parameter
            p: number of channels"""

        if torch.is_tensor(beta_0):
            beta = torch.zeros((p, self.numtrials))
        else:
            beta = np.zeros((p, self.numtrials))

        beta[:, 0] = beta_0
        for j in range(1, self.numtrials):
            beta[:, j] = beta[:, j - 1] + alpha[j] * beta_grad

        return beta

    def calculate_y(self, x, beta, epsilon, t):
        """For calculating y from the data (x), betas and noise (epsilon)
        Args:
            x: input data
            beta: beta parameters
            epsilon: noise term
            t: time points per trial"""

        y = np.zeros((self.numtrials, t))

        for j in range(self.numtrials):
            for tp in range(t):
                y[j, tp] = np.dot(x[j, tp, :], beta[:, j]) + epsilon[j, tp]

        return y
