import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from typing import Dict, List, Optional
from platform import python_version
from math import exp
assert pyro.__version__.startswith("1.8")  # I'm writing this tutorial with version


class SetParameters:

    def __init__(self, num_coeff, alpha_shape: str, n):
        """For initialising the MaximumAPosterioriModel object
        Args:
            num_coeff: number of coefficients for modulating alpha shape, i.e. length of w
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
