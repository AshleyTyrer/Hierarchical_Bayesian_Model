import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from typing import Dict, List, Optional
from platform import python_version
assert pyro.__version__.startswith("1.8")  # I'm writing this tutorial with version


class MaximumAPosterioriModel:

    def __init__(self, num_coeff):
        """For initialising the MaximumAPosterioriModel object
        Args:
            num_coeff: number of coefficients for modulating alpha shape, i.e. length of w"""

        self.ncoeff = num_coeff

    def model_map(self, x_model, y_model):
        """Encodes observations, latent random variables and parameters in the model
        Args:
            x_model: neural data in form: number of trials x timepoints x channels
            y_model: behavior/continuous learning in form: number of trials x timepoints"""

        sigma_mod = 1
        N, T, p = x_model.shape  # trials, time, and channels/ROIs
        tau_grad, tau0, tau_w = (1 / N), 1, 1  # variance of the beta,w distributions
        sigma_shape, sigma_rate = N * T, sigma_mod * N * T  # prior parameters for the noise distribution

        # learning coefficients
        w = torch.zeros((self.ncoeff,))
        for i in range(self.ncoeff):
            w[i] = pyro.sample(f"w_{i}", dist.Normal(0.0, tau_w))

        # learning weights per trial, computed using the learning coefficients
        alpha = torch.zeros((N,))
        ii = torch.linspace(-1, 1, N)

        for j in range(N):
            iij = torch.empty((self.ncoeff,))
            for i in range(self.ncoeff):
                iij[i] = ii[j] ** (i + 1)
            alpha[j] = 1 + torch.dot(iij.double(), w.double())

        # intercept
        intercept = pyro.sample("intercept", dist.Normal(0.0, 10.0))

        # variance of the noise error
        sigma = pyro.sample("sigma", dist.Gamma(sigma_shape, sigma_rate))

        # betas trial by trial
        beta_grad = torch.zeros((p,))  # beta gradient
        beta0 = torch.zeros((p,))  # initial beta
        for i in range(p):
            beta_grad[i] = pyro.sample(f"beta_grad_{i}", dist.Normal(0.0, tau_grad))
            beta0[i] = pyro.sample(f"beta0_{i}", dist.Normal(0.0, tau0))
        beta = torch.zeros((p, N))
        beta[:, 0] = beta0

        # mean prediction
        mean = torch.zeros((N, T)) + intercept
        mean[0, :] = mean[0, :] + torch.matmul(x_model[0, :, :], beta[:, 0])

        for j in range(1, N):
            beta[:, j] = beta[:, j - 1] + alpha[j] * beta_grad
            mean[j, :] = mean[j, :] + torch.matmul(x_model[j, :, :], beta[:, j])

        mean = mean.flatten()
        y_model = y_model.flatten()

        with pyro.plate('data_plate'):
            pyro.sample("y", dist.Normal(mean, sigma), obs=y_model)

    def guide_map(self, x_guide, y_guide):
        """Defines the variational distribution which serves as an approximation to the posterior
        Args:
            x_guide: neural data in form: number of trials x timepoints x channels
            y_guide: behavior/continuous learning in form: number of trials x timepoints"""

        N, T, p = x_guide.shape

        # mean y
        meany = y_guide.flatten().mean()

        # compute beta in the first trials and in the last trials
        N_star = round(N / 10)
        x0 = torch.reshape(x_guide[0:N_star, :, :], (N_star * T, p))
        y0 = torch.reshape(y_guide[0:N_star, :], (N_star * T,))
        R = 0.01 * torch.eye(p)
        beta_init = torch.matmul(torch.inverse(torch.matmul(x0.T, x0) + R), torch.matmul(x0.T, y0))
        er1 = torch.sum(torch.square(y0 - torch.matmul(x0, beta_init))) / N
        x0 = torch.reshape(x_guide[-N_star:, :, :], (N_star * T, p))
        y0 = torch.reshape(y_guide[-N_star:, :], (N_star * T,))
        beta_end = torch.matmul(torch.inverse(torch.matmul(x0.T, x0) + R), torch.matmul(x0.T, y0))
        er2 = torch.sum(torch.square(y0 - torch.matmul(x0, beta_end))) / N
        er = (er1 + er2) / 2

        # compute the average gradient
        beta_grad_init = (beta_end - beta_init) / (N - 1)

        # initial learning coefficients
        w_init = torch.zeros((5,))

        intercept_map = pyro.param("intercept_map", meany.clone().detach())
        beta_grad_map = pyro.param("beta_grad_map", beta_grad_init.clone().detach())
        beta0_map = pyro.param("beta0_map", beta_init.clone().detach())
        sigma_map = pyro.param("sigma_map", er.clone().detach())
        w_map = pyro.param("w_map", w_init.clone().detach())

        pyro.sample("intercept", dist.Delta(intercept_map))
        pyro.sample("sigma", dist.Delta(sigma_map))
        for i in range(p):
            pyro.sample(f"beta_grad_{i}", dist.Delta(beta_grad_map[i]))
            pyro.sample(f"beta0_{i}", dist.Delta(beta0_map[i]))

        for i in range(self.ncoeff):
            pyro.sample(f"w_{i}", dist.Delta(w_map[i]))

    @staticmethod
    def train(model, guide, X_train_torch, y_train_torch, lr=0.001, n_steps=10001):
        """For training the model using stochastic variational inference (SVI)
        Args:
            model: encodes observations, latent random variables and parameters in the model
            guide: defines the variational distribution which serves as an approximation to the posterior
            lr: learning rate of model; = 0.001
            n_steps: number of steps/iterations; =  have used between 10001 - 40001
            X_train_torch: values of X to be used in training trials
            y_train_torch: values of Y to be used in training trials"""

        pyro.clear_param_store()
        adam = pyro.optim.Adam({"lr": lr})
        svi = SVI(model, guide, adam, loss=Trace_ELBO())

        for step in range(n_steps):
            loss = svi.step(X_train_torch, y_train_torch)
            if step % 200 == 0:
                print('[iter {}]  loss: {:.4f}'.format(step, loss))
