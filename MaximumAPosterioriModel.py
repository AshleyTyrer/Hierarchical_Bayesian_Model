# Class: MaximumAPosterioriModel
# For defining MAP model, setting parameter estimates and sample distributions
# Contributors: Ashley Tyrer
# Centre of Functionally Integrative Neuroscience, Aarhus University
# Created 12-09-2022
# Edited by Ashley Tyrer, date of last edit: 02-02-2023

import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro.infer import SVI, Trace_ELBO
from typing import Dict, List, Optional
from platform import python_version
from SetParameters import SetParameters
assert pyro.__version__.startswith("1.8")  # I'm writing this tutorial with version


class MaximumAPosterioriModel:

    def __init__(self, num_coeff, alpha_shape: str):
        """For initialising the MaximumAPosterioriModel object
        Args:
            num_coeff: number of coefficients for modulating alpha shape, i.e. length of w
            alpha_shape: string stating whether alpha will be polynomial or sigmoid"""

        self.ncoeff = num_coeff
        self.alpha_shape = alpha_shape

    def model_map(self, x_model, y_model):
        """Encodes observations, latent random variables and parameters in the model
        Args:
            x_model: neural data in form: number of trials x timepoints x channels
            y_model: behaviour/continuous learning in form: number of trials x timepoints"""

        sigma_mod = 1
        N, T, p = x_model.shape  # trials, time, and channels/ROIs
        tau_grad, tau0 = (1 / N), 1  # variance of the beta distributions
        sigma_shape, sigma_rate = N * T, sigma_mod * N * T  # prior parameters for the noise distribution

        if self.alpha_shape == 'sigmoid':
            tau_w, tau_m = 8, 0.5
            midpoint = torch.zeros((self.ncoeff,))
            for i in range(self.ncoeff):
                midpoint[i] = pyro.sample(f"midpoint_{i}", dist.Normal(1.5, tau_m))
                # constrained to be within interval 0-3
        else:
            tau_w = 1

        # learning coefficients
        w = torch.zeros((self.ncoeff,))
        for i in range(self.ncoeff):
            w[i] = pyro.sample(f"w_{i}", dist.Normal(10.0, tau_w))
            # constrained to be within interval 0-100

        # learning weights per trial, computed using the learning coefficients
        setparam = SetParameters(self.ncoeff, self.alpha_shape, N)
        if self.alpha_shape == 'polynomial':
            alpha = setparam.polynomial_alpha(w)
        elif self.alpha_shape == 'sigmoid':
            alpha = setparam.sigmoid_alpha(w, midpoint)

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
        beta = setparam.calculate_beta(beta0, beta_grad, alpha, p)

        # mean prediction
        mean = torch.zeros((N, T)) + intercept
        mean[0, :] = mean[0, :] + torch.matmul(x_model[0, :, :], beta[:, 0])

        for j in range(1, N):
            mean[j, :] = mean[j, :] + torch.matmul(x_model[j, :, :], beta[:, j])

        mean = mean.flatten()
        y_model = y_model.flatten()

        with pyro.plate('data_plate'):
            pyro.sample("y", dist.Normal(mean, sigma), obs=y_model)

    def guide_map(self, x_guide, y_guide):
        """Defines the variational distribution which serves as an approximation to the posterior
        Args:
            x_guide: neural data in form: number of trials x timepoints x channels
            y_guide: behaviour/continuous learning in form: number of trials x timepoints"""

        N, T, p = x_guide.shape
        w_prior = 10.0
        midpoint_prior = 1.5

        # mean y
        meany = y_guide.flatten().mean()

        # compute beta in the first trials and in the last trials
        n_star = round(N / 10)
        x0 = torch.reshape(x_guide[0:n_star, :, :], (n_star * T, p))
        y0 = torch.reshape(y_guide[0:n_star, :], (n_star * T,))
        R = 0.01 * torch.eye(p)
        beta_init = torch.matmul(torch.inverse(torch.matmul(x0.T, x0) + R), torch.matmul(x0.T, y0))
        er1 = torch.sum(torch.square(y0 - torch.matmul(x0, beta_init))) / N
        x0 = torch.reshape(x_guide[-n_star:, :, :], (n_star * T, p))
        y0 = torch.reshape(y_guide[-n_star:, :], (n_star * T,))
        beta_end = torch.matmul(torch.inverse(torch.matmul(x0.T, x0) + R), torch.matmul(x0.T, y0))
        er2 = torch.sum(torch.square(y0 - torch.matmul(x0, beta_end))) / N
        er = (er1 + er2) / 2

        # compute the average gradient
        beta_grad_init = (beta_end - beta_init) / (N - 1)

        # initial learning coefficients
        # w_init = torch.zeros((self.ncoeff,))
        w_init = torch.tensor([w_prior])

        intercept_map = pyro.param("intercept_map", meany.clone().detach())
        beta_grad_map = pyro.param("beta_grad_map", beta_grad_init.clone().detach())
        beta0_map = pyro.param("beta0_map", beta_init.clone().detach())
        sigma_map = pyro.param("sigma_map", er.clone().detach())
        w_map = pyro.param("w_map", w_init.clone().detach(), constraint=constraints.interval(0.0, 100.0))
        if self.alpha_shape == 'sigmoid':
            # midpoint_init = torch.zeros((1,))
            midpoint_init = torch.tensor([midpoint_prior])
            midpoint_map = pyro.param("midpoint_map", midpoint_init.clone().detach(), constraint=constraints.interval(
                0.0, 3.0))

        pyro.sample("intercept", dist.Delta(intercept_map))
        pyro.sample("sigma", dist.Delta(sigma_map))
        for i in range(p):
            pyro.sample(f"beta_grad_{i}", dist.Delta(beta_grad_map[i]))
            pyro.sample(f"beta0_{i}", dist.Delta(beta0_map[i]))

        for i in range(self.ncoeff):
            pyro.sample(f"w_{i}", dist.Delta(w_map[i]))
            if self.alpha_shape == 'sigmoid':
                pyro.sample(f"midpoint_{i}", dist.Delta(midpoint_map[i]))

    @staticmethod
    def train(model, guide, x_train_torch, y_train_torch, lr=0.001, n_steps=10001):
        """For training the model using stochastic variational inference (SVI)
        Args:
            model: encodes observations, latent random variables and parameters in the model
            guide: defines the variational distribution which serves as an approximation to the posterior
            lr: learning rate of model; = 0.001
            n_steps: number of steps/iterations; =  have used between 10001 - 40001
            x_train_torch: values of X to be used in training trials
            y_train_torch: values of Y to be used in training trials"""

        pyro.clear_param_store()
        adam = pyro.optim.Adam({"lr": lr})
        svi = SVI(model, guide, adam, loss=Trace_ELBO())

        for step in range(n_steps):
            loss = svi.step(x_train_torch, y_train_torch)
            if step % 200 == 0:
                print('[iter {}]  loss: {:.4f}'.format(step, loss))
