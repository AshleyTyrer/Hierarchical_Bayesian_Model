# Hierarchical Bayesian Modelling (HBM) of simulated neural data
# Contributors: Ashley Tyrer, Diego Vidaurre
# Centre of Functionally Integrative Neuroscience, Aarhus University
# Created 12-06-2022

import os
from functools import partial
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from sklearn.metrics import r2_score
from itertools import product
from pyro.infer import SVI, Trace_ELBO
from DataFormat_Sigmoid_Saver import DataFormatSigmoidSaver
from Maximum_aposteriori_model import MaximumAPosterioriModel
from DataPlotter_Saver import DataPlotterSaver
from SetParameters import SetParameters
import utils
from math import exp
from platform import python_version
assert pyro.__version__.startswith("1.8")  # I'm writing this tutorial with version


N, T, p = 200, 100, 6  # will change depending on real data used, will define data size if using synthetic data

# Cycle through different parameter combinations
w_params_list = [1]
sigma_params_to_use = [0.01]
midpoint_list = [1]
# w_params_list = [1, 2, 4, 8, 12, 16, 32]
# sigma_params_to_use = [0.01, 0.1, 0.0001]
# midpoint_list = [1, 1.2, 1.5, 1.8, 2]

loop_through_this = product(w_params_list, sigma_params_to_use, midpoint_list)

for w_sim, sigma_sim, midpoint_sim in loop_through_this:

    w_true = w_sim
    sigma = sigma_sim
    midpoint_true = midpoint_sim
    if w_true is list:
        ncoeff = len(w_true)
    else:
        ncoeff = 1

    alpha_shape = 'sigmoid'

    map_model = MaximumAPosterioriModel(ncoeff, alpha_shape)
    data_format = DataFormatSigmoidSaver()

    est_params_alpha = np.zeros(N)
    est_params_beta_hat = np.zeros((N, p))
    est_params_wmap = np.zeros(ncoeff)

    # Data for the time-varying regression - generate random synthetic data
    X = np.random.normal(np.zeros((N, T, p)), np.ones((N, T, p)))

    setparam = SetParameters(ncoeff, alpha_shape, N)
    w_tensor = torch.tensor([w_true, ], dtype=torch.float32)
    m_tensor = torch.tensor([midpoint_true, ], dtype=torch.float32)
    alpha_true_ten = setparam.sigmoid_alpha(w_tensor, m_tensor)
    alpha_true = alpha_true_ten.detach().numpy()

    for j in range(N):
        print(alpha_true[j])

    beta0 = np.random.normal(np.zeros((p,)), np.ones((p,)))
    beta_grad = np.random.normal(np.zeros((p,)), (1 / alpha_true.sum()) * np.ones((p,)))
    beta_true = np.zeros((p, N))
    beta_true[:, 0] = beta0

    for j in range(1, N):
        beta_true[:, j] = beta_true[:, j - 1] + alpha_true[j] * beta_grad

    epsilon = sigma * np.random.normal(np.zeros((N, T)), np.ones((N, T)))
    y = np.zeros((N, T))

    for j in range(N):
        for t in range(T):
            y[j, t] = np.dot(X[j, t, :], beta_true[:, j]) + epsilon[j, t]

    X_train_torch = torch.tensor(X, dtype=torch.float)
    y_train_torch = torch.tensor(y, dtype=torch.float)

    dp = DataPlotterSaver(num_subs=1, which_model=2)

    fig = plt.figure()
    dp.betas_heatmap_plotting(fig, beta_true, 2, 1, 1)
    dp.alpha_line_plotting(fig, alpha_true, 2, 1, 2)
    data_format.save_figure(w_true, sigma, midpoint_true, 'true')
    plt.close()

    map_model.train(map_model.model_map, map_model.guide_map, X_train_torch, y_train_torch)
    pyro.param("sigma_map")

    beta_grad_map = pyro.param("beta_grad_map").detach().numpy()
    beta0_map = pyro.param("beta0_map").detach().numpy()

    beta_hat = np.zeros(beta_true.shape)
    beta_hat[:, 0] = beta0_map

    for j in range(1, N):
        beta_hat[:, j] = beta_hat[:, j - 1] + beta_grad_map

    w_map = pyro.param("w_map")
    midpoint_map = pyro.param("midpoint_map")
    alpha = setparam.sigmoid_alpha(w_map, midpoint_map)

    for j in range(N):
        print(alpha[j])

    alphas_concat = [np.concatenate((alpha_true, alpha.detach().numpy()))]
    betas_concat = [np.concatenate((beta_true, beta_hat))]

    fig = plt.figure()
    dp.betas_heatmap_plotting(fig, beta_true, 2, 2, 1, betas_concat)
    dp.betas_heatmap_plotting(fig, beta_hat, 2, 2, 2, betas_concat)
    dp.alpha_line_plotting(fig, alpha_true, 2, 2, 3, alphas_concat)
    dp.alpha_line_plotting(fig, alpha.detach().numpy(), 2, 2, 4, alphas_concat)
    plt.tight_layout()
    data_format.save_figure(w_true, sigma, midpoint_true, 'inferred')
    plt.close()

    beta_corr = np.corrcoef(beta_true.flatten(), beta_hat.flatten())[0, 1]

    sigma_map = pyro.param("sigma_map").detach().numpy()
    epsilon_map = sigma_map * np.random.normal(np.zeros((N, T)), np.ones((N, T)))
    y_map = np.zeros((N, T))

    for j in range(N):
        for t in range(T):
            y_map[j, t] = np.dot(X[j, t, :], beta_hat[:, j]) + epsilon_map[j, t]

    y_corr = np.corrcoef(y.flatten(), y_map.flatten())
    alpha_corr = np.corrcoef(alpha_true.flatten(), alpha.flatten())

    model_accuracy = utils.compute_accuracy(y, y_map)
    fig = plt.figure()
    dp.accuracy_over_trial(model_accuracy)
    data_format.save_figure(w_true, sigma, midpoint_true, 'tp_acc')
    plt.close()

    est_params_sigma = sigma
    est_params_midpoint = midpoint_map.detach().numpy()
    for j in range(N):
        est_params_alpha[j] = alpha[j]
        for chan in range(p):
            est_params_beta_hat[j, chan] = beta_hat[chan, j]

    for n in range(ncoeff):
        est_params_wmap[n] = w_map[n].detach().numpy()

    data_format.save_model_pkl(
        w_true, sigma, midpoint_true,
        [est_params_sigma, est_params_alpha, est_params_beta_hat, est_params_wmap, beta_corr, y_corr, alpha_corr,
         est_params_midpoint, w_true, sigma, midpoint_true, y, y_map, beta_true, alpha_true, model_accuracy])
