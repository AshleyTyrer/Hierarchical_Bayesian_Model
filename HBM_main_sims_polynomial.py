# Hierarchical Bayesian Modelling (HBM) of simulated neural data
# Contributors: Ashley Tyrer, Diego Vidaurre
# Centre of Functionally Integrative Neuroscience, Aarhus University
# Created 12-06-2022
# Edited by Ashley Tyrer, date of last edit: 01-02-2023

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyro
from itertools import product
from DataFormat_Saver import DataFormatSaver
from DataPlotter_Saver import DataPlotterSaver
from Maximum_aposteriori_model import MaximumAPosterioriModel
from SetParameters import SetParameters
import utils
from platform import python_version
assert pyro.__version__.startswith("1.8")  # I'm writing this tutorial with version


N, T, p = 200, 100, 3  # will change depending on real data used, will define data size if using synthetic data

# Cycle through different parameter combinations
w_params_list = [[1, 1, 1]]
sigma_params_to_use = [0.01]
run_num_list = [1]
# w_params_list = [[1, 1, 1], [0, 0, 0], [0, -1, 1, -1, 1], [0.25, -0.75, 0.5, 0.25, -0.25], [0.25, 0.5, 0.025],
#                  [0.2, 0.1, 0.3, 0.8, 0.5], [0.25, 0.25, 0.25]]
# sigma_params_to_use = [0.01, 0.1, 0.0001]
# run_num_list = [1, 2, 3, 4, 5, 6]

loop_through_this = product(w_params_list, sigma_params_to_use, run_num_list)

for w_sim, sigma_sim, run_num in loop_through_this:

    w_true = np.array(w_sim)
    sigma = sigma_sim

    ncoeff = len(w_true)

    alpha_shape = 'polynomial'

    map_model = MaximumAPosterioriModel(ncoeff, alpha_shape)

    est_params_alpha = np.zeros(N)
    est_params_beta_hat = np.zeros((N, p))
    est_params_wmap = np.zeros(ncoeff)

    # Data for the time-varying regression - generate random synthetic data
    X = np.random.normal(np.zeros((N, T, p)), np.ones((N, T, p)))

    setparam = SetParameters(ncoeff, alpha_shape, N)
    w_tensor = torch.tensor(w_sim, dtype=torch.float)
    alpha_true_ten = setparam.polynomial_alpha(w_tensor)
    alpha_true = alpha_true_ten.detach().numpy()

    for j in range(N):
        print(alpha_true[j])

    beta0 = np.random.normal(np.zeros((p,)), np.ones((p,)))
    beta_grad = np.random.normal(np.zeros((p,)), (1 / alpha_true.sum()) * np.ones((p,)))
    beta_true = setparam.calculate_beta(beta0, beta_grad, alpha_true, p)

    epsilon = sigma * np.random.normal(np.zeros((N, T)), np.ones((N, T)))
    y = setparam.calculate_y(X, beta_true, epsilon, T)

    X_train_torch = torch.tensor(X, dtype=torch.float)
    y_train_torch = torch.tensor(y, dtype=torch.float)

    dp = DataPlotterSaver(num_subs=1, which_model=1)

    fig = plt.figure()
    data_format = DataFormatSaver(N, T, p, run_num)
    dp.betas_heatmap_plotting(fig, beta_true, 2, 1, 1)
    dp.alpha_line_plotting(fig, alpha_true, 2, 1, 2)
    data_format.save_figure(w_true, sigma, 'true')
    plt.close()

    map_model.train(map_model.model_map, map_model.guide_map, X_train_torch, y_train_torch)
    pyro.param("sigma_map")

    beta_grad_map = pyro.param("beta_grad_map").detach().numpy()
    beta0_map = pyro.param("beta0_map").detach().numpy()

    w_map = pyro.param("w_map")
    alpha = setparam.polynomial_alpha(w_map)

    for j in range(N):
        print(alpha[j])

    beta_hat = setparam.calculate_beta(beta0_map, beta_grad_map, alpha, p)

    alphas_concat = [np.concatenate((alpha_true, alpha.detach().numpy()))]
    betas_concat = [np.concatenate((beta_true, beta_hat))]

    fig = plt.figure()
    dp.betas_heatmap_plotting(fig, beta_true, 2, 2, 1, betas_concat)
    dp.betas_heatmap_plotting(fig, beta_hat, 2, 2, 2, betas_concat)
    dp.alpha_line_plotting(fig, alpha_true, 2, 2, 3, alphas_concat)
    dp.alpha_line_plotting(fig, alpha.detach().numpy(), 2, 2, 4, alphas_concat)
    plt.tight_layout()
    data_format.save_figure(w_true, sigma, 'inferred')
    plt.close()

    beta_corr = np.corrcoef(beta_true.flatten(), beta_hat.flatten())[0, 1]

    sigma_map = pyro.param("sigma_map").detach().numpy()
    epsilon_map = sigma_map * np.random.normal(np.zeros((N, T)), np.ones((N, T)))
    y_map = setparam.calculate_y(X, beta_hat, epsilon_map, T)

    y_corr = np.corrcoef(y.flatten(), y_map.flatten())
    alpha_nump = alpha.detach().numpy()
    alpha_corr = np.corrcoef(alpha_true.flatten(), alpha_nump.flatten())

    model_accuracy = utils.compute_accuracy(y, y_map)

    est_params_sigma = sigma
    for j in range(N):
        est_params_alpha[j] = alpha[j]
        for chan in range(p):
            est_params_beta_hat[j, chan] = beta_hat[chan, j]

    for n in range(ncoeff):
        est_params_wmap[n] = w_map[n]

    data_format.save_model_pkl(
        w_true, sigma,
        [est_params_sigma, est_params_alpha, est_params_beta_hat, est_params_wmap, beta_corr, y_corr, alpha_corr,
         w_true, sigma, y, y_map, beta_true, alpha_true, X, model_accuracy])
