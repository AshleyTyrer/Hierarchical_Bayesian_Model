# Hierarchical Bayesian Modelling (HBM) of simulated neural data
# Contributors: Ashley Tyrer, Diego Vidaurre
# Centre of Functionally Integrative Neuroscience, Aarhus University
# Created 12-06-2022
# Edited by Ashley Tyrer, date of last edit: 02-02-2023

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyro
from itertools import product
from DataFormatSaverSigmoid import DataFormatSaverSigmoid
from MaximumAPosterioriModel import MaximumAPosterioriModel
from DataPlotter import DataPlotter
from SetParameters import SetParameters
import utils
from platform import python_version
assert pyro.__version__.startswith("1.8")  # I'm writing this tutorial with version

matplotlib.use('Agg')
N, T, p = 200, 100, 6  # will change depending on real data used, will define data size if using synthetic data

# Cycle through different parameter combinations
w_params_list = [1]
sigma_params_to_use = [0.01]
midpoint_list = [1]
run_num_list = [1]
# w_params_list = [1, 2, 4, 8, 12, 16, 32]
# sigma_params_to_use = [0.01, 0.1, 0.0001]
# midpoint_list = [1, 1.2, 1.5, 1.8, 2]
# run_num_list = [1, 2, 3, 4, 5, 6]

loop_through_this = product(w_params_list, sigma_params_to_use, midpoint_list, run_num_list)

for w_sim, sigma_sim, midpoint_sim, run_num in loop_through_this:

    w_true = w_sim
    sigma = sigma_sim
    midpoint_true = midpoint_sim
    if w_true is list:
        ncoeff = len(w_true)
    else:
        ncoeff = 1

    alpha_shape = 'sigmoid'

    map_model = MaximumAPosterioriModel(ncoeff, alpha_shape)
    data_format = DataFormatSaverSigmoid(N, T, p, run_num)

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

    beta_true = setparam.calculate_beta(beta0, beta_grad, alpha_true, p)
    epsilon = sigma * np.random.normal(np.zeros((N, T)), np.ones((N, T)))
    y = setparam.calculate_y(X, beta_true, epsilon, T)

    X_train_torch = torch.tensor(X, dtype=torch.float)
    y_train_torch = torch.tensor(y, dtype=torch.float)

    dp = DataPlotter(num_subs=1, which_model=alpha_shape)

    fig = plt.figure()
    dp.betas_heatmap_plotting(fig, beta_true, 2, 1, 1)
    dp.alpha_line_plotting(fig, alpha_true, 2, 1, 2)
    data_format.save_figure(w_true, sigma, midpoint_true, 'true')
    plt.close()

    map_model.train(map_model.model_map, map_model.guide_map, X_train_torch, y_train_torch, lr=0.0001)
    pyro.param("sigma_map")

    beta_grad_map = pyro.param("beta_grad_map").detach().numpy()
    beta0_map = pyro.param("beta0_map").detach().numpy()

    w_map = pyro.param("w_map")
    midpoint_map = pyro.param("midpoint_map")
    alpha = setparam.sigmoid_alpha(w_map, midpoint_map)

    for j in range(N):
        print(alpha[j])

    alpha_np = alpha.detach().numpy()
    beta_hat = setparam.calculate_beta(beta0_map, beta_grad_map, alpha_np, p)

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
    y_map = setparam.calculate_y(X, beta_hat, epsilon_map, T)

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
