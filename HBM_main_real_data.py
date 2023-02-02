# Hierarchical Bayesian Modelling (HBM) of neuroimaging data
# Contributors: Ashley Tyrer, Diego Vidaurre
# Centre of Functionally Integrative Neuroscience, Aarhus University
# Created 15-05-2022

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyro
from itertools import product
from DataLoader import DataLoader
from MaximumAPosterioriModel import MaximumAPosterioriModel
from DataPlotter import DataPlotter
from SetParameters import SetParameters
import utils
from platform import python_version
assert pyro.__version__.startswith("1.8")  # I'm writing this tutorial with version
matplotlib.use('Agg')


# Set options for dataset selection
dataset_name = 'Ca_imaging'
animals_list = [233]
task_alignment = 'Response'
max_point_list = ['Single', 'Split']
# signals_list = ['non_osc', 'osc']
# max_point = 'Split'
signal_types = 'allsignals'
decoding_method_suffix = 'reg'

# Set prior parameter values
w_true = np.array((0.25, 0.5, 0.025))
sigma = 0.01

loop_through_this = product(max_point_list, animals_list)

for max_point_select, animal_select in loop_through_this:

    max_point = max_point_select
    # signal_types = signal_types_select
    animal_number = animal_select

    if max_point == 'Single':
        trial_split = 'alltrials'
    elif max_point == 'Split':
        trial_split = 'concathalves'
    else:
        raise ValueError('Valid method of selecting max point needs to be defined')

    variable_suffix = '{sig_types}_singles_{t_split}_{dec_meth}'.format(
        sig_types=signal_types, t_split=trial_split, dec_meth=decoding_method_suffix)

    if dataset_name == 'Nadine':
        data_loader = DataLoader('single_timepoint_data', variable_suffix, max_point, dataset_name)
    elif dataset_name == 'Ca_imaging':
        data_loader = DataLoader('single_timepoint_data', variable_suffix, max_point, dataset_name, animal_number,
                                 task_alignment)
    else:
        raise ValueError('Valid dataset must be specified')

    subs_list = data_loader.get_data_files()
    no_subs = len(subs_list)

    for sub in range(no_subs):

        # Data for the time-varying regression - input real data
        sub_data_dict, sub_code = data_loader.load_data_for_subject(subs_list, sub)
        X = data_loader.format_data(sub_data_dict)

        print(sub_code, X.shape)
        N, T, p = X.shape

        ncoeff = len(w_true)
        alpha_shape = 'polynomial'
        map_model = MaximumAPosterioriModel(ncoeff, alpha_shape)

        est_params_alpha = np.zeros(N)
        est_params_beta_hat = np.zeros((N, p))
        est_params_wmap = np.zeros(ncoeff)

        setparam = SetParameters(ncoeff, alpha_shape, N)
        w_tensor = torch.tensor(w_true, dtype=torch.float)
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

        dp = DataPlotter(num_subs=1, which_model=alpha_shape)

        fig = plt.figure()
        dp.betas_heatmap_plotting(fig, beta_true, 2, 1, 1)
        dp.alpha_line_plotting(fig, alpha_true, 2, 1, 2)
        data_loader.save_figure(sub_code, 'true')
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
        data_loader.save_figure(sub_code, 'inferred')
        plt.close()

        beta_corr = np.corrcoef(beta_true.flatten(), beta_hat.flatten())[0, 1]

        sigma_map = pyro.param("sigma_map").detach().numpy()
        epsilon_map = sigma_map * np.random.normal(np.zeros((N, T)), np.ones((N, T)))
        y_map = setparam.calculate_y(X, beta_hat, epsilon_map, T)

        y_corr = np.corrcoef(y.flatten(), y_map.flatten())
        alpha_nump = alpha.detach().numpy()
        alpha_corr = np.corrcoef(alpha_true.flatten(), alpha_nump.flatten())

        model_accuracy = utils.compute_accuracy(y, y_map)

        fig = plt.figure()
        dp.accuracy_over_trial(model_accuracy)
        data_loader.save_figure(sub_code, 'tp_acc')
        plt.close()

        est_params_sigma = sigma_map
        for j in range(N):
            est_params_alpha[j] = alpha[j]
            for chan in range(p):
                est_params_beta_hat[j, chan] = beta_hat[chan, j]

        for n in range(ncoeff):
            est_params_wmap[n] = w_map[n]

        data_loader.save_model_pkl(
            sub_code,
            [est_params_sigma, est_params_alpha, est_params_beta_hat, est_params_wmap, beta_corr, y_corr, alpha_corr,
             alpha_true, beta_true, y, w_true, y_map, sigma, X, model_accuracy])
