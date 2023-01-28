# Hierarchical Bayesian Modelling (HBM) results plotting and analysis
# Contributors: Ashley Tyrer, Diego Vidaurre
# Centre of Functionally Integrative Neuroscience, Aarhus University
# Created 14-06-2022

import os
from functools import partial
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from DataLoader import DataLoader
from platform import python_version
assert pyro.__version__.startswith("1.8")  # I'm writing this tutorial with version

dataset_name = 'Ca_imaging'

animal_number = 233
task_alignment = 'Response'

max_point = 'Split'
signal_types = 'allsignals'
decoding_method_suffix = 'reg'

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
    sn = 5
elif dataset_name == 'Ca_imaging':
    data_loader = DataLoader('single_timepoint_data', variable_suffix, max_point, dataset_name, animal_number,
                             task_alignment)
    sn = 4
else:
    raise ValueError('Valid dataset must be specified')

subs_list = data_loader.get_data_files()
no_subs = len(subs_list)
param_names = ['y', 'beta', 'alpha', 'w']
no_params = len(param_names)

sub_codes_list = [None] * no_subs
beta_hat_all = [None] * no_subs
beta_hat_1st_half = [None] * no_subs
beta_hat_2nd_half = [None] * no_subs
beta_correlations = np.zeros(no_subs)
alpha_correlations = np.zeros(no_subs)
y_correlations = np.zeros(no_subs)
w_correlations = np.zeros(no_subs)
y_trues = [None] * no_subs
alpha_trues = [None] * no_subs
w_trues = [None] * no_subs
sigma_trues = [None] * no_subs
beta_trues = [None] * no_subs
w_infs = [None] * no_subs
sigma_infs = [None] * no_subs
y_infs = [None] * no_subs
alpha_infs = [None] * no_subs
heatmap_array = np.zeros((no_subs, no_params))
ratio = 1.0

fig = plt.figure(figsize=(15, 10))

for sub in range(no_subs):
    sub_data_dict, sub_code = data_loader.load_data_for_subject(subs_list, sub)
    data_objects = data_loader.load_model_pkl(sub_code)
    sub_codes_list[sub] = sub_code
    beta_hat_all[sub] = data_objects[2]
    beta_hat_all[sub] = beta_hat_all[sub].transpose(1, 0)
    beta_correlations[sub] = data_objects[4]
    p, N = beta_hat_all[sub].shape

    y_corr_arr = data_objects[5]
    alpha_corr_arr = data_objects[6]

    y_correlations[sub] = y_corr_arr[0, 1]
    alpha_correlations[sub] = alpha_corr_arr[0, 1]

    alpha_trues[sub] = data_objects[7]
    beta_trues[sub] = data_objects[8]
    y_trues[sub] = data_objects[9]
    w_trues[sub] = data_objects[10]
    sigma_trues[sub] = data_objects[12]

    sigma_infs[sub] = data_objects[0]
    alpha_infs[sub] = data_objects[1]
    w_infs[sub] = data_objects[3]
    y_infs[sub] = data_objects[11]

    w_correlations[sub] = np.corrcoef(w_trues[sub].flatten(), w_infs[sub].flatten())[0, 1]

    heatmap_array[sub, 0] = y_correlations[sub]
    heatmap_array[sub, 1] = beta_correlations[sub]
    heatmap_array[sub, 2] = alpha_correlations[sub]
    heatmap_array[sub, 3] = w_correlations[sub]

    # beta_hat_sub = np.zeros(shape=(p, N))
    betas_1st_half = np.zeros(shape=(round(N / 2), p))
    betas_2nd_half = np.zeros(shape=(round(N / 2) - 1, p))

    beta_hat_sub = beta_hat_all[sub]

    for chan in range(p):
        for n in range(round(N/2)):
            betas_1st_half[n, chan] = beta_hat_sub[chan, n]
        for n in range((round(N/2))-1):
            betas_2nd_half[n, chan] = beta_hat_sub[chan, round(N/2)+n]

    beta_hat_1st_half[sub] = betas_1st_half
    beta_hat_2nd_half[sub] = betas_2nd_half

    ax = fig.add_subplot(sn, sn, sub+1)
    pos = ax.imshow(beta_hat_all[sub])
    cbar = fig.colorbar(pos)
    cbar.set_label('Correlation')
    ax.set_xlabel('Trials')
    ax.set_ylabel('Channels')
    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_bottom - y_top)) * ratio)
    if N >= 200:
        ax.set_xticks([0, 100, 200])
    elif 100 <= N < 200:
        ax.set_xticks([0, 100])
    elif 50 <= N < 100:
        ax.set_xticks([0, 50])
    elif N < 50:
        ax.set_xticks([0, 25])
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

plt.suptitle('All Subjects Inferred Betas')
data_loader.save_all_subs_figure('beta_hats', 'inferred')
# plt.close()

min_inferred = np.min([y_correlations, beta_correlations, alpha_correlations, w_correlations])
max_inferred = np.max([y_correlations, beta_correlations, alpha_correlations, w_correlations])
beta_1st_half_np_min = np.zeros(no_subs)
beta_2nd_half_np_min = np.zeros(no_subs)
beta_1st_half_np_max = np.zeros(no_subs)
beta_2nd_half_np_max = np.zeros(no_subs)

for sub in range(no_subs):
    beta_1st_half_np_sing = beta_hat_1st_half[sub]
    beta_2nd_half_np_sing = beta_hat_2nd_half[sub]
    beta_1st_half_np_min[sub] = np.min(beta_1st_half_np_sing)
    beta_2nd_half_np_min[sub] = np.min(beta_2nd_half_np_sing)
    beta_1st_half_np_max[sub] = np.max(beta_1st_half_np_sing)
    beta_2nd_half_np_max[sub] = np.max(beta_2nd_half_np_sing)

min_betas_1st = np.min(beta_1st_half_np_min)
min_betas_2nd = np.min(beta_2nd_half_np_min)
max_betas_1st = np.max(beta_1st_half_np_max)
max_betas_2nd = np.max(beta_2nd_half_np_max)
y_marg_inf = (max_inferred - min_inferred) * plt.margins()[1]
y_marg_beta_1st = (max_betas_1st - min_betas_1st) * plt.margins()[1]
y_marg_beta_2nd = (max_betas_2nd - min_betas_2nd) * plt.margins()[1]

fig = plt.figure(figsize=(15, 10))

for sub in range(no_subs):
    betas_1st_half_plot = beta_hat_1st_half[sub]
    betas_2nd_half_plot = beta_hat_2nd_half[sub]

    if len(beta_hat_1st_half[sub]) > len(beta_hat_2nd_half[sub]):
        betas_1st_half_plot = betas_1st_half_plot[:-1, :]

    title_beta_corr = round(np.corrcoef(betas_1st_half_plot.flatten(), betas_2nd_half_plot.flatten())[0, 1], 3)
    ax = fig.add_subplot(sn, sn, sub + 1)
    plt.scatter(betas_1st_half_plot, betas_2nd_half_plot, s=2, marker='.')
    ax.title.set_text(str(title_beta_corr))
    plt.xlabel('Betas 1st Half')
    plt.ylabel('Betas 2nd Half')
    plt.xlim(min_betas_1st - y_marg_beta_1st, max_betas_1st + y_marg_beta_1st)
    plt.ylim(min_betas_2nd - y_marg_beta_2nd, max_betas_2nd + y_marg_beta_2nd)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=1.0, hspace=1.0)

plt.suptitle('All Subjects Correlations Between Betas in 1st vs 2nd Half Trials')
data_loader.save_all_subs_figure('beta_hats', '1st_vs_2nd_half_scatter')
# plt.close()

betas_dict = {'Betas_1st_half': beta_hat_1st_half, 'Betas_2nd_half': beta_hat_2nd_half}
data_loader.save_model_mat(betas_dict, 'Betas_1st_vs_2nd_half')

fig = plt.figure(figsize=(15, 10))

for sub in range(no_subs):
    beta_corr_plot = beta_correlations[sub]
    alpha_corr_plot = alpha_correlations[sub]
    y_corr_plot = y_correlations[sub]
    w_corr_plot = w_correlations[sub]
    sub_code_plot = sub_codes_list[sub]
    title_bar_plot = 'S{sub_id}'.format(sub_id=sub_code_plot)

    ax = fig.add_subplot(sn, sn, sub + 1)
    correlations_vec = [y_corr_plot, beta_corr_plot, alpha_corr_plot, w_corr_plot]
    ax.bar(param_names, correlations_vec)
    ax.title.set_text(title_bar_plot)
    ax.set_xticklabels(param_names, size=8)
    plt.ylabel('True vs Inferred corr.')
    plt.ylim(-1, max_inferred + y_marg_inf)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=1.0, hspace=1.0)
    # ax.set_xticks([0, 1, 2])
    # plt.xlim(-1, 3)
    # x_left, x_right = ax.get_xlim()
    # y_bottom, y_top = ax.get_ylim()
    # ax.set_aspect(abs((x_right - x_left) / (y_bottom - y_top)) * ratio)

plt.suptitle('All Subjects Correlations Between True and Inferred Parameter Values')
data_loader.save_all_subs_figure('all_params', 'bar_plot')
# plt.close()

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1)
pos = ax.imshow(heatmap_array, cmap='seismic')
pos.set_clim(vmin=-1, vmax=1)
cbar = fig.colorbar(pos)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(param_names)
ax.set_ylabel('Subjects')
cbar.set_label('Correlation')
x_left, x_right = ax.get_xlim()
y_bottom, y_top = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left) / (y_bottom - y_top)) * ratio)
fig.suptitle('All Subjects Correlations Between True and Inferred Parameter Values')
data_loader.save_all_subs_figure('all_params', 'heatmap')
# plt.close()

sigma_tru_np = np.zeros(no_subs)
sigma_inf_np = np.zeros(no_subs)
for i in range(no_subs):
    sigma_tru_np[i] = sigma_trues[i]
    sigma_inf_np[i] = sigma_infs[i]

title_sigma_corr = round(np.corrcoef(sigma_tru_np.flatten(), sigma_inf_np.flatten())[0, 1], 3)

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1)
plt.scatter(sigma_tru_np, sigma_inf_np, s=100, marker='.')
plt.xlabel('True Values')
plt.ylabel('Inferred Values')
param_fig_title = 'All Subjects Correlations Between True and Inferred Sigma Values, corr = {sig_corr}'.format(
    sig_corr=title_sigma_corr)
fig.suptitle(param_fig_title)
data_loader.save_all_subs_figure('sigma', 'scatter')

beta_infs = beta_hat_all
beta_trues_tr = beta_trues
beta_infs_tr = beta_infs

for i in range(no_subs):
    beta_trues[i] = np.transpose(beta_trues_tr[i])
    beta_infs[i] = np.transpose(beta_infs_tr[i])

for param in range(no_params):
    param_label = param_names[param]
    trues_plot_str = '{p_label}_trues'.format(p_label=param_label)
    infs_plot_str = '{p_label}_infs'.format(p_label=param_label)
    trues_plot = eval(trues_plot_str)
    infs_plot = eval(infs_plot_str)

    params_true_np_min = np.zeros(no_subs)
    params_infs_np_min = np.zeros(no_subs)
    params_true_np_max = np.zeros(no_subs)
    params_infs_np_max = np.zeros(no_subs)

    if param_label == 'w':
        point_size = 100
    else:
        point_size = 20

    for sub in range(no_subs):
        params_true_np_sing = trues_plot[sub]
        params_infs_np_sing = infs_plot[sub]
        params_true_np_min[sub] = np.min(params_true_np_sing)
        params_infs_np_min[sub] = np.min(params_infs_np_sing)
        params_true_np_max[sub] = np.max(params_true_np_sing)
        params_infs_np_max[sub] = np.max(params_infs_np_sing)

    min_params_true = np.min(params_true_np_min)
    min_params_infs = np.min(params_infs_np_min)
    max_params_true = np.max(params_true_np_max)
    max_params_infs = np.max(params_infs_np_max)
    y_marg_inf = (max_inferred - min_inferred) * plt.margins()[1]
    y_marg_params_true = (max_params_true - min_params_true) * plt.margins()[1]
    y_marg_params_infs = (max_params_infs - min_params_infs) * plt.margins()[1]

    fig = plt.figure(figsize=(15, 10))

    for sub in range(no_subs):
        cols = trues_plot[sub].shape[0]
        plot_cols = np.arange(cols)
        title_param_corr = round(np.corrcoef(trues_plot[sub].flatten(), infs_plot[sub].flatten())[0, 1], 3)
        ax = fig.add_subplot(sn, sn, sub + 1)
        if param_label == 'beta':
            p = trues_plot[sub].shape[1]
            for k in range(p):
                rainbow_scat = plt.scatter(trues_plot[sub][:, k], infs_plot[sub][:, k], s=point_size, marker='.',
                                           c=plot_cols, cmap='inferno')
        elif param_label == 'w':
            plt.scatter(trues_plot[sub], infs_plot[sub], s=point_size, marker='.')
        else:
            rainbow_scat = plt.scatter(trues_plot[sub], infs_plot[sub], s=point_size, marker='.', c=plot_cols,
                                       cmap='inferno')
        ax.title.set_text(str(title_param_corr))
        plt.xlabel('True Values')
        plt.ylabel('Inferred Values')
        plt.xlim(min_params_true - y_marg_params_true, max_params_true + y_marg_params_true)
        plt.ylim(min_params_infs - y_marg_params_infs, max_params_infs + y_marg_params_infs)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=1.0, hspace=1.0)
        if param_label != 'w':
            cbar = fig.colorbar(rainbow_scat)
            cbar.set_label('Trial Number')

    param_fig_title = 'All Subjects Correlations Between True and Inferred {p_label} Values'.format(p_label=param_label)
    fig.suptitle(param_fig_title)
    data_loader.save_all_subs_figure(param_label, 'scatter')
