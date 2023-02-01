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
import math
import time
import pyro.distributions as dist
from itertools import product
from pyro.infer import SVI, Trace_ELBO
from DataFormat_Saver_sigmoid import DataFormatSigmoidSaver
from DataFormat_Saver import DataFormatSaver
from DataPlotter_Saver import DataPlotterSaver
from platform import python_version
assert pyro.__version__.startswith("1.8")  # I'm writing this tutorial with version

which_model = 1
param_names = ['y', 'beta', 'alpha']
sigma_params_to_use = [0.01, 0.1, 0.0001]
N, T, p = 200, 100, 6
run_num = 5

if which_model == 1:
    w_params_list = [[1, 1, 1], [0, 0, 0], [0, -1, 1, -1, 1], [0.25, -0.75, 0.5, 0.25, -0.25], [0.25, 0.5, 0.025],
                     [0.2, 0.1, 0.3, 0.8, 0.5], [0.25, 0.25, 0.25]]
    midpoint_list = [np.nan]
    all_axis_ticks = [0, 1, 2, 3]
    no_scalar_params = 1
    # no_subs = (len(w_params_list) * len(sigma_params_to_use))
    # num_fig_splits = 1
    param_names.append('w')
    data_loader = DataFormatSaver(N, T, p, run_num)

elif which_model == 2:
    w_params_list = [1, 2, 4, 8, 12, 16, 32]
    midpoint_list = [1]  # [1, 1.2, 1.5, 1.8, 2]
    all_axis_ticks = [0, 1, 2]
    no_scalar_params = 3
    # no_subs = (len(w_params_list) * len(sigma_params_to_use) * len(midpoint_list))
    # num_fig_splits = len(midpoint_list)
    data_loader = DataFormatSigmoidSaver()
else:
    raise ValueError('Valid model number not entered')

no_subs = (len(w_params_list) * len(sigma_params_to_use) * len(midpoint_list))
no_params = len(param_names)

dp = DataPlotterSaver(no_subs, which_model)

param_set_labels_single_line = [None] * no_subs
param_set_labels_multi_lines = [None] * no_subs

heatmap_array = np.zeros((no_subs, no_params))
ratio = 1.0
num_fig_splits = len(midpoint_list)
col_num = 6
row_num = 4
num_panels = int(round(no_subs/num_fig_splits))

removed_axes = (col_num*row_num)-num_panels

loop_through_this = product(w_params_list, sigma_params_to_use, midpoint_list)

for sub_index, (w_sim, sigma, midpoint_tru) in enumerate(loop_through_this):
    w_true = np.array(w_sim)
    param_set_single_line, param_set_multi_lines = dp.axis_label_setting(w_true, sigma, midpoint_tru)
    ncoeff = len(w_true)

    param_set_labels_single_line[sub_index] = param_set_single_line
    param_set_labels_multi_lines[sub_index] = param_set_multi_lines

    data_objects = data_loader.load_model_pkl(w_true, sigma, midpoint_tru)

    if which_model == 1:
        [sigma_infs, alpha_infs, beta_infs, w_infs, beta_correlations, y_correlations, alpha_correlations, w_trues,
         sigma_trues, y_trues, y_infs, beta_trues, alpha_trues, w_correlations, p, N, T, beta_hat_1st_half,
         beta_hat_2nd_half] = dp.data_formatting(data_objects, sub_index)
    elif which_model == 2:
        [sigma_infs, alpha_infs, beta_infs, w_infs, beta_correlations, y_correlations, alpha_correlations,
         midpoint_infs, w_trues, sigma_trues, midpoint_trues, y_trues, y_infs, beta_trues,
         alpha_trues, p, N, T, beta_hat_1st_half, beta_hat_2nd_half] = dp.data_formatting(data_objects, sub_index)
    else:
        raise ValueError('Valid model number not entered')

    # p, N = beta_hat_all[sub_index].shape
    # T = y_trues[sub_index].shape[1]

corrs_dict = {0: y_correlations, 1: beta_correlations, 2: alpha_correlations, 3: []}

min_max_list = [np.nan] * no_params

if which_model == 1:
    corrs_dict[3] = w_correlations

for sub_index in range(no_subs):
    for hm in range(no_params):
        heatmap_array[sub_index, hm] = corrs_dict[hm][sub_index]

for hm in range(no_params):
    min_max_list[hm] = corrs_dict[hm]

min_inferred = np.min(min_max_list)
max_inferred = np.max(min_max_list)
y_marg_inf = (max_inferred - min_inferred) * plt.margins()[1]

min_betas_1st, max_betas_1st, y_marg_beta_1st = dp.calculate_plot_min_max(beta_hat_1st_half)
min_betas_2nd, max_betas_2nd, y_marg_beta_2nd = dp.calculate_plot_min_max(beta_hat_2nd_half)

for k in range(num_fig_splits):
    # fig = plt.figure(figsize=(15, 10))
    fig, axes = plt.subplots(nrows=row_num, ncols=col_num, figsize=(15, 10))
    axes = axes.flatten()

    for params_index in range(num_panels):
        big_ind = (num_panels * k) + params_index
        betas_1st_half_plot = beta_hat_1st_half[big_ind]
        betas_2nd_half_plot = beta_hat_2nd_half[big_ind]

        if len(beta_hat_1st_half[big_ind]) > len(beta_hat_2nd_half[big_ind]):
            betas_1st_half_plot = betas_1st_half_plot[:-1, :]

        title_beta_corr = round(np.corrcoef(betas_1st_half_plot.flatten(), betas_2nd_half_plot.flatten())[0, 1], 3)
        # ax = fig.add_subplot(4, 6, math.floor((params_index/(k+1)) + 1))
        ax = axes[params_index]
        ax.scatter(betas_1st_half_plot, betas_2nd_half_plot, s=2, marker='.')
        ax.set_title(str(title_beta_corr))
        ax.set_xlabel('Betas 1st Half')
        ax.set_ylabel('Betas 2nd Half')
        ax.set_xlim(min_betas_1st - y_marg_beta_1st, max_betas_1st + y_marg_beta_1st)
        ax.set_ylim(min_betas_2nd - y_marg_beta_2nd, max_betas_2nd + y_marg_beta_2nd)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=1.0, hspace=1.0)

    for z in range(removed_axes):
        ax_rem = ((col_num*row_num)-1)-z
        axes[ax_rem].set_axis_off()

    plt.suptitle('Correlations Between Betas in 1st vs 2nd Half Trials, Simulated Data {num}'.format(num=k+1))
    # data_loader.save_all_subs_figure('beta_hats', '1st_vs_2nd_half_scatter_{num}'.format(num=k+1))
    plt.close()

betas_dict = {'Betas_1st_half': beta_hat_1st_half, 'Betas_2nd_half': beta_hat_2nd_half}
alphas_dict = {'Alphas_inferred': alpha_infs, 'Alphas_true': alpha_trues}
# data_loader.save_model_mat(betas_dict, 'Betas_1st_vs_2nd_half')
# data_loader.save_model_mat(alphas_dict, 'Alphas_true_and_inferred')

for k in range(num_fig_splits):
    # fig = plt.figure(figsize=(15, 10))
    fig, axes = plt.subplots(nrows=row_num, ncols=col_num, figsize=(15, 10))
    axes = axes.flatten()
    for xxx in range(num_panels):

        big_ind = (num_panels * k) + xxx
        beta_corr_plot = beta_correlations[big_ind]
        alpha_corr_plot = alpha_correlations[big_ind]
        y_corr_plot = y_correlations[big_ind]
        w_corr_plot = w_correlations[big_ind]
        title_bar_plot = str(param_set_labels_multi_lines[big_ind])
        # ax = fig.add_subplot(4, 6, math.floor((params_index/(k+1)) + 1))
        # subplot_num = math.floor(params_index / (k + 1))
        ax = axes[xxx]
        ax.cla()
        correlations_vec = [y_corr_plot, beta_corr_plot, alpha_corr_plot]
        if which_model == 1:
            correlations_vec.append(w_corr_plot)
        ax.bar(param_names, correlations_vec)
        ax.set_title(title_bar_plot, size=7)
        # ax.title.set_text(title_bar_plot)
        ax.set_ylabel('True vs Inferred corr.')
        ax.set_ylim(-1, max_inferred + y_marg_inf)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=1.0, hspace=1.0)

    for z in range(removed_axes):
        ax_rem = ((col_num*row_num)-1)-z
        axes[ax_rem].set_axis_off()
    plt.suptitle('Correlations Between True and Inferred Parameter Values, Simulated Data {num}'.format(num=k+1))
    # data_loader.save_all_subs_figure('all_params', 'bar_plot_{num}'.format(num=k+1))
    # plt.close('all')
    plt.close()

ticks_list = list(range(no_subs))

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1)
pos = ax.imshow(heatmap_array, cmap='seismic')
pos.set_clim(vmin=-1, vmax=1)
cbar = fig.colorbar(pos)
ax.set_xticks(all_axis_ticks)
ax.set_yticks(ticks_list)
ax.set_xticklabels(param_names)
ax.set_yticklabels(param_set_labels_single_line, size=4)
ax.set_ylabel('Parameter Sets')
cbar.set_label('Correlation')
x_left, x_right = ax.get_xlim()
y_bottom, y_top = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left) / (y_bottom - y_top)) * ratio)
fig.suptitle('Correlations Between True and Inferred Parameter Values, Simulated Data, Run {run}'.format(run=run_num))
# data_loader.save_all_subs_figure('all_params', 'heatmap')
# plt.close()

sigma_tru_np = np.zeros(no_subs)
sigma_inf_np = np.zeros(no_subs)
w_tru_np = np.zeros((no_subs, 5))
w_inf_np = np.zeros((no_subs, 5))
if which_model == 2:
    midpoint_tru_np = np.zeros(no_subs)
    midpoint_inf_np = np.zeros(no_subs)

for i in range(no_subs):
    sigma_tru_np[i] = sigma_trues[i]
    sigma_inf_np[i] = sigma_infs[i]
    for j in range(ncoeff):
        w_tru_np[i, j] = w_trues[i][j]
        w_inf_np[i, j] = w_infs[i][j]
    if which_model == 2:
        midpoint_tru_np[i] = midpoint_trues[i]
        midpoint_inf_np[i] = midpoint_infs[i]

fig = plt.figure(figsize=(15, 5))
for g in range(no_scalar_params):
    if g == 0:
        param = 'sigma'
        scatters_true = sigma_tru_np
        scatters_infs = sigma_inf_np
    elif g == 1:
        param = 'w'
        scatters_true = w_tru_np
        scatters_infs = w_inf_np
    elif g == 2:
        param = 'midpoint'
        scatters_true = midpoint_tru_np
        scatters_infs = midpoint_inf_np

    cols_sing = len(scatters_true)
    plot_col_singles = np.arange(cols_sing)
    ax = fig.add_subplot(1, no_scalar_params, g+1)
    plt.scatter(scatters_true, scatters_infs, s=100, marker='.')  # , c=plot_col_singles, cmap='inferno')
    plt.xlabel('True Values')
    plt.ylabel('Inferred Values')
    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_bottom - y_top)) * ratio)
    title_singles_corr = round(np.corrcoef(scatters_true.flatten(), scatters_infs.flatten())[0, 1], 3)
    if math.isnan(title_singles_corr):
        title_singles_corr = 0
    plt.title('{par_name}, corr = {title_corr}'.format(par_name=param, title_corr=title_singles_corr))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=1.0, hspace=1.0)

param_fig_title = 'All Subjects Correlations Between True and Inferred Values'
fig.suptitle(param_fig_title)
# data_loader.save_all_subs_figure('single_params', 'scatter')
plt.close()

beta_trues_tr = beta_trues
beta_infs_tr = beta_infs

for i in range(no_subs):
    beta_trues[i] = np.transpose(beta_trues_tr[i])
    beta_infs[i] = np.transpose(beta_infs_tr[i])

for param in range(no_params):
    param_label = param_names[param]
    trues_plot_str = '{p_label}_trues'.format(p_label=param_label)
    infs_plot_str = '{p_label}_infs'.format(p_label=param_label)
    corrs_plot_str = '{p_label}_correlations'.format(p_label=param_label)
    trues_plot = eval(trues_plot_str)
    infs_plot = eval(infs_plot_str)
    corrs_plot = eval(corrs_plot_str)

    if param_label == 'w':
        point_size = 100
        cbar_label = 'w values'
    else:
        point_size = 20
        cbar_label = 'Trial Number'

    min_params_true, max_params_true, y_marg_params_true = dp.calculate_plot_min_max(trues_plot)
    min_params_infs, max_params_infs, y_marg_params_infs = dp.calculate_plot_min_max(infs_plot)

    for k in range(num_fig_splits):
        fig, axes = plt.subplots(nrows=row_num, ncols=col_num, figsize=(15, 10))
        axes = axes.flatten()

        for sub in range(num_panels):
            big_ind = (num_panels * k) + sub
            cols = trues_plot[big_ind].shape[0]
            plot_cols = np.arange(cols)
            # title_param_corr = round(np.corrcoef(trues_plot[sub].flatten(), infs_plot[sub].flatten())[0, 1], 3)
            title_param_corr = round(corrs_plot[big_ind], 3)
            if math.isnan(title_param_corr):
                title_param_corr = 0
            # ax = fig.add_subplot(4, 6, math.floor((sub/(k+1)) + 1))
            ax = axes[sub]
            if param_label == 'beta':
                for h in range(p):
                    rainbow_scat = ax.scatter(trues_plot[big_ind][:, h], infs_plot[big_ind][:, h], s=point_size,
                                              marker='.', c=plot_cols, cmap='inferno')
            elif param_label == 'y':
                for i in range(T):
                    rainbow_scat = ax.scatter(trues_plot[big_ind][:, i], infs_plot[big_ind][:, i], s=point_size,
                                              marker='.', c=plot_cols, cmap='inferno')
            else:
                rainbow_scat = ax.scatter(trues_plot[big_ind], infs_plot[big_ind], s=point_size, marker='.',
                                          c=plot_cols, cmap='inferno')
            ax.set_title(str(title_param_corr))
            # ax.title.set_text(param_set_labels_multi_lines[params_index])
            ax.set_xlabel('True Values')
            ax.set_ylabel('Inferred Values')
            ax.set_xlim(min_params_true - y_marg_params_true, max_params_true + y_marg_params_true)
            ax.set_ylim(min_params_infs - y_marg_params_infs, max_params_infs + y_marg_params_infs)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=1.0, hspace=1.0)

            cbar = fig.colorbar(rainbow_scat, ax=ax)
            cbar.set_label(cbar_label)

        for z in range(removed_axes):
            ax_rem = ((col_num * row_num) - 1) - z
            axes[ax_rem].set_axis_off()

        param_fig_title = 'All Subjects Correlations Between True and Inferred {p_label} Values, {num}'.format(
            p_label=param_label, num=k+1)
        fig.suptitle(param_fig_title)
        # data_loader.save_all_subs_figure(param_label, 'scatter_{num}'.format(num=k+1))
        # plt.close('all')
        plt.close()

run_nums_list = [1, 2, 3, 4, 5, 6]
num_runs = len(run_nums_list)

data_objs_per_sub = [np.nan] * no_subs
data_objects_list = [data_objs_per_sub.copy()] * num_runs
big_data_structure = [np.nan] * num_runs
all_runs_X_data = [np.nan] * num_runs
big_corr_histogram = np.zeros((no_params, num_runs, no_subs))
big_corrs_dict = {0: 'y_corr_ind', 1: 'beta_corr_ind', 2: 'alpha_corr_ind', 3: []}

if which_model == 1:
    big_corrs_dict[3] = 'w_corr_ind'

for rn in range(num_runs):
    run = run_nums_list[rn]
    d_obj = DataFormatSaver(N, T, p, run)

    loop_through_this_again = product(w_params_list, sigma_params_to_use, midpoint_list)
    d_fp = DataPlotterSaver(no_subs, which_model)
    formatted_data = [np.nan] * 19

    for sub_index, (w_sim, sigma, midpoint_tru) in enumerate(loop_through_this_again):
        w_true = np.array(w_sim)
        ncoeff = len(w_true)
        data_objects_list[rn][sub_index] = d_obj.load_model_pkl(w_true, sigma, midpoint_tru)
        formatted_data = d_fp.data_formatting(data_objects_list[rn][sub_index], sub_index)

    big_data_structure[rn] = d_fp.multi_run_data_format()
    all_runs_X_data[rn] = d_fp.X_data

for prm in range(no_params):
    for sub in range(no_subs):
        for rn in range(num_runs):
            big_corr_histogram[prm, rn, sub] = big_data_structure[rn][big_corrs_dict[prm]][sub]

corr_coeffs_dict = {'Corr_Coeffs': big_corr_histogram}
data_loader.save_model_mat(corr_coeffs_dict, 'All_Params_Corr_Coeffs')

bin_edges = np.linspace(-1, 1, 50)

for count, sig_loop in enumerate(sigma_params_to_use):
    red_sub = np.linspace((0+count), (18+count), 7)
    y_marg_hist = (1 - -1) * plt.margins()[1]
    fig, axes = plt.subplots(nrows=len(red_sub), ncols=no_params, figsize=(12, 10), squeeze=False)
    for prm in range(no_params):
        for sub, red_ind in enumerate(red_sub):
            single_hist = np.zeros(num_runs)
            red_ind = int(red_ind)
            for rn in range(num_runs):
                single_hist[rn] = big_corr_histogram[prm, rn, red_ind]
            ax = axes[sub, prm]
            ax.hist(single_hist, bins=bin_edges, align='mid')
            ax.set_xlim(-1 - y_marg_hist, 1 + y_marg_hist)
            ax.set_ylim(0, num_runs)
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            if prm == 0:
                w_label_str = 'w = {w_list}'.format(w_list=str(w_params_list[sub]).replace('[', '').replace(']', ''))
                ax.set_ylabel(w_label_str, rotation=60, size=8)
                ax.yaxis.set_label_coords(-0.3, 0.5)
            if sub == len(red_sub)-1:
                ax.set_xlabel(param_names[prm])
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=1.0, hspace=1.0)
    histogram_fig_title = 'Correlations Between True and Inferred Parameter Values Over All Runs, Sigma = {sig_val}'.format(
        sig_val=sig_loop)
    fig.suptitle(histogram_fig_title)
    plt.tight_layout(pad=2.2)
    # plt.close()
    # data_loader.save_all_subs_figure('all_params', 'histograms_sig_{sig}'.format(sig=sig_loop))
