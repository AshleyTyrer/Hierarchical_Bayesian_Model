# Hierarchical Bayesian Modelling (HBM) of neuroimaging data
# Contributors: Ashley Tyrer, Diego Vidaurre
# Centre of Functionally Integrative Neuroscience, Aarhus University
# Created 15-05-2022

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from itertools import product
from pyro.infer import SVI, Trace_ELBO
from DataLoader import DataLoader
from platform import python_version
assert pyro.__version__.startswith("1.8")  # I'm writing this tutorial with version

ncoeff = 3
ratio = 1.0


def model_map(x_model, y_model):
    """Encodes observations, latent random variables and parameters in the model
    Args:
        x_model: neural data in form: number of trials x timepoints x channels
        y_model: behavior/continuous learning in form: number of trials x timepoints"""

    sigma_mod = 1
    N, T, p = x_model.shape  # trials, time, and channels/ROIs
    tau_grad, tau0, tau_w = (1 / N), 1, 1  # variance of the beta,w distributions
    sigma_shape, sigma_rate = N * T, sigma_mod * N * T  # prior parameters for the noise distribution

    # learning coefficients
    w = torch.zeros((ncoeff,))
    for i in range(ncoeff):
        w[i] = pyro.sample(f"w_{i}", dist.Normal(0.0, tau_w))

    # learning weights per trial, computed using the learning coefficients
    alpha = torch.zeros((N,))
    ii = torch.linspace(-1, 1, N)

    for j in range(N):
        iij = torch.empty((ncoeff,))
        for i in range(ncoeff):
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


def guide_map(x_guide, y_guide):
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

    for i in range(ncoeff):
        pyro.sample(f"w_{i}", dist.Delta(w_map[i]))


def train(model, guide, lr=0.001, n_steps=10001):
    """For training the model using stochastic variational inference (SVI)
    Args:
        model: encodes observations, latent random variables and parameters in the model
        guide: defines the variational distribution which serves as an approximation to the posterior
        lr: learning rate of model; = 0.001
        n_steps: number of steps/iterations; =  have used between 10001 - 40001"""

    pyro.clear_param_store()
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    for step in range(n_steps):
        loss = svi.step(X_train_torch, y_train_torch)
        if step % 200 == 0:
            print('[iter {}]  loss: {:.4f}'.format(step, loss))


# N, T, p = 100, 139, 64  # will change depending on real data used, will define data size if using synthetic data

dataset_name = 'Ca_imaging'

animals_list = [233]
task_alignment = 'Response'

max_point_list = ['Single', 'Split']
# signals_list = ['non_osc', 'osc']

loop_through_this = product(max_point_list, animals_list)

# max_point = 'Split'
signal_types = 'allsignals'
decoding_method_suffix = 'reg'

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

        sigma = 0.01

        # Data for the time-varying regression - input real data, adjust N, T, p accordingly
        # sub_data_dict = data_loader.load_data_for_subject(sub)
        # X = data_loader.format_data(sub_data_dict, N, T, p)

        sub_data_dict, sub_code = data_loader.load_data_for_subject(subs_list, sub)
        X = data_loader.format_data(sub_data_dict)

        print(sub_code, X.shape)
        N, T, p = X.shape

        est_params_alpha = np.zeros(N)
        est_params_beta_hat = np.zeros((N, p))
        est_params_wmap = np.zeros(ncoeff)

        # Data for the time-varying regression - generate random synthetic data
        # X = np.random.normal(np.zeros((N, T, p)), np.ones((N, T, p)))

        w_true = np.array((0.25, 0.5, 0.025))
        alpha_true = np.zeros((N,))
        ii = np.linspace(-1, 1, N)
        for j in range(N):
            iij = np.array((ii[j], ii[j] ** 2, ii[j] ** 3))
            alpha_true[j] = 1 + np.dot(iij, w_true)
            print(alpha_true[j])

        # alpha_true = alpha_true / 4

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

        fig = plt.figure()
        ax = fig.add_subplot(211)
        pos = ax.imshow(beta_true)
        fig.colorbar(pos)
        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()
        ax.set_aspect(abs((x_right-x_left)/(y_bottom-y_top))*ratio)

        ax = fig.add_subplot(212)
        plt.plot(alpha_true)
        data_loader.save_figure(sub_code, 'true')
        plt.close()

        train(model_map, guide_map)
        pyro.param("sigma_map")

        beta_grad_map = pyro.param("beta_grad_map").detach().numpy()
        beta0_map = pyro.param("beta0_map").detach().numpy()

        beta_hat = np.zeros(beta_true.shape)
        beta_hat[:, 0] = beta0_map
        for j in range(1, N):
            beta_hat[:, j] = beta_hat[:, j - 1] + beta_grad_map

        w_map = pyro.param("w_map")
        alpha = torch.zeros((N,))
        ii = torch.linspace(-1, 1, N)
        for j in range(N):
            iij = torch.tensor((ii[j], ii[j] ** 2, ii[j] ** 3, ii[j] ** 4, ii[j] ** 5))
            alpha[j] = 1 + torch.dot(iij, w_map)
            print(alpha[j])
        # alpha = alpha / 4

        min_alpha_tru = np.min(alpha_true)
        max_alpha_tru = np.max(alpha_true)
        min_alpha_inf = np.min(alpha.detach().numpy())
        max_alpha_inf = np.max(alpha.detach().numpy())
        min_alphas_all = min(min_alpha_tru, min_alpha_inf)
        max_alphas_all = max(max_alpha_tru, max_alpha_inf)

        y_marg = (max_alphas_all - min_alphas_all) * plt.margins()[1]

        min_beta_tru = np.min(beta_true)
        max_beta_tru = np.max(beta_true)
        min_beta_inf = np.min(beta_hat)
        max_beta_inf = np.max(beta_hat)
        min_betas_all = min(min_beta_tru, min_beta_inf)
        max_betas_all = max(max_beta_tru, max_beta_inf)

        fig = plt.figure()
        ax = fig.add_subplot(221)
        pos = ax.imshow(beta_true)
        pos.set_clim(vmin=min_betas_all, vmax=max_betas_all)
        fig.colorbar(pos)

        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_bottom - y_top)) * ratio)

        ax = fig.add_subplot(222)
        pos = ax.imshow(beta_hat)
        pos.set_clim(vmin=min_betas_all, vmax=max_betas_all)
        fig.colorbar(pos)

        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_bottom - y_top)) * ratio)

        ax = fig.add_subplot(223)
        plt.plot(alpha_true)
        plt.ylim(min_alphas_all - y_marg, max_alphas_all + y_marg)

        ax = fig.add_subplot(224)
        plt.plot(alpha.detach().numpy())
        plt.ylim(min_alphas_all - y_marg, max_alphas_all + y_marg)
        data_loader.save_figure(sub_code, 'inferred')
        plt.close()

        beta_corr = np.corrcoef(beta_true.flatten(), beta_hat.flatten())[0, 1]

        sigma_map = pyro.param("sigma_map").detach().numpy()
        epsilon_map = sigma_map * np.random.normal(np.zeros((N, T)), np.ones((N, T)))
        y_map = np.zeros((N, T))

        for j in range(N):
            for t in range(T):
                y_map[j, t] = np.dot(X[j, t, :], beta_hat[:, j]) + epsilon_map[j, t]

        y_corr = np.corrcoef(y.flatten(), y_map.flatten())
        alpha_nump = alpha.detach().numpy()
        alpha_corr = np.corrcoef(alpha_true.flatten(), alpha_nump.flatten())

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
             alpha_true, beta_true, y, w_true, y_map, sigma, X])
