import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from scipy.io import loadmat
from scipy.io import savemat


class DataFormatSaver:

    def __init__(self, n, t, p, run) -> None:
        """For initialising the DataFormatSaver object
        Args:
            N: number of trials in dataset
            T: number of timepoints per trial in dataset
            P: number of channels/electrodes in dataset
            run: repetition number of simulation"""

        self.doc_path = r"C:\Users\au699132\OneDrive - Aarhus Universitet\Documents"
        new_folder_name = 'N_{N}_T_{T}_P_{P}'.format(N=n, T=t, P=p)
        self.figure_dir = os.path.join(
            self.doc_path, 'Python', 'Modelling_simulations_figs', 'Repetitions', new_folder_name)
        self.model_data_dir = os.path.join(
            self.doc_path, 'Python', 'Modelling_simulations_params', 'Repetitions', new_folder_name)

        fig_dir_exists = os.path.exists(self.figure_dir)
        if not fig_dir_exists:
            os.mkdir(self.figure_dir)

        dat_dir_exists = os.path.exists(self.model_data_dir)
        if not dat_dir_exists:
            os.mkdir(self.model_data_dir)

        self.number_trials = n
        self.timepoints = t
        self.channels = p
        self.run_num = run

    def save_figure(self, w_true: np.ndarray, sigma: float, figure_suffix: str) -> None:
        """For saving figures showing true and/or inferred parameter values
        Args:
            w_true: true w parameter values
            sigma: true value of sigma
            figure_suffix: true or true and inferred parameter values"""

        w_params_shape = len(w_true)
        w_val_string = '_'.join([str(wv) for wv in w_true]).replace('.', '-')
        fig_name = 'Sim_{run}_{w_shape}w_params_{w_str}_sigma_{sig}_N{N}_T{T}_P{P}_{fig_suffix}.png'.format(
            run=self.run_num, w_shape=w_params_shape, w_str=w_val_string, sig=sigma, N=self.number_trials,
            T=self.timepoints, P=self.channels, fig_suffix=figure_suffix)
        fig_save_path = os.path.join(self.figure_dir, fig_name)
        plt.savefig(fig_save_path)

    def save_model_pkl(self, w_true: np.ndarray, sigma: float, objects: List) -> None:
        """For saving inferred parameter values per subject in .pkl file
        Args:
            w_true: true w parameter values
            sigma: true value of sigma
            objects: List of inferred parameter values to be saved in file"""
        w_params_shape = len(w_true)
        w_val_string = '_'.join([str(wv) for wv in w_true]).replace('.', '-')
        pkl_save_name = 'Sim_{run}_{w_shape}w_params_{w_str}_sigma_{sig}_N{N}_T{T}_P{P}.pkl'.format(
            run=self.run_num, w_shape=w_params_shape, w_str=w_val_string, sig=sigma, N=self.number_trials,
            T=self.timepoints, P=self.channels)
        pkl_save_path = os.path.join(self.model_data_dir, pkl_save_name)
        with open(pkl_save_path, 'wb') as file_h:
            pickle.dump(objects, file_h)

    def load_model_pkl(self, w_true: np.ndarray, sigma: float, midpoint: Optional[float] = None) -> List:
        """For saving inferred parameter values per subject in .pkl file
        Args:
            w_true: true w parameter values
            sigma: true value of sigma
            midpoint: true value of midpoint in logistic function"""

        w_params_shape = len(w_true)
        w_val_string = '_'.join([str(wv) for wv in w_true]).replace('.', '-')
        pkl_save_name = 'Sim_{run}_{w_shape}w_params_{w_str}_sigma_{sig}_N{N}_T{T}_P{P}.pkl'.format(
            run=self.run_num, w_shape=w_params_shape, w_str=w_val_string, sig=sigma, N=self.number_trials,
            T=self.timepoints, P=self.channels)
        pkl_save_path = os.path.join(self.model_data_dir, pkl_save_name)
        with open(pkl_save_path, 'rb') as file_h:
            model_data = pickle.load(file_h)
        return model_data

    def save_all_subs_figure(self, plotted_params: str, figure_suffix: str) -> None:
        """For saving figures showing true and/or inferred parameter values
        Args:
            plotted_params: which parameters are plotted in the figure
            figure_suffix: true or true and inferred parameter values"""

        fig_name = 'ALL_Sims_run_{run}_{params}_{fig_suffix}_N{N}_T{T}_P{P}.png'.format(
            run=self.run_num, params=plotted_params, fig_suffix=figure_suffix, N=self.number_trials, T=self.timepoints,
            P=self.channels)
        fig_save_path = os.path.join(self.figure_dir, fig_name)
        plt.savefig(fig_save_path)

    def save_model_mat(self, objects: Dict, file_suffix: str) -> None:
        """For saving inferred parameter values per subject in .mat file
        Args:
            objects: List of inferred parameter values to be saved in file
            file_suffix: string containing which parameters are saved here"""

        mat_save_name = 'ALL_Subs_run_{run}_params_{file_suf}_N{N}_T{T}_P{P}.mat'.format(
            run=self.run_num, file_suf=file_suffix, N=self.number_trials, T=self.timepoints, P=self.channels)
        mat_save_path = os.path.join(self.model_data_dir, mat_save_name)
        savemat(mat_save_path, objects)
