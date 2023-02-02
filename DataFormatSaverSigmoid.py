# Class: DataFormatSaverSigmoid
# For setting directories and formatting/saving data and figures, for model with sigmoid alpha
# Contributors: Ashley Tyrer
# Centre of Functionally Integrative Neuroscience, Aarhus University
# Created 12-09-2022
# Edited by Ashley Tyrer, date of last edit: 02-02-2023

import os
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from typing import List
matplotlib.use('Agg')


class DataFormatSaverSigmoid:

    def __init__(self, n, t, p, run) -> None:
        """For initialising the DataFormat_Saver_sigmoid object
        Args:
            n: number of trials in dataset
            t: number of timepoints per trial in dataset
            p: number of channels/electrodes in dataset
            run: repetition number of simulation"""

        self.doc_path = r"C:\Users\au699132\OneDrive - Aarhus Universitet\Documents"
        new_folder_name = 'N_{N}_T_{T}_P_{P}'.format(N=n, T=t, P=p)
        self.figure_dir = os.path.join(
            self.doc_path, 'Python', 'Modelling_simulations_figs', 'Sigmoid_alpha', new_folder_name)
        self.model_data_dir = os.path.join(
            self.doc_path, 'Python', 'Modelling_simulations_params', 'Sigmoid_alpha', new_folder_name)

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

    def save_figure(self, w_true: np.ndarray, sigma: float, midpoint: np.ndarray, figure_suffix: str) -> None:
        """For saving figures showing true and/or inferred parameter values
        Args:
            w_true: true w parameter values
            sigma: true value of sigma
            midpoint: true midpoint of sigmoid curve
            figure_suffix: true or true and inferred parameter values"""

        # w_val_string = '_'.join([str(wv) for wv in w_true]).replace('.', '-')
        fig_name = 'Sim_{run}_w_params_{w_str}_sigma_{sig}_mid_{m}_alpha_beta_hat_{fig_suffix}.png'.format(
            run=self.run_num, w_str=w_true, sig=sigma, m=midpoint, fig_suffix=figure_suffix)
        fig_save_path = os.path.join(self.figure_dir, fig_name)
        plt.savefig(fig_save_path)

    def save_model_pkl(self, w_true, sigma: float, midpoint, objects: List) -> None:
        """For saving inferred parameter values per subject in .pkl file
        Args:
            w_true: true w parameter values
            sigma: true value of sigma
            midpoint: true midpoint of sigmoid curve
            objects: List of inferred parameter values to be saved in file"""

        pkl_save_name = 'Sim_{run}_w_params_{w_str}_sigma_{sig}_mid_{m}.pkl'.format(
            run=self.run_num, w_str=w_true, sig=sigma, m=midpoint)
        pkl_save_path = os.path.join(self.model_data_dir, pkl_save_name)
        with open(pkl_save_path, 'wb') as file_h:
            pickle.dump(objects, file_h)

    def load_model_pkl(self, w_true, sigma: float, midpoint) -> List:
        """For saving inferred parameter values per subject in .pkl file
        Args:
            w_true: true w parameter values
            sigma: true value of sigma
            midpoint: true midpoint of sigmoid curve"""

        pkl_save_name = 'Sim_{run}_w_params_{w_str}_sigma_{sig}_mid_{m}.pkl'.format(
            run=self.run_num, w_str=w_true, sig=sigma, m=midpoint)
        pkl_save_path = os.path.join(self.model_data_dir, pkl_save_name)
        with open(pkl_save_path, 'rb') as file_h:
            model_data = pickle.load(file_h)
        return model_data

    def save_all_subs_figure(self, plotted_params: str, figure_suffix: str) -> None:
        """For saving figures showing true and/or inferred parameter values
        Args:
            plotted_params: which parameters are plotted in the figure
            figure_suffix: true or true and inferred parameter values"""

        fig_name = 'ALL_Sims_run{run}_{params}_{fig_suffix}.png'.format(
            run=self.run_num, params=plotted_params, fig_suffix=figure_suffix)
        fig_save_path = os.path.join(self.figure_dir, fig_name)
        plt.savefig(fig_save_path)
