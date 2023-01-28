# Class: DataLoader
# For loading neuroimaging data files, formatting data for modelling with PyTorch, and saving output of modelling
# as .pkl files containing inferred parameter values, and figures as .png
# Contributors: Ashley Tyrer
# Centre of Functionally Integrative Neuroscience, Aarhus University
# Created 16-05-2022
# Edited by Ashley Tyrer 17-05-2022 for Nadine's MEG data

import os
import glob
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from scipy.io import loadmat
from scipy.io import savemat


class DataLoader:

    def __init__(self, task_type: str, variable_suffix: str, max_point: str, dataset: str,
                 animal_number: Optional[int] = None, alignment: Optional[str] = None) -> None:
        """For initialising the DataLoader object
        Args:
            task_type: either perception or imagery, single time-point or whole trial
            variable_suffix: string containing info about signal type, decoding method and trial split
            max_point: states if the time-point of maximum decoding accuracy was taken from whole trial or from 1st and
                2nd half of trials separately
            animal_number: optional arg, int containing animal number
            alignment: optional arg, string stating the event to which the Ca imaging data was aligned"""

        if variable_suffix[-3:] == 'reg':
            decoding_method = 'regression'
        else:
            decoding_method = variable_suffix[-3:]

        self.output_folder = '{max_pt}_Max_Point'.format(max_pt=max_point)
        self.doc_path = r"C:\Users\au699132\OneDrive - Aarhus Universitet\Documents"
        self.decoding_file = 'Standard_{dec_meth}'.format(dec_meth=decoding_method)
        self.task_type = task_type
        self.main_results_path = os.path.join(self.doc_path, 'Python')
        self.set_dirs: List = []
        self.set_dirs = self.set_data_dirs(dataset, animal_number, alignment)
        self.var_suffix = variable_suffix

    def set_data_dirs(self, dataset: str, animal_number: Optional[int] = None, alignment: Optional[str] = None) -> List:
        """For setting the data and results directories based on the dataset used
        Args:
            dataset: string stating which dataset is being analysed
            animal_number: optional arg, int containing animal number
            alignment: optional arg, string stating the event to which the Ca imaging data was aligned
        Returns:
            data_dir: string containing dir from which to load neural data
            figure_dir: string containing dir for saving modelling figures
            model_data_dir: string containing dir for saving modelling results"""

        if dataset == 'Nadine':
            data_dir = os.path.join(
                self.doc_path, 'MATLAB', 'for_Ashley', 'Nadine_rawdata', 'Recon_SourceSpace_Data', 'NEW',
                'Single_timepoint_data', self.decoding_file)
            figure_dir = os.path.join(self.main_results_path, 'Nadine_modelling_figs', self.output_folder)
            model_data_dir = os.path.join(self.main_results_path, 'Nadine_modelling_params',
                                          self.output_folder)
        elif dataset == 'Ca_imaging':
            data_dir = os.path.join(
                self.doc_path, 'MATLAB', 'Ca_imaging_analysis', 'animal_{an_num}', '{align}_aligned', 'All_sessions',
                'Single_timepoint_data', self.decoding_file).format(an_num=animal_number, align=alignment)
            figure_dir = os.path.join(self.main_results_path, 'Ca_imaging_modelling_figs', 'animal_{an_num}',
                                      '{align}_aligned', self.output_folder).format(
                an_num=animal_number, align=alignment)
            model_data_dir = os.path.join(self.main_results_path, 'Ca_imaging_modelling_params', 'animal_{an_num}',
                                          '{align}_aligned', self.output_folder).format(
                an_num=animal_number, align=alignment)
        else:
            raise ValueError('Existing dataset not entered')

        return [data_dir, figure_dir, model_data_dir]

    def get_data_files(self) -> List:
        """For the number of files that match the subject type and frequency band given
        Returns:
            the number of files"""

        file_paths = glob.glob(os.path.join(self.set_dirs[0], 'S*_{}*'.format(self.task_type)))
        return file_paths

    def load_data_for_subject(self, file_paths: List, subject_id: int) -> tuple[Dict, str]:
        """For loading the data for a given subject.
        Args:
            file_paths: list of file paths for data
            subject_id: between 1 and 25 inclusive
        Returns:
            subject_data_dict: dictionary with 'X' and 'T' keys
            subject code: string of subject ID"""

        matrix_name = 'X_single_timepoint_{sig_type}'.format(sig_type=self.var_suffix)
        subject_data_dict = loadmat(file_paths[subject_id], variable_names=matrix_name)
        subject_code = file_paths[subject_id][-28:-26]
        return subject_data_dict, subject_code

    def format_data(self, subject_data_dict: Dict) -> np.ndarray:
        """For formatting data ready for pytorch modelling
        Args:
            subject_data_dict: dictionary with 'X' key
        Returns:
            3D numpy array containing data in form: number of trials x timepoints x channels"""

        x_neuro_array = subject_data_dict['X_single_timepoint_{sig_type}'.format(sig_type=self.var_suffix)]
        single_tp, n_trials, channels = x_neuro_array.shape

        ts_trial = single_tp
        X = x_neuro_array.reshape(n_trials, ts_trial, channels)
        return X

    def save_figure(self, subject_code: str, figure_suffix: str) -> None:
        """For saving figures showing true and/or inferred parameter values
        Args:
            subject_code: string containing subject ID
            figure_suffix: true or true and inferred parameter values"""

        fig_name = 'S{sub_code}_{task_type}_alpha_beta_hat_{fig_suffix}_{sig_type}.png'.format(
            sub_code=subject_code, task_type=self.task_type, fig_suffix=figure_suffix, sig_type=self.var_suffix)
        fig_save_path = os.path.join(self.set_dirs[1], fig_name)
        plt.savefig(fig_save_path)

    def save_model_pkl(self, subject_code: str, objects: List) -> None:
        """For saving inferred parameter values per subject in .pkl file
        Args:
            subject_code: string containing subject ID
            objects: List of inferred parameter values to be saved in file"""

        pkl_save_name = 'S{sub_code}_{task_type}_params_{sig_type}.pkl'.format(
            sub_code=subject_code, task_type=self.task_type, sig_type=self.var_suffix)
        pkl_save_path = os.path.join(self.set_dirs[2], pkl_save_name)
        with open(pkl_save_path, 'wb') as file_h:
            pickle.dump(objects, file_h)

    def load_model_pkl(self, subject_code: str) -> List:
        """For saving inferred parameter values per subject in .pkl file
        Args:
            subject_code: string containing subject ID"""

        pkl_save_name = 'S{sub_code}_{task_type}_params_{sig_type}.pkl'.format(
            sub_code=subject_code, task_type=self.task_type, sig_type=self.var_suffix)
        pkl_save_path = os.path.join(self.set_dirs[2], pkl_save_name)
        with open(pkl_save_path, 'rb') as file_h:
            model_data = pickle.load(file_h)
        return model_data

    def save_all_subs_figure(self, plotted_params: str, figure_suffix: str) -> None:
        """For saving figures showing true and/or inferred parameter values
        Args:
            plotted_params: which parameters are plotted in the figure
            figure_suffix: true or true and inferred parameter values"""

        fig_name = 'ALL_Subs_{task_type}_{params}_{fig_suffix}_{sig_type}.png'.format(
            task_type=self.task_type, params=plotted_params, fig_suffix=figure_suffix, sig_type=self.var_suffix)
        fig_save_path = os.path.join(self.set_dirs[1], fig_name)
        plt.savefig(fig_save_path)

    def save_model_mat(self, objects: Dict, file_suffix: str) -> None:
        """For saving inferred parameter values per subject in .mat file
        Args:
            objects: List of inferred parameter values to be saved in file
            file_suffix: string containing which parameters are saved here"""

        mat_save_name = 'ALL_Subs_{task_type}_params_{file_suf}_{sig_type}.mat'.format(
            task_type=self.task_type, file_suf=file_suffix, sig_type=self.var_suffix)
        mat_save_path = os.path.join(self.set_dirs[2], mat_save_name)
        savemat(mat_save_path, objects)
