import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


class DataPlotterSaver:

    def __init__(self, num_subs, which_model) -> None:
        """For initialising the DataPlotterSaver object and defining attributes
        Args:
            num_subs: number of subjects/parameter combinations
            which_model: which version of the HBM the data was generated from"""

        self.alpha_trues = [np.nan] * num_subs
        self.y_trues = [np.nan] * num_subs
        self.w_trues = [np.nan] * num_subs
        self.sigma_trues = [np.nan] * num_subs
        self.beta_trues = [np.nan] * num_subs
        self.midpoint_trues = [np.nan] * num_subs
        self.alpha_infs = [np.nan] * num_subs
        self.y_infs = [np.nan] * num_subs
        self.w_infs = [np.nan] * num_subs
        self.sigma_infs = [np.nan] * num_subs
        self.beta_infs = [np.nan] * num_subs
        self.midpoint_infs = [np.nan] * num_subs
        self.y_correlations = np.zeros(num_subs)
        self.beta_correlations = np.zeros(num_subs)
        self.alpha_correlations = np.zeros(num_subs)
        self.w_correlations = np.zeros(num_subs)
        self.beta_infs_1st_half = [np.nan] * num_subs
        self.beta_infs_2nd_half = [np.nan] * num_subs
        self.X_data = [np.nan] * num_subs

        self.which_model = which_model
        self.num_subs = num_subs

    def define_data_objects_indices(self) -> Dict:
        """For extracting the correct indices of each parameter from the data_objects list
        Returns:
            data_dict: dictionary of indices for data_objects list"""

        data_dict = {'sigma_inf_ind': 0, 'alpha_inf_ind': 1, 'beta_inf_ind': 2, 'w_inf_ind': 3, 'beta_corr_ind': 4,
                     'y_corr_ind': 5, 'alpha_corr_ind': 6, 'm_inf_ind': [], 'w_tru_ind': [], 'sigma_tru_ind': [],
                     'y_tru_ind': [], 'y_inf_ind': [], 'beta_tru_ind': [], 'alpha_tru_ind': np.nan, 'm_tru_ind': [],
                     'X_ind': []}

        if self.which_model == 1:
            data_dict['w_tru_ind'] = 7
            data_dict['sigma_tru_ind'] = 8
            data_dict['y_tru_ind'] = 9
            data_dict['y_inf_ind'] = 10
            data_dict['beta_tru_ind'] = 11
            data_dict['alpha_tru_ind'] = 12
            data_dict['X_ind'] = 13
        elif self.which_model == 2:
            data_dict['m_inf_ind'] = 7
            data_dict['w_tru_ind'] = 8
            data_dict['sigma_tru_ind'] = 9
            data_dict['m_tru_ind'] = 10
            data_dict['y_tru_ind'] = 11
            data_dict['y_inf_ind'] = 12
            data_dict['beta_tru_ind'] = 13
            data_dict['alpha_tru_ind'] = 14
            data_dict['X_ind'] = 15
        else:
            raise ValueError('Valid model number not entered')

        return data_dict

    def data_formatting(self, data_objects, subject_index) -> List:
        """For extracting the data structures of parameter values from data_objects list
        Args:
            data_objects: structure containing model data
            subject_index: index of subject or parameter set
        Returns:
            list of lists, containing all true and inferred parameter values"""

        data_obj_dict = self.define_data_objects_indices()
        self.alpha_trues[subject_index] = data_objects[data_obj_dict['alpha_tru_ind']]
        self.sigma_trues[subject_index] = data_objects[data_obj_dict['sigma_tru_ind']]
        self.w_trues[subject_index] = data_objects[data_obj_dict['w_tru_ind']]
        self.y_trues[subject_index] = data_objects[data_obj_dict['y_tru_ind']]
        self.beta_trues[subject_index] = data_objects[data_obj_dict['beta_tru_ind']]
        self.alpha_infs[subject_index] = data_objects[data_obj_dict['alpha_inf_ind']]
        self.beta_infs[subject_index] = data_objects[data_obj_dict['beta_inf_ind']].transpose(1, 0)
        self.y_infs[subject_index] = data_objects[data_obj_dict['y_inf_ind']]
        self.w_infs[subject_index] = data_objects[data_obj_dict['w_inf_ind']]
        self.sigma_infs[subject_index] = data_objects[data_obj_dict['sigma_inf_ind']]
        self.y_correlations[subject_index] = data_objects[data_obj_dict['y_corr_ind']][0, 1]
        self.beta_correlations[subject_index] = data_objects[data_obj_dict['beta_corr_ind']]
        self.alpha_correlations[subject_index] = data_objects[data_obj_dict['alpha_corr_ind']][0, 1]
        self.alpha_correlations[np.isnan(self.alpha_correlations)] = 0
        self.X_data = data_objects[data_obj_dict['X_ind']]

        p, N = self.beta_infs[subject_index].shape
        T = self.y_trues[subject_index].shape[1]

        self.split_betas_into_halves(subject_index, p, N)

        if self.which_model == 1:
            self.w_correlations[subject_index] = np.corrcoef(self.w_trues[subject_index].flatten(), self.w_infs[
                subject_index].flatten())[0, 1]
            self.w_correlations[np.isnan(self.w_correlations)] = 0
            return [self.sigma_infs, self.alpha_infs, self.beta_infs, self.w_infs, self.beta_correlations,
                    self.y_correlations, self.alpha_correlations, self.w_trues, self.sigma_trues, self.y_trues,
                    self.y_infs, self.beta_trues, self.alpha_trues, self.w_correlations, p, N, T,
                    self.beta_infs_1st_half, self.beta_infs_2nd_half]
        elif self.which_model == 2:
            self.midpoint_trues[subject_index] = data_obj_dict['m_tru_ind']
            self.midpoint_infs[subject_index] = data_obj_dict['m_inf_ind']
            return [self.sigma_infs, self.alpha_infs, self.beta_infs, self.w_infs, self.beta_correlations,
                    self.y_correlations, self.alpha_correlations, self.midpoint_infs, self.w_trues, self.sigma_trues,
                    self.midpoint_trues, self.y_trues, self.y_infs, self.beta_trues, self.alpha_trues, p, N, T,
                    self.beta_infs_1st_half, self.beta_infs_2nd_half]
        else:
            raise ValueError('Valid model number not entered')

    def split_betas_into_halves(self, subject_index, p, N):
        """For splitting the inferred betas into those from the first and second half of trials
        Args:
            subject_index: index of subject or parameter set
            p: number of channels
            N: number of trials"""

        betas_1st_half = np.zeros(shape=(round(N / 2), p))
        betas_2nd_half = np.zeros(shape=(round(N / 2) - 1, p))

        beta_inf_sub = self.beta_infs[subject_index]

        for chan in range(p):
            for n in range(round(N / 2)):
                betas_1st_half[n, chan] = beta_inf_sub[chan, n]
            for n in range((round(N / 2)) - 1):
                betas_2nd_half[n, chan] = beta_inf_sub[chan, round(N / 2) + n]

        self.beta_infs_1st_half[subject_index] = betas_1st_half
        self.beta_infs_2nd_half[subject_index] = betas_2nd_half

        # return self.beta_infs_1st_half, self.beta_infs_2nd_half

    def axis_label_setting(self, w_true, sigma, midpoint_tru: Optional[float] = None) -> Tuple[str, str]:
        """For setting the tick labels for plot axes
        Args:
            w_true: true value of w
            sigma: true value of sigma
            midpoint_tru: true value of midpoint
        Returns:
            param_set_single_line: string to be used as axis tick labels on a single line
            param_set_multi_lines: string to be used as axis tick labels on multiple lines"""

        if self.which_model == 1:
            w_val_string = '_'.join([str(wv) for wv in w_true]).replace('.', '-')
            param_set_single_line = 'w = {w}; sigma = {sig}'.format(w=w_val_string, sig=sigma)
            param_set_multi_lines = 'w = {w}\nsigma = {sig}'.format(w=w_val_string, sig=sigma)
        elif self.which_model == 2:
            param_set_single_line = 'w = {w}; sigma = {sig}; m = {m}'.format(w=w_true, sig=sigma, m=midpoint_tru)
            param_set_multi_lines = 'w = {w}\nsigma = {sig}\nm = {m}'.format(w=w_true, sig=sigma, m=midpoint_tru)
        else:
            raise ValueError('Valid model number not entered')

        return param_set_single_line, param_set_multi_lines

    def calculate_plot_min_max(self, param_struct):
        """For calculating the minimum and maximum values of parameter sets for defining axes limits
        Args:
            param_struct: data structure containing parameter values to be plotted
        Returns:
            param_plot_final_min: minimum value for uniform plot axes across subplots
            param_plot_final_max: maximum value for uniform plot axes across subplots
            y_marg_param: margins between extreme data points and y axis limits"""

        param_plot_min = np.zeros(self.num_subs)
        param_plot_max = np.zeros(self.num_subs)

        for n in range(self.num_subs):
            param_struct_single = param_struct[n]
            param_plot_min[n] = np.min(param_struct_single)
            param_plot_max[n] = np.max(param_struct_single)

        param_plot_final_min = np.min(param_plot_min)
        param_plot_final_max = np.max(param_plot_max)
        y_marg_param = (param_plot_final_max - param_plot_final_min) * plt.margins()[1]

        return param_plot_final_min, param_plot_final_max, y_marg_param

    def multi_run_data_format(self) -> Dict:
        """For formatting the data structures of parameter values from data_objects list across all subjects
            Returns:
                big_data_dict: dictionary of parameter values across all subjects"""

        data_obj_dict = self.define_data_objects_indices()
        data_obj_dict['w_corr_ind'] = []
        del data_obj_dict['X_ind']
        if self.which_model == 1:
            del data_obj_dict['m_tru_ind'], data_obj_dict['m_inf_ind']

        all_params_list = [self.sigma_infs, self.alpha_infs, self.beta_infs, self.w_infs, self.beta_correlations,
                           self.y_correlations, self.alpha_correlations, self.w_trues, self.sigma_trues, self.y_trues,
                           self.y_infs, self.beta_trues, self.alpha_trues, self.w_correlations]
        big_data_dict = data_obj_dict.copy()
        empty_placeholder = [np.nan] * self.num_subs

        for keys in big_data_dict:
            big_data_dict[keys] = empty_placeholder

        for count, keys in enumerate(big_data_dict):
            big_data_dict[keys] = all_params_list[count]

        return big_data_dict

    def alpha_line_plotting(self, fig, alpha_plot, rows, columns, splot_index, alphas_concat: Optional = None):
        """For creating a line plot to display alpha parameter values over trials
        Args:
            fig: main figure to contain subplots
            alpha_plot: alpha data structure to be plotted
            rows: number of rows of subplots in figure
            columns: number of columns of subplots in figure
            splot_index: index of subplot to be created
            alphas_concat: all alpha parameter values concatenated in a 1-D vector"""

        string_indices = str(rows) + str(columns) + str(splot_index)
        int_indices = int(string_indices)
        ax = fig.add_subplot(int_indices)
        plt.plot(alpha_plot)

        if alphas_concat:
            self.common_axes_limits(alphas_concat)

    def betas_heatmap_plotting(self, fig, beta_plot, rows, columns, splot_index, betas_concat: Optional = None):
        """For creating a heatmap to display beta parameter values over trials and channels
        Args:
            fig: main figure to contain subplots
            beta_plot: alpha data structure to be plotted
            rows: number of rows of subplots in figure
            columns: number of columns of subplots in figure
            splot_index: index of subplot to be created
            betas_concat: all beta parameter values concatenated in a 2-D matrix"""

        ratio = 1.0
        string_indices = str(rows) + str(columns) + str(splot_index)
        int_indices = int(string_indices)
        ax = fig.add_subplot(int_indices)
        pos = ax.imshow(beta_plot)
        if betas_concat:
            min_betas_all, max_betas_all, y_marg_beta = self.calculate_plot_min_max(betas_concat)
            pos.set_clim(vmin=min_betas_all, vmax=max_betas_all)
        fig.colorbar(pos)

        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_bottom - y_top)) * ratio)

    def common_axes_limits(self, data_concat):
        """For setting the axes limits to be constant across all subplots in a large figure
        Args:
            data_concat: data for multiple subplots concatenated"""

        min_params_all, max_params_all, y_marg = self.calculate_plot_min_max(data_concat)
        plt.ylim(min_params_all - y_marg, max_params_all + y_marg)
