from sklearn.metrics import r2_score
import numpy as np


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """For calculating an accuracy score for each time point
    Args:
        y_true: numpy array dimensions (num trials, num timepoints)
        y_pred: numpy array dimensions (num trials, num timepoints)
    Returns:
        r2_scores: numpy array dimensions (num timepoints)"""
    num_trials, num_timepoints = y_true.shape
    r2_scores: np.ndarray = np.zeros(num_timepoints)
    # iterating over every label and checking it with the true sample
    for timepoint, (actual, predicted) in enumerate(zip(y_true.T, y_pred.T)):
        r2_scores[timepoint] = r2_score(actual, predicted)
    return r2_scores
