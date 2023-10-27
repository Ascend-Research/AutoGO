import torch
import numpy as np
from constants import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr, spearmanr


def get_reg_truth_and_preds(model, loader, fwd_func):

    labels = []
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch_labs = batch[DK_BATCH_TARGET_TSR]
            labels += batch_labs.detach().tolist()
            batch_preds = fwd_func(model, batch)
            preds += batch_preds.detach().tolist()

    labels = np.array(labels).squeeze()
    preds = np.array(preds).squeeze()

    return labels, preds

def get_seg_truth_and_preds(model, loader, fwd_func):

    labels = []
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch_labs = batch[DK_BATCH_TARGET_TSR]
            labels += batch_labs.detach().tolist()
            batch_preds = fwd_func(model, batch)
            preds += batch_preds.detach().tolist()

    labels = np.array(labels).squeeze()
    preds = np.array(preds).squeeze()

    return labels, preds


# Mean Squared Error - MSE
# Mean Absolute Error - MAE
# Mean Absolute Percentage Error - MAPE
def pure_regressor_metrics(targets, preds):

    # MSE, MAE, MAPE
    target_mse = mean_squared_error(targets, preds)
    target_mae = mean_absolute_error(targets, preds)
    target_mape = mean_absolute_percentage_error(targets, preds)

    return [target_mse, target_mae, target_mape]


# Spearmann Rank Correlation - SRCC
# Pearson Correlation
def correlation_metrics(targets, preds, pearson=False):

    metrics = []
    if pearson:
        # Pearson Correlation
        pcc, pp = pearsonr(targets, preds)
        metrics.append(pcc)

    # Spearman Correlation
    srcc, sp = spearmanr(targets, preds)
    metrics.append(srcc)

    return metrics
