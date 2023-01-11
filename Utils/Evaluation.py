import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

def MAPE(y_pred, y_true):
    return np.mean(np.abs((y_pred - y_true) / y_true))

def MAE_per(y_pred, y_true):
    return np.mean(np.abs((y_pred - y_true))) / (np.max(y_true) - np.min(y_true))

def eval(x, y, x_mean, x_std, y_original):
    rmse = np.sqrt(mean_squared_error(x, y))
    mae = mean_absolute_error(x, y)
    r2 = r2_score(x*x_std + x_mean, y_original)
    mape = mean_absolute_percentage_error(x*x_std + x_mean, y_original)
    mae_per = MAE_per(x, y)
    return rmse, mae, r2, mape, mae_per
    
