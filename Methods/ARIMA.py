from pmdarima.arima import auto_arima
import pandas as pd
import numpy as np

def lxy_ARIMA(x, train_len):
    train_data = x[:train_len]
    test_data = x[train_len:]
    model = auto_arima(train_data, trace=True, error_action="ignore", suppress_warnings=True)
    model.fit(train_data)
    test_pre = model.predict(n_periods=len(test_data))
    test_pre = np.hstack((train_data, test_pre))
    return test_pre
