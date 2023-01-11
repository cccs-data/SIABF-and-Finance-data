from sklearn.linear_model import Lasso, ARDRegression, LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.fft import fft
def interrupted_function_1(x):
    return ((1-np.floor(np.pi*x)%2)-0.5)*2

def interrupted_function_2(x):
    return ((1-np.floor(np.pi*(x-0.5))%2)-0.5)*2


def lxy_basis_noncycle(x):
    # 在未知参数的情况下生成的基底
    my_basis = []
    '''
    for i in range(3, 5):
        my_basis.append(x**(i/2))
    for i in range(2, 17):
        my_basis.append(x**(1/(i-1)))
    '''
    for i in range(1,5):
        my_basis.append(np.sin(2*np.pi*x*i))
        my_basis.append(np.cos(2*np.pi*x*i))
    # my_basis.append(np.exp(0.1*x))
    # my_basis.append(np.log(np.abs(x) + 1))
    return np.array(my_basis)


def lxy_basis_cycle(t, x, max_k=100, norm=False):
    # 通过傅里叶变换，在已知先验的情况下进行辨识
    if not norm:
        x = (x - np.mean(x))/np.std(x)
    # print(x)
    my_basis = []
    start_t = min(t)
    end_t = max(t)
    n = len(t)
    sample_freq = (end_t-start_t)/n 
    fft_x = np.fft.rfft(x)
    fft_x_abs = np.abs(fft_x)
    freq = np.fft.fftfreq(len(x), sample_freq)
    need_len = len(x) // 2
    T_max_k = 1/freq[fft_x_abs.argsort()[::-1][:max_k]]
    freq_max_k = fft_x_abs[fft_x_abs.argsort()[::-1][:max_k]]
    # print(T_max_k)
    for i in range(max_k):
        my_basis.append(np.sin(2*np.pi*t/T_max_k[i]))
        my_basis.append(np.cos(2*np.pi*t/T_max_k[i]))
        # my_basis.append(interrupted_function_1(2*np.pi*t/T_max_k[i]))
        # my_basis.append(interrupted_function_2(2*np.pi*t/T_max_k[i]))

    '''
    for i in range(2, 17):
        my_basis.append(x**(1/(i-1)))
    '''
    return np.array(my_basis)
    


def lxy_lasso(x, y, train_rate=0.7, def_alpha=1e-2, it = 10000, norm=False):
    train_len = int(train_rate*len(y))
    y_mean = np.mean(y[:train_len])
    y_std = np.std(y[:train_len])
    if not norm:
        y = (y - y_mean)/y_std
    # train_len = len(y)
    
    x_train = x[:, :train_len]
    y_train = y[:train_len]
    lasso = Lasso(alpha=def_alpha, max_iter=it)
    lasso.fit(np.transpose(x_train), y_train.reshape(-1, 1))
    coef_lasso = np.hstack((lasso.coef_, lasso.intercept_))
    x_ones = np.ones((1, len(y)))
    # print(x.shape, x_ones.shape)
    x = np.vstack((x, x_ones))
    # print(coef_lasso.shape, x.shape)
    y_pre = coef_lasso @ x
    y_pre = y_pre * y_std + y_mean
    # return coef_lasso, y_pre
    return y_pre


def lxy_SBL(x, y, train_rate=0.7, def_alpha_1=1e-6, def_alpha_2=1e-6, def_lambda_1=1e-6, 
            def_lambda_2=1e-6, thro=1000, tole=0.001, it = 500, norm=False):
    train_len = int(train_rate*len(y))
    
    y_mean = np.mean(y[:train_len])
    y_std = np.std(y[:train_len])
    if not norm:
        y = (y - y_mean)/y_std
    # train_len = len(y)
    
    x_train = x[:, :train_len]
    y_train = y[:train_len]
    # alpha=def_alpha, max_iter=it
    SBL = ARDRegression(alpha_1=def_alpha_1, alpha_2=def_alpha_2, compute_score=False,
        copy_X=True, fit_intercept=True, lambda_1=def_lambda_1, lambda_2=def_lambda_2,
        n_iter=it, threshold_lambda=thro, tol=tole, verbose=False)
    SBL.fit(np.transpose(x_train), y_train)
    coef_sbl = np.hstack((SBL.coef_, SBL.intercept_))
    x_ones = np.ones((1, len(y)))
    # print(x.shape, x_ones.shape)
    x = np.vstack((x, x_ones))
    # print(coef_sbl.shape, x.shape)
    y_pre = coef_sbl @ x
    y_pre = y_pre * y_std + y_mean
    # return coef_lasso, y_pre
    return y_pre


def my_SINDy(X, Y, train_rate=0.7, lammbda=1e-5, norm=False):  # This is STLS regression
    train_len = int(train_rate*len(Y))
    
    y_mean = np.mean(Y[:train_len])
    y_std = np.std(Y[:train_len])
    
    X = np.transpose(X)
    X_train = X[:train_len, :]
    Y_train = Y[:train_len]
    
    if not norm:
        Y = (Y - y_mean)/y_std
    model = LinearRegression()
    final_model = LinearRegression()
    model = model.fit(X_train, Y_train)
    inds = model.coef_
    times = 10
    for i in range(1, times + 1):
        smallinds = abs(inds) < lammbda
        inds[smallinds] = 0
        biginds = ~smallinds
        new_model = LinearRegression()
        new_model.fit(X_train[:, biginds.ravel()], Y_train)
        inds[biginds] = new_model.coef_.ravel()  # convert 2D vector to 1D vector
        if i == times:
            final_model = new_model
    inter = final_model.intercept_
    inds_inter = np.hstack((inds, inter))  # Convert two vector
    
    X_ones = np.ones((len(Y), 1))
    X = np.hstack((X, X_ones))
    Y_pre = inds_inter @ np.transpose(X)
    Y_pre = Y_pre * y_std + y_mean
    
    return Y_pre


'''
# STLS展示:
x = np.linspace(1, 10, 100)
y = np.sin(x) + np.cos(5*x) + 0.01*np.random.randn(x.shape[0])
y_pre = my_SINDy(lxy_basis_cycle(x, y, max_k=10), y, lammbda=0.75)
plt.plot(x, y_pre, label='predict')
plt.plot(x, y, label='ground truth')
plt.legend()
plt.savefig('try.png')
'''




'''
t_len = 10000
train_rate = 0.8
train_len = int(t_len*train_rate)
x = np.linspace(0,10,t_len)
y = 3*x + 5*np.sin(np.pi*2*x*3) + 0.5*np.random.randn(t_len)
# print(lxy_basis_noncycle(x).shape)
# x_try = lxy_basis_noncycle(x)
m,n = lxy_lasso(lxy_basis_noncycle(x), y)
rmse = np.sqrt(mean_squared_error(n[train_len:], y[train_len:]))
mae = mean_absolute_error(n[train_len:], y[train_len:])
r_square = r2_score(n[train_len:], y[train_len:])
print('rmse:', rmse, ' mae:', mae, ' r2_score:', r_square)
plt.plot(x, y)
plt.plot(x[train_len:], n[train_len:])
plt.savefig('/home/ubuntu/code/Method Comparison/try.png')
'''

