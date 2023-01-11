import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def random_loss(x, train_len):
    x_train = x[:train_len]
    t = range(len(x_train))
    # Lose 5% data information randomly and add Gaussian noise
    loss_loc = np.random.choice(len(x_train), int(len(x_train)*0.05), replace=False)
    loss_set = set(loss_loc)
    loss_set.discard(0)
    loss_set.discard(len(x_train)-1)
    loss_loc = list(loss_set)
    res_loc = list(set(t)^set(loss_loc))
    x_loss = x_train[res_loc]
    x_loss = x_loss + (0.01*(max(x_loss) - min(x_loss)))*np.random.randn(len(x_loss))
    f_cubic = interp1d(res_loc, x_loss, kind='cubic')
    x_new = f_cubic(t)
    return np.hstack((x_new, x[train_len:]))

def flow_data_pre(data_flow, year, t_len):
    x_ori = data_flow['max_mean'].values
    y = []

    for i in range(0,24*t_len,24):
        y.append(np.mean(x_ori[i:i+24]))        # data aggregation
    x = np.array(y)

    pre_len = 4*365
    train_len = t_len-pre_len
    train_rate = train_len/t_len

    t = np.linspace(1, year, t_len)
    mean_x = np.mean(x[:train_len])
    std_x = np.std(x[:train_len])
    x_norm = (x - mean_x) / std_x
    
    x_loss = random_loss(x, train_len)
    mean_x_loss = np.mean(x_loss[:train_len])
    std_x_loss = np.std(x_loss[:train_len])
    x_norm_loss = (x_loss - mean_x_loss) / std_x_loss
    
    # Autoformer data preprocess
    autoformer_year = year + 1
    autoformer_t_len = int((365*autoformer_year+1)/2)
    x_need = np.zeros(autoformer_t_len)
    for i in range(autoformer_t_len):
        x_need[i] = np.mean(x_ori[48*i:48*(i+1)])
    x_need_loss = random_loss(x_need, train_len)
    date = pd.date_range(start='1979-02-01', periods=len(x_need), freq='2D')
    x_need_df = pd.DataFrame(x_need, date, ['max_mean'])
    x_need_df_loss = pd.DataFrame(x_need_loss, date, ['max_mean'])
    x_need_df.to_csv('/home/ubuntu/paper/Autoformer/Autoformer-myself/data/ETT/flow_need.csv')
    x_need_df_loss.to_csv('/home/ubuntu/paper/Autoformer/Autoformer-myself/data/ETT/flow_need_loss.csv')
    
    return t, x, x_norm, mean_x, std_x, train_len, train_rate, x_need_df, x_norm_loss, mean_x_loss, std_x_loss
 

def air_data_pre(air_data, year, t_len, place):
    x = air_data[place, :t_len]
    pre_len = 20*61
    train_len = t_len-pre_len
    train_rate = train_len/t_len

    t = np.linspace(1, year, t_len)
    mean_x = np.mean(x[:train_len])
    std_x = np.std(x[:train_len])
    x_norm = (x - mean_x) / std_x
    
    x_loss = random_loss(x, train_len)
    mean_x_loss = np.mean(x_loss[:train_len])
    std_x_loss = np.std(x_loss[:train_len])
    x_norm_loss = (x_loss - mean_x_loss) / std_x_loss
    
    date = pd.date_range(start='1950-01-01', periods=len(x_norm), freq='6D')
    x_norm_df = pd.DataFrame(x_norm, date, ['temperature'])
    x_norm_loss_df = pd.DataFrame(x_norm_loss, date, ['temperature'])
    x_norm_df.to_csv('/home/ubuntu/paper/Autoformer/Autoformer-myself/data/paper/data_set_2_lossless.csv')
    x_norm_loss_df.to_csv('/home/ubuntu/paper/Autoformer/Autoformer-myself/data/paper/data_set_2_loss.csv')
    
    return t, x, x_norm, mean_x, std_x, train_len, train_rate, x_norm_loss, mean_x_loss, std_x_loss


def finance_data_pre(finance_data, start_date, end_date, t_len, city):
    x = finance_data[city].values
    x = np.hstack((x[(start_date-2)*60*24:(6-2)*60*24], x[(9-2)*60*24:(end_date-2)*60*24]))
    pre_len = 60*24
    train_len = t_len-pre_len
    train_rate = train_len/t_len

    t = np.linspace(0, end_date-start_date-2, t_len)
    mean_x = np.mean(x[:train_len])
    std_x = np.std(x[:train_len])
    x_norm = (x - mean_x) / std_x
    
    x_loss = random_loss(x, train_len)
    mean_x_loss = np.mean(x_loss[:train_len])
    std_x_loss = np.std(x_loss[:train_len])
    x_norm_loss = (x_loss - mean_x_loss) / std_x_loss
    
    date = pd.date_range(start='2021-11-2 00:00:00', periods=len(x_norm), freq='1min')
    x_norm_df = pd.DataFrame(x_norm, date, ['finance'])
    x_norm_loss_df = pd.DataFrame(x_norm_loss, date, ['finance'])
    x_norm_df.to_csv('/home/ubuntu/paper/Autoformer/Autoformer-myself/data/paper/data_set_3_lossless.csv')
    x_norm_loss_df.to_csv('/home/ubuntu/paper/Autoformer/Autoformer-myself/data/paper/data_set_3_loss.csv')
    
    return t, x, x_norm, mean_x, std_x, train_len, train_rate, x_norm_loss, mean_x_loss, std_x_loss


# 
'''
# test for function random_loss
t = np.array(range(365*20))
train_len = 365*16
x = 100*np.sin(t/1000)
plt.plot(t, x)
out = random_loss(x, train_len)
plt.plot(t, out)
plt.savefig('/home/ubuntu/paper/try.png')
'''    

'''
# test for function air_data_pre
air = pd.read_csv('/home/ubuntu/paper/Myself/prior_SI/Data/air data.csv', header=None)
t, x, x_norm, mean_x, std_x, train_len, train_rate, x_norm_loss, mean_x_loss, std_x_loss = air_data_pre(
    air.values, 70, 70*61, 500
)

plt.figure(figsize=(20, 4))
plt.plot(t, x_norm)
plt.savefig('/home/ubuntu/paper/try.png')

print(x.shape)
'''

'''
# test for function finance_data_pre
finance = pd.read_csv('/home/ubuntu/paper/Myself/prior_SI/Data/1min_down.csv', index_col=0)
start_day = 2
end_day = 12
t_len = (end_day-start_day-3)*60*24
t, x, x_norm, mean_x, std_x, train_len, train_rate, x_norm_loss, mean_x_loss, std_x_loss = finance_data_pre(
    finance, start_day, end_day, t_len, 'DC_BJ_agg'
)

plt.figure(figsize=(20, 4))
plt.plot(t, x_norm_loss)
plt.savefig('/home/ubuntu/paper/try.png')
'''

