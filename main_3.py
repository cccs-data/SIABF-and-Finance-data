#%% import and function module
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import time
import os
import sys
need_path = '/home/ubuntu/paper/Myself/prior_SI'
if os.path.abspath('.') != need_path:
    os.chdir(need_path)
sys.path.append(need_path)

import Utils.Flow_data_pre as data_pre
import Utils.Evaluation as Eval
import Utils.Figure_maker as Fig
import Methods.Sparse_Identification as SI
import Methods.ARIMA as Arima
import Methods.GRU_Net as GRU
import Methods.LSTM_Net as LSTM
import Methods.ResNet as Resnet

def EXP_flow(name, model, train_len, t_len, x_norm, mean_x, std_x, x, print_1, Time_compute=True):
    gpu_time = pd.read_csv('./Save_data/gpu_time.csv', index_col=0)
    if name+'.npy' not in os.listdir(need_path+'/Save_data/Data-set_3'):
        start_time = time.time()
        pred = eval(model)
        end_time = time.time()
        if Time_compute:
            gpu_time[name.split('_')[-1]]['Data-set_3'] = end_time - start_time
            gpu_time.to_csv('./Save_data/gpu_time.csv')     # Saving time
        np.save('./Save_data/Data-set_3/'+name, pred)
    else:
        pred = np.load('./Save_data/Data-set_3/'+name+'.npy')
    rmse, mae, r2, mape, mae_per = Eval.eval(
        pred[train_len:t_len], x_norm[train_len:t_len], mean_x, std_x, x[train_len:t_len])  # Evaluation
    # print(print_1)
    print_name = name if Time_compute else name[:-5]
    '''
    print('RMSE:', '%.3f' %rmse, ' MAE:', '%.3f' %mae, 'r_2: ', '%.3f' %r2, ' MAPE_original:', '%.3f' % (mape*100), '%', 
      ' MAE%:', '%.3f' % (mae_per*100), '%', ' run time:',  '%.3f' %gpu_time[print_name.split('_')[-1]]['Data-set_3'], 's')
    '''
    # the print below is only for paper
    if Time_compute:
        print('&', name.split('_')[-1], '&', '%.3f' %rmse, '&', '%.3f' %mae, '&', '%.3f' %r2, '&', '%.3f' %(mape*100))
    else:
        print('&', name.split('_')[-2], '&', '%.3f' %rmse, '&', '%.3f' %mae, '&', '%.3f' %r2, '&', '%.3f' %(mape*100), '&', '%.3f' %gpu_time[print_name.split('_')[-1]]['Data-set_3'], '\\\\')
    
    return pred


#%%  Data Processing

data_flow = pd.read_csv('/home/ubuntu/paper/Myself/prior_SI/Data/1min_down.csv', index_col=0)
# print(data_flow.head())     # See data

start_day = 2
end_day = 12
t_len = (end_day-start_day-3)*60*24
place = 'DC_BJ_agg'     # 495
t, x, x_norm, mean_x, std_x, train_len, train_rate, x_norm_loss, mean_x_loss, std_x_loss = data_pre.finance_data_pre(
    data_flow, start_day, end_day, t_len, 'DC_BJ_agg'
)

exp_name = ['finance_pre_SIPBF', 'finance_pre_ARIMA', 'finance_pre_SBL', 'finance_pre_SINDy', 
            'finance_pre_LSTM', 'finance_pre_GRU', 'finance_pre_ResNet', 'finance_pre_Autoformer']
exp_name_loss = [x+'_loss' for x in exp_name]
if 'gpu_time.csv' not in os.listdir(need_path+'/Save_data'):
    gpu_time = pd.DataFrame(np.zeros((3, 7)), index=['Data-set_1', 'Data-set_2', 'Data-set_3'], columns=exp_name)
    gpu_time.to_csv('./Save_data/gpu_time.csv')


#%% EXP: Data-set 2(Finance prediction)
x_pre_SI = EXP_flow(
    exp_name[0], 'SI.lxy_lasso(SI.lxy_basis_cycle(t,x_norm,max_k=30), x_norm, train_rate=train_rate, def_alpha=5e-4, it=1000, norm=True)', 
    train_len, t_len,  x_norm, mean_x, std_x, x, 'Finance Data Predict(prior SI):'
)


# EXP: ARIMA
x_pre_ARIMA = EXP_flow(
    exp_name[1], 'Arima.lxy_ARIMA(x_norm, train_len)', 
    train_len, t_len,  x_norm, mean_x, std_x, x, 'Finance Data Predict(ARIMA):'
)



# EXP: SBM
str_SBL = '''SI.lxy_SBL(SI.lxy_basis_cycle(t,x_norm,max_k=30), x_norm, train_rate=train_rate
, def_alpha_1=1e-8, def_alpha_2=5e-5, def_lambda_1=1e-8, def_lambda_2=5e-5
, thro=10, tole=0.1, it=10000, norm=True)
'''
x_pre_SBL = EXP_flow(
    exp_name[2], str_SBL
    , train_len, t_len,  x_norm, mean_x, std_x, x, 'Finance Data Predict(prior SBL):'
)


# EXP: SINDy
x_pre_SINDy = EXP_flow(
    exp_name[3], 'SI.my_SINDy(SI.lxy_basis_noncycle(t), x_norm, lammbda=0.01)'
    , train_len, t_len,  x_norm, mean_x, std_x, x, 'Finance Data Predict(prior SINDy):'
)


# EXP: LSTM
x_pre_LSTM = EXP_flow(
    exp_name[4], 'LSTM.lxy_LSTM_train(x_norm, ep=300, train_net_len=500, pred_net_len=250, hidden_s=100, train_rate=train_rate, learn_rate=1e-3, hidden_l=2)', 
    train_len, t_len,  x_norm, mean_x, std_x, x, 'Finance Data Predict(LSTM):'
)

# EXP: GRU
x_pre_GRU = EXP_flow(
    exp_name[5], 'GRU.lxy_GRU_train(x_norm, ep=200, train_net_len=500, pred_net_len=250, hidden_s=100, train_rate=train_rate, learn_rate=1e-3, hidden_l=2)', 
    train_len, t_len,  x_norm, mean_x, std_x, x, 'Finance Data Predict(GRU):'
)

# EXP: ResNet
x_pre_ResNet = EXP_flow(
    exp_name[6], 'Resnet.lxy_ResNet_train(x_norm, ep=200, train_net_len=500, pred_net_len=150, train_rate=train_rate, learn_rate=1e-3)', 
    train_len, t_len,  x_norm, mean_x, std_x, x, 'Finance Data Predict(ResNet):'
)

'''
# EXP: Autoformer
x_pre_Autoformer = EXP_flow(
    exp_name[7], ' ', 
    train_len, t_len,  x_norm, mean_x, std_x, x, 'Finance Data Predict(Autoformer):'
)
'''




#%% EXP: Data-set 2 with incomplete and noisy(Finance prediction)
x_pre_SI_loss = EXP_flow(
    exp_name_loss[0], 'SI.lxy_lasso(SI.lxy_basis_cycle(t,x_norm_loss,max_k=30), x_norm_loss, train_rate=train_rate, def_alpha=5e-4, it=10000, norm=True)', 
    train_len, t_len, x_norm_loss, mean_x_loss, std_x_loss, x, 'Finance Data Predict Loss(prior SI):', False
)

# EXP: ARIMA
x_pre_ARIMA_loss = EXP_flow(
    exp_name_loss[1], 'Arima.lxy_ARIMA(x_norm_loss, train_len)', 
    train_len, t_len, x_norm_loss, mean_x_loss, std_x_loss, x, 'Finance Data Predict Loss(ARIMA):', False
)

# EXP: SBL
str_SBL_loss = '''SI.lxy_SBL(SI.lxy_basis_noncycle(t), x_norm_loss, train_rate=train_rate
, def_alpha_1=1e-8, def_alpha_2=5e-5, def_lambda_1=1e-8, def_lambda_2=5e-5
, thro=10, tole=0.1, it=10000, norm=True)
'''
x_pre_SBL_loss = EXP_flow(
    exp_name_loss[2], str_SBL_loss
    , train_len, t_len, x_norm_loss, mean_x_loss, std_x_loss, x, 'Finance Data Predict Loss(prior SBL):', False
)

# EXP: SINDy
x_pre_SINDy_loss = EXP_flow(
    exp_name_loss[3], 'SI.my_SINDy(SI.lxy_basis_noncycle(t), x_norm_loss, lammbda=0.01)'
    , train_len, t_len, x_norm_loss, mean_x_loss, std_x_loss, x, 'Finance Data Predict Loss(prior SINDy):', False
)

# EXP: LSTM
x_pre_LSTM_loss = EXP_flow(
    exp_name_loss[4], 'LSTM.lxy_LSTM_train(x_norm_loss, ep=300, train_net_len=500, pred_net_len=250, hidden_s=100, train_rate=train_rate, learn_rate=1e-3, hidden_l=2)', 
    train_len, t_len, x_norm_loss, mean_x_loss, std_x_loss, x, 'Finance Data Predict Loss(LSTM):', False
)

# EXP: GRU
x_pre_GRU_loss = EXP_flow(
    exp_name_loss[5], 'GRU.lxy_GRU_train(x_norm_loss, ep=200, train_net_len=500, pred_net_len=250, hidden_s=100, train_rate=train_rate, learn_rate=1e-3, hidden_l=2)', 
    train_len, t_len,x_norm_loss, mean_x_loss, std_x_loss, x, 'Finance Data Predict Loss(GRU):', False
)

# EXP: ResNet
x_pre_ResNet_loss = EXP_flow(
    exp_name_loss[6], 'Resnet.lxy_ResNet_train(x_norm_loss, ep=200, train_net_len=500, pred_net_len=150, train_rate=train_rate, learn_rate=1e-3)', 
    train_len, t_len,x_norm_loss, mean_x_loss, std_x_loss, x, 'Finance Data Predict Loss(ResNet):', False
)


'''
# EXP: Autoformer
x_pre_Autoformer_loss = EXP_flow(
    exp_name_loss[7], ' ', 
    train_len, t_len, x_norm_loss, mean_x_loss, std_x_loss, x, 'Finance Data Predict Loss(Autoformer):', False
)
'''



#%% Figure Appendix
predict_len = range(len(t[train_len:t_len]))
predict_name = ['SIPBF', 'ARIMA', 'SBL', 'SINDy', 'LSTM', 'GRU', 'ResNet']
predict_method = [x_pre_SI, x_pre_ARIMA, x_pre_SBL, x_pre_SINDy,
                  x_pre_LSTM, x_pre_GRU, x_pre_ResNet]
n_pic = len(predict_name)
plt.figure(figsize=(15,35))
for i in range(n_pic):
    Fig.Figure_Appendx(n_pic, i+1, predict_len, x_norm, predict_method[i], train_len, t_len, predict_name[i])

plt.savefig('/home/ubuntu/paper/try.png')
# plt.savefig('./Figure/Appendix.png', bbox_inches='tight', pad_inches=0.05)
# plt.savefig('./Figure/Appendix.pdf', bbox_inches='tight', pad_inches=0.05)
# plt.savefig('./Figure/Appendix.svg', bbox_inches='tight', pad_inches=0.05)

'''
plt.figure(figsize=[12,4])
plt.plot(t[train_len:t_len], x_norm[train_len:t_len])
plt.plot(t[train_len:t_len], x_pre_SI[train_len:t_len], label='Sparse Identification')
plt.plot(t[train_len:t_len], x_pre_SBL[train_len:t_len], label='SBL')
plt.plot(t[train_len:t_len], x_pre_ARIMA[train_len:t_len], label='ARIMA')
plt.plot(t[train_len:t_len], x_pre_LSTM[train_len:t_len], label='LSTM')
plt.plot(t[train_len:t_len], x_pre_GRU[train_len:t_len], label='GRU')
plt.plot(t[train_len:t_len], x_pre_ResNet[train_len:t_len], label='ResNet')
plt.plot(t[train_len:t_len], x_pre_Autoformer[train_len:t_len], label='Autoformer')
plt.plot(t[train_len:t_len], x_pre_SINDy[train_len:t_len], label='SINDy')
plt.legend()
plt.savefig('./Figure/Flow_Prior_SI.png')
'''

