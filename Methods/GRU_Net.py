import torch
from torch import nn
from time import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

class MyGRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(MyGRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        # self.seq_len = seq_len
        self.pred_len = pred_len
        '''
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, output_size*3),
            nn.ReLU(),
            nn.Linear(output_size*3, output_size)
            )
        '''
        
    def forward(self, x):
        y, _ = self.gru(x)
        y = y[:, -1, :]
        # y = y.squeeze()
        y = self.fc(y)
        y = y.view(-1, self.pred_len, 1)
        return y

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

'''
my_net = MyGRUNet(input_size=1, hidden_size=10, num_layers=3, seq_len=10000, pred_len=100).to(device)
x = torch.rand(10, 10000, 1).to(device)    # banch * seq_len * input_size
y = my_net.forward(x)       # banch * seq_len * hidden_size
print(x.shape)
print(y.shape)
'''

def lxy_GRU_train(x, train_rate=0.7, train_net_len=100, pred_net_len=20, ep=500, learn_rate=1e-3, batch_size=50, hidden_s=50, hidden_l=3):
    train_len = int(train_rate*len(x))
    x_max = max(x[:train_len])
    x_min = min(x[:train_len])
    x_max_min = (x - x_min) / (x_max - x_min)
    x_max_min_torch = torch.tensor(x_max_min, dtype=torch.float).to(device)
    x_need_pred = torch.zeros(x.shape).to(device)
    x_need_pred[:train_len] = x_max_min_torch[:train_len]
    # print('x_max_min\'s shape: ', x_max_min.shape)

    def create_dataset(def_data, def_time_step=train_net_len, def_pred_step=pred_net_len):
        arr_x, arr_y = [], []
        for i in range(0, len(def_data) - def_time_step - def_pred_step):
            x = def_data[i: i + def_time_step]
            y = def_data[i + def_time_step: i + def_time_step + def_pred_step]
            arr_x.append(x)
            arr_y.append(y)
        return np.array(arr_x), np.array(arr_y)
    
    X, Y = create_dataset(x_max_min)
    X = torch.tensor(X.reshape(-1, train_net_len, 1), dtype=torch.float).to(device)
    Y = torch.tensor(Y.reshape(-1, pred_net_len, 1), dtype=torch.float).to(device)
    X_train, Y_train = X[:train_len-train_net_len-pred_net_len, :, :], Y[:train_len-train_net_len-pred_net_len, :, :]
    # print('Total datasets: ', X.shape, '-->', Y.shape)
    # print('Train datasets: ', X_train.shape, '-->', Y_train.shape)
    # print(X_train[50,:,:])
    # print(Y_train[50,:,:])

    
    ds = TensorDataset(X, Y)
    ds_train = TensorDataset(X_train, Y_train)
    dl_train = DataLoader(ds_train, batch_size=batch_size)
    model = MyGRUNet(1, hidden_s, hidden_l, pred_net_len).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    def train_step(model, features, labels):
        # 正向传播求损失
        predictions = model.forward(features)
        loss = loss_function(predictions, labels)
        # 反向传播求梯度
        loss.backward()
        # 参数更新
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    def train_model(model, epochs):
        for epoch in range(1, epochs+1):
            list_loss = []
            for features, labels in dl_train:
                loss_i = train_step(model, features, labels)
                list_loss.append(loss_i)
            loss = np.mean(list_loss)
            if epoch % 5 == 0:
                print('epoch={} | loss={} '.format(epoch,loss))
            if loss < 1e-4:
                print('epoch={} | loss={} (advance break)'.format(epoch,loss))
                break

    train_model(model, ep)
    # print(train_len, train_net_len, pred_net_len)
    # print(train_len-train_net_len-pred_net_len, train_len-pred_net_len)
    for i in range(len(x) - train_len):
        temp = x_need_pred[train_len-train_net_len-pred_net_len+i:train_len-pred_net_len+i]
        # print(train_len-train_net_len-pred_net_len+i, train_len-pred_net_len+i)
        temp = temp.view(1, -1, 1)
        temp = model.forward(temp).detach().squeeze()
        x_need_pred[train_len+i] = temp[-1]
    # print(x_need_pred)
    x_need_pred = x_need_pred.cpu().numpy().squeeze()
    x_need_pred = x_need_pred * (x_max - x_min) +  x_min
    return x_need_pred
    

'''
length = 200
rate = 0.9
train_length = int(length*rate)
t = np.linspace(1,1000,length)
x = np.sin(np.pi*t/100)
y = lxy_GRU_train(x, train_rate=rate, ep=1000, learn_rate=5e-2)

plt.plot(t, x)
plt.plot(t, y)
plt.savefig('/home/ubuntu/code/Method Comparison/try4.png')

print(y.shape)
print(y)

'''








