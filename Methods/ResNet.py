import torch
from torch import nn
from time import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader


class MyResNet(nn.Module):
    def __init__(self, time_len, pre_len):
        super(MyResNet, self).__init__()
        self.mask = nn.Sequential(
            nn.Linear(time_len, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, time_len)
        )
        self.fc = nn.Sequential(
            nn.Linear(time_len, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, pre_len)
        )
    def forward(self, x):
        y = self.mask(x)
        y = y+x
        y = self.fc(y)
        return y

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

'''
my_net = MyResNet(200,100).to(device)
x = torch.rand(200).to(device)
y = my_net.forward(x).to(device)
print(x.shape)
print(y.shape)
'''

def lxy_ResNet_train(x, train_rate=0.7, train_net_len=200, pred_net_len=100, ep=500, learn_rate=1e-2):
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
    X = torch.tensor(X.reshape(-1, 1, train_net_len), dtype=torch.float).to(device)
    Y = torch.tensor(Y.reshape(-1, 1, pred_net_len), dtype=torch.float).to(device)
    X_train, Y_train = X[:train_len-train_net_len-pred_net_len, :, :], Y[:train_len-train_net_len-pred_net_len, :, :]
    # print('Total datasets: ', X.shape, '-->', Y.shape)
    # print('Train datasets: ', X_train.shape, '-->', Y_train.shape)

    batch_size = 40
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=batch_size)
    ds_train = TensorDataset(X_train, Y_train)
    dl_train = DataLoader(ds_train, batch_size=batch_size)
    model = MyResNet(train_net_len, pred_net_len).to(device)
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
            if epoch % 50 == 0:
                print('epoch={} | loss={} '.format(epoch,loss))
    
    train_model(model, ep)
    # print(len(x) - train_len)
    for i in range(len(x) - train_len):
        temp = x_need_pred[train_len-train_net_len-pred_net_len+i:train_len-pred_net_len+i]
        # print(train_len-train_net_len-pred_net_len+i, train_len-pred_net_len+i)
        temp = temp.view(1, 1, -1)
        temp = model.forward(temp).detach().squeeze()
        x_need_pred[train_len+i] = temp[-1]
    # print(x_need_pred)
    x_need_pred = x_need_pred.cpu().numpy().squeeze()
    x_need_pred = x_need_pred * (x_max - x_min) +  x_min
    return x_need_pred

'''
x = np.linspace(1,1000,1000)
y = lxy_BP_train(x)
print(x)
print(y)
'''

