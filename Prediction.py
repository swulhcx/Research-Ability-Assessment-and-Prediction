import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import random
from random import shuffle
random.seed(0)

import torch
import os, sys
import scipy.io as io
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import pylab
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm
import logging
import time

plt.rc('font', family='Times New Roman')

def mean(arr):
    return sum(arr) / len(arr)

def std_var(arr):
    miu = mean(arr)
    var = sum((ai-miu)**2 for ai in arr)/(len(arr)-1)
    return var**0.5

def MAE(y, y_pred):
    n = len(y)
    return sum(abs(y[i]-y_pred[i]) for i in range(n)) / n

def MAPE(y, y_pred):
    n = len(y)
    return sum(abs(y[i]-y_pred[i])/abs(y[i]) for i in range(n)) / n

def RMSE(y, y_pred):
    n = len(y)
    return (sum((y[i]-y_pred[i])**2 for i in range(n)) / n) ** 0.5

data = pd.read_csv('new.csv')

data['x1'] = data['type']
data['x2'] = data['admission_type']
data['x3'] = data['rank']
data['x4'] = data['grade']
data['x5'] = data['tutor_ability']
data['x6'] = data['tutor_guided_number']
data['x7'] = data['help_gained']
data['x8'] = data['grade']
data['x9'] = data['research_projects_involved']
data['x10'] = data['english']
data['x11'] = data['student_affairs_involved']
data['x12'] = data['student_affairs_duration']
data['x13'] = data['research_time']
data['x14'] = data['research_writing_involved']

data['y'] = data['assessment']

for i in range(39):
    data['x3'][i] = eval(data['x3'][i])
    data['x10'][i] = eval(data['x10'][i])
# print(data['x10'])

# Normalization
idxs = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14']
idy = ['y']

for idx in idxs:
    print(idx)
    m = min(data[idx])
    M = max(data[idx])
    data[idx] = (data[idx] - m) / (M-m)

data = pd.DataFrame(data = data, columns=idxs+idy)
print(data)

train_num = 39 * 7 // 10
test_num = 39 - train_num
l = [i for i in range(39)]
shuffle(l)
train_id = l[:train_num]
test_id = l[train_num:]
# print(train_id)
# print(test_id)
train_data = data.iloc[train_id]
test_data = data.iloc[test_id]
# print(train_data)
# print(test_data)

# 0: multiple regression; 1: SVR; 2: DNN; 3: RF
flag = 3

if flag == 0:
    # 1. multivariate linear regression
    x = sm.add_constant(train_data[idxs])
    y = train_data[idy]
    model = sm. OLS(y,x.astype(float))
    result = model.fit()
    predict_val = result.fittedvalues
    print(result.summary())

    def mul(x):
        coef = [0.7484, -1.0230, -0.0137, 0.8136, 0.1542, -0.9444, -0.0051, 0.8136, 0.4251, -0.5050, -0.5567, 1.0366, -0.0163, -0.1722]
        return np.dot(np.array(coef), np.array(x))+0.4149

    y_pred = []
    print(test_data)
    for i in range(test_num):
        # print(test_data.iloc[i,:14])
        y_pred.append(mul(np.array(test_data.iloc[i,:14])))
    # print(type(list(test_data['y'])))
    # print(type(y_pred))
    mae = MAE(list(test_data['y']), y_pred)
    mape = MAPE(list(test_data['y']), y_pred)
    rmse = RMSE(list(test_data['y']), y_pred)
    print(mae, mape, rmse)

elif flag == 1:
    # 2. STM
    clf = SVR(kernel='rbf', C = 10)
    clf.fit(train_data[idxs],train_data[idy])
    y_pred = clf.predict(test_data.iloc[:,:14])
    mae = MAE(list(test_data['y']), y_pred)
    mape = MAPE(list(test_data['y']), y_pred)
    rmse = RMSE(list(test_data['y']), y_pred)
    print(mae, mape, rmse)

elif flag == 2:
    # 3. DNN
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(14, 50)
            self.fc2 = nn.Linear(50, 10)
            self.fc3 = nn.Linear(10, 3)
            self.fc4 = nn.Linear(3, 1)
            self.sigmod = nn.Sigmoid()
            self.logsoftmax = nn.LogSoftmax()
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            out = self.sigmod(self.fc1(x))
            out = self.sigmod(self.fc2(out))  #
            out = self.sigmod(self.fc3(out))  #
            out = self.sigmod(self.fc4(out))  #

            return out

    model = Net()
    model = model
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')

    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(train_num):
        X_train.append(train_data.iloc[i][:14])
        Y_train.append(train_data.iloc[i][14])
    for i in range(test_num):
        X_test.append(test_data.iloc[i][:14])
        Y_test.append(test_data.iloc[i][14])
    f = lambda x: torch.tensor(np.array(x).astype(float)).to(torch.float)
    X_train = f(X_train)
    Y_train = f(Y_train)
    X_test = f(X_test)
    Y_test = f(Y_test)

    Max_epoch = 1000
    for epoch in range(Max_epoch):
        running_loss_train = 0.0
        running_loss_test = 0.0
        n = len(test_data)

        for t, x in tqdm(enumerate(X_train)):
            data, target = Variable(x), Variable(Y_train[t])
            # b_x,b_y,b_y1 = data.cuda(),target1.cuda(),target2.cuda()
            b_x, b_y1 = data, target
            b_x = b_x.float()
            model.train()
            optimizer.zero_grad()
            out = model(b_x)
            loss = loss_fn(out, b_y1)
            # running_loss += loss.item() * b_y1.size(0)
            running_loss_train += abs(loss.item())
            # print(num_correct1, num_correct2)
            # accuracy = (pred2 == b_y2).float().mean()
            # running_acc1_train += num_correct1_train.item()
            # running_acc2_train += num_correct2_train.item()
            # print(running_acc1, running_acc2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for t, x in tqdm(enumerate(X_test)):
            data, target = Variable(x), Variable(Y_test[t])
            # b_x,b_y,b_y1 = data.cuda(),target1.cuda(),target2.cuda()
            b_x, b_y1 = data, target
            b_x = b_x.float()
            out = model(b_x)
            loss = loss_fn(out, b_y1)
            running_loss_test += abs(loss.item())
        if epoch % 10 == 0:
            print('*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)
            print('MSE in training dataset: {}'.format((running_loss_train / len(train_data)) ** 0.5))
            print('MSE in testing dataset: {}'.format((running_loss_test / len(test_data)) ** 0.5))
    y_pred = []
    for t, x in tqdm(enumerate(X_test)):
        data, target = Variable(x), Variable(Y_test[t])
        # b_x,b_y,b_y1 = data.cuda(),target1.cuda(),target2.cuda()
        b_x, b_y1 = data, target
        b_x = b_x.float()
        out = model(b_x)
        y_pred.append(out.float())
    mae = MAE(list(test_data['y']), y_pred)
    mape = MAPE(list(test_data['y']), y_pred)
    rmse = RMSE(list(test_data['y']), y_pred)
    print(mae, mape, rmse)

elif flag == 3:
    # 4. Random Forest
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(train_num):
        X_train.append(train_data.iloc[i][:14])
        Y_train.append(train_data.iloc[i][14])
    for i in range(test_num):
        X_test.append(test_data.iloc[i][:14])
        Y_test.append(test_data.iloc[i][14])
    # f = lambda x: torch.tensor(np.array(x).astype(float)).to(torch.float)
    # X_train = f(X_train)
    # Y_train = f(Y_train)
    # X_test = f(X_test)
    # Y_test = f(Y_test)

    RF = RandomForestRegressor(n_estimators=2000, random_state=0)
    RF.fit(X_train, Y_train)
    y_pred = RF.predict(X_test)
    mae = MAE(list(test_data['y']), y_pred)
    mape = MAPE(list(test_data['y']), y_pred)
    rmse = RMSE(list(test_data['y']), y_pred)
    print(mae, mape, rmse)