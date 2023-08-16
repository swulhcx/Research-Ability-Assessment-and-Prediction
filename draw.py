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

y_real = list(test_data['y'])
# Y = [[y_real[i], y_pred[i]] for i in range(test_num)]
# Y.sort()
# y_real, y_pred = zip(*Y)
plt.plot(range(1,test_num+1), y_real, label = 'original')
plt.plot(range(1,test_num+1), y_pred, label = 'prediction')
plt.legend()
plt.xlabel('Item')
plt.ylabel('Research Ability')
# plt.show()
plt.savefig('real_pred.png', dpi=600)